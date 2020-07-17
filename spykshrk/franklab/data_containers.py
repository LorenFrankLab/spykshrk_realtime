import numpy as np
import pandas as pd
import functools
from abc import ABC, ABCMeta, abstractclassmethod, abstractmethod
from itertools import product
import uuid
import enum
import os
import warnings
import traceback
import pickle

from spykshrk.franklab import print_util

from spykshrk.franklab.warnings import ConstructorWarning, OverridingAttributeWarning
from spykshrk.franklab.errors import HDFKeyExistsError

from spykshrk.franklab.pp_decoder.util import gaussian
from spykshrk.util import AttrDict, EnumMapping



class UnitTime(EnumMapping):
    SEC = 1
    SECOND = 1
    SAMPLE = 2
    MS = 3
    MILLISECOND = 3


def partialclass(cls, parent_class=None, *args, **kwds):

    if parent_class is not None:

        NewCls = type('_' + cls.__name__, (cls,),
                      {'__init__': functools.partialmethod(cls.__init__, *args, **kwds),
                       '__module__': __name__,
                       '_parent_class': parent_class})
    else:
        NewCls = type('_' + cls.__name__, (cls,),
                      {'__init__': functools.partialmethod(cls.__init__, *args, **kwds),
                       '__module__': __name__,
                       '_parent_class': cls})

    return NewCls


class DataFormatError(RuntimeError):
    pass


def pos_col_format(ind, num_bins):
    return 'x{:0{dig}d}'.format(ind, dig=len(str(num_bins)))


class SeriesClass(pd.Series):
    _metadata = pd.Series._metadata + ['history', 'kwds']

    def __init__(self, data=None, index=None, dtype=None, name=None,
                 copy=False, fastpath=False, history=None, **kwds):
        super().__init__(data=data, index=index, dtype=dtype, name=name, copy=copy, fastpath=fastpath)
        self.history = history
        self.kwds = kwds

    @property
    def _constructor(self):
        cls = partialclass(type(self), history=self.history, **self.kwds)
        return cls

    @property
    def _constructor_expanddim(self):
        # TODO: DataFrameClass is abstract, shouldn't be created
        return partialclass(DataFrameClass, history=self.history, **self.kwds)


class DataFrameClass(pd.DataFrame):

    _metadata = pd.DataFrame._metadata + ['kwds', 'history', 'desc', 'user_key']
    _internal_names = pd.DataFrame._internal_names + ['uuid']
    _internal_names_set = set(_internal_names)

    def __init__(self, data=None, index=None, columns=None, dtype=None, copy=False, parent=None, history=None,
                 custom_uuid=None, user_key=None, desc='', **kwds):
        """
        
        Args:
            data: 
            index: 
            columns: 
            dtype: 
            copy: 
            parent: Uses parent history if avaliable.
            history: List that is set as the history of this object.  
                Overrides both the history of the parent and data source.
            **kwds: 
        """

        if custom_uuid is None:
            self.uuid = uuid.uuid4()
        else:
            self.uuid = custom_uuid

        self.kwds = kwds

        self.user_key = user_key
        self.desc = desc

        if history is not None:
            self.history = list(history)
        elif parent is not None:
            if hasattr(parent, 'history'):
                self.history = list(parent.history)
                self.history.append(parent)
            else:
                self.history = [parent]
        else:
            if hasattr(data, 'history'):
                self.history = list(data.history)
                self.history.append(data)
            else:
                self.history = [data]
        #self.history.append(self)

        if '_parent_class' not in type(self).__dict__:
            setattr(type(self), '_parent_class', self.__class__)

        super().__init__(data, index, columns, dtype, copy)

    def __reduce__(self):
        parent_class = type(self).__dict__['_parent_class']
        return (parent_class, (self.to_dict(), None, None, None, None, False, self.history,
                               self.uuid, self.user_key, self.desc, self.kwds))

    def __setstate__(self, state):
        # Insert new uuid (internal value)
        self.uuid = uuid.uuid4()
        super().__setstate__(state)

    #def __setstate__(self, state):
    #    #self.__init__(data=state['_data'], history=state['history'], kwds=state['kwds'])
    #    self.__dict__.update(state.copy())

    # def __getstate__(self):
    #     state = self.__dict__.copy()
    #     return state

    @property
    def _constructor(self):
        if hasattr(self, 'history'):
            return partialclass(type(self), parent_class=self._parent_class, history=self.history, **self.kwds)
        else:
            return type(self, **self.kwds)

    @property
    def _constructor_sliced(self):
        return SeriesClass

    @property
    def _constructor_expanddim(self):
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def create_default(cls, df, parent=None, **kwd):
        pass

    def to_dataframe(self):
        return pd.DataFrame(self)

    def _to_hdf_store(self, file_path, hdf_base, hdf_grps, hdf_label, overwrite=False):
        with pd.HDFStore(file_path, 'a') as store:
            main_path = os.path.join(hdf_base, hdf_grps, hdf_label)
            if (main_path in store.keys()) and (not overwrite):
                raise HDFKeyExistsError('HDF key {} exists in file {}. '
                                        'Either rename key or set overwrite flag'.format(main_path, file_path))
            store.put(main_path, self.to_dataframe(), format='t')
            #store[main_path] = self.to_dataframe()
            save_history = []
            for hist_en in self.history:
                try:
                    save_history.append((type(hist_en), hist_en.uuid))
                except AttributeError:
                    save_history.append((type(hist_en), repr(hist_en)))

            main_storer = store.get_storer(main_path)
            main_storer.attrs.history = save_history
            main_storer.attrs.kwds = self.kwds
            main_storer.attrs.uuid = self.uuid

            main_storer.attrs.user_key = self.user_key

            main_storer.attrs.classtype = type(self._parent_class)

    @classmethod
    def _from_hdf_store(cls, file_path, hdf_base, hdf_grps, hdf_label):
        with pd.HDFStore(file_path, 'r') as store:
            main_path = os.path.join(hdf_base, hdf_grps, hdf_label)
            dataframe = store[main_path]
            main_storer = store.get_storer(main_path)
            save_history = main_storer.attrs.history
            kwds = main_storer.attrs.kwds
            custom_uuid = main_storer.attrs.uuid
            try:
                user_key = main_storer.attrs.user_key
            except AttributeError:
                user_key = None

            newcls = main_storer.attrs.classtype

            return newcls(data=dataframe, history=save_history, custom_uuid=custom_uuid, user_key=user_key, **kwds)

    def __repr__(self):
        return '<{}: {}, shape: ({})>'.format(self.__class__.__name__, self.uuid, self.shape)


class DayDataFrame(DataFrameClass):
    def __init__(self, data=None, index=None, columns=None, dtype=None, copy=False, parent=None, history=None,
                 custom_uuid=None, user_key=None, desc='', **kwds):

        if isinstance(data, pd.DataFrame):

            if not ('day' in data.index.names):
                raise DataFormatError("DayEpochTimeSeries must have index named day.")

            # Make sure days are integers
            day_level = data.index.names.index('day')
            data.index.set_levels(data.index.levels[day_level].astype('int'), level=day_level, inplace=True)

        super().__init__(data=data, index=index, columns=columns, dtype=dtype,
                         copy=copy, parent=parent, history=history, custom_uuid=custom_uuid,
                         user_key=user_key, desc=desc, **kwds)


class DayEpochEvent(DayDataFrame):
    _metadata = DayDataFrame._metadata + ['time_unit']

    def __init__(self, data=None, index=None, columns=None, dtype=None, copy=False, parent=None, history=None,
                 custom_uuid=None, user_key=None, desc='', **kwds):
        self.time_unit = kwds['time_unit']  # type: UnitTime

        if isinstance(data, pd.DataFrame):
            if not isinstance(data.index, pd.MultiIndex):
                raise DataFormatError("DataFrame index must use MultiIndex as index.")

            if not all([col in data.index.names for col in ['day', 'epoch', 'event']]):
                raise DataFormatError("DayEpochTimeSeries must have index with 3 levels named: "
                                      "day, epoch, event.")

            if not all([col in data.columns for col in ['starttime', 'endtime']]):
                raise DataFormatError("RippleTimes must contain columns 'starttime' and 'endtime'.")

            # Make sure epochs are integers
            epoch_level = data.index.names.index('epoch')
            data.index.set_levels(data.index.levels[epoch_level].astype('int'), level=epoch_level, inplace=True)

        super().__init__(data=data, index=index, columns=columns, dtype=dtype,
                         copy=copy, parent=parent, history=history, custom_uuid=custom_uuid,
                         user_key=user_key, desc=desc, **kwds)

        if index is not None and not isinstance(index, pd.MultiIndex):
            raise DataFormatError("Index to be set must be MultiIndex.")

    def get_range_view(self):
        return self[['starttime', 'endtime']]

    def get_num_events(self):
        return len(self)

    def find_events(self, times):
        event_id_list = []
        for time in times:
            res = self.query('starttime < @time and endtime > @time')
            id_list = res.index.get_level_values('event')
            event_id_list.extend(id_list)

        return event_id_list


class DayEpochTimeSeries(DayDataFrame):

    _metadata = DayDataFrame._metadata + ['sampling_rate']

    def __init__(self, data=None, index=None, columns=None, dtype=None, copy=False, parent=None, history=None,
                 custom_uuid=None, user_key=None, desc='', sampling_rate=0, **kwds):

        if isinstance(data, DayEpochTimeSeries):
            pass
        elif isinstance(data, DataFrameClass):
            warnings.warn(ConstructorWarning('Trying to create a DataEpochTimeSeries from a {}. '
                                             'Not guarenteed to be compatible'.format(kwds['data'].__name__)))
        self.sampling_rate = sampling_rate
        kwds['sampling_rate'] = self.sampling_rate

        if isinstance(data, pd.DataFrame):
            if not isinstance(data.index, pd.MultiIndex):
                raise DataFormatError("DataFrame index must use MultiIndex as index.")

            if not all([col in data.index.names for col in ['day', 'epoch', 'timestamp', 'time']]):
                raise DataFormatError("DayEpochTimeSeries must have index with 4 levels named: "
                                      "day, epoch, timestamp, time.")

            # Make sure epochs and timestamps are integers
            epoch_level = data.index.names.index('epoch')
            data.index.set_levels(data.index.levels[epoch_level].astype('int'), level=epoch_level, inplace=True)

            ts_level = data.index.names.index('timestamp')
            data.index.set_levels(data.index.levels[ts_level].astype('int'), level=ts_level, inplace=True)

            epoch_grps = data.groupby(['day', 'epoch'])
            kwds['start_times'] = []
            kwds['start_timestamps'] = []

            for key, epoch_data in epoch_grps:
                kwds['start_times'].append(epoch_data.index.get_level_values('time')[0])
                kwds['start_timestamps'].append(epoch_data.index.get_level_values('timestamp')[0])

        if index is not None and not isinstance(index, pd.MultiIndex):
            raise DataFormatError("Index to be set must be MultiIndex.")

        super().__init__(data=data, index=index, columns=columns, dtype=dtype,
                         copy=copy, parent=parent, history=history, custom_uuid=custom_uuid,
                         user_key=user_key, desc=desc, **kwds)

    @classmethod
    @abstractclassmethod
    def create_default(cls, df, sampling_rate, arm_coord=None, parent=None, **kwds):
        pass

    def get_time(self):
        return self.index.get_level_values('time')

    def get_timestamp(self):
        return self.index.get_level_values('timestamp')

    def get_timestamp_range(self):
        return [self.index.get_level_values('timestamp')[0], self.index.get_level_values('timestamp')[-1]]

    def get_timestamp_start(self):
        return self.index.get_level_values('timestamp')[0]

    def get_timestamp_end(self):
        return self.index.get_level_values('timestamp')[-1]

    def get_timestamp_total(self):
        return self.get_timestamp_end() - self.get_timestamp_start()

    def get_time_range(self):
        return [self.index.get_level_values('time')[0], self.index.get_level_values('time')[-1]]

    def get_time_start(self):
        return self.index.get_level_values('time')[0]

    def get_time_end(self):
        return self.index.get_level_values('time')[-1]

    def get_time_total(self):
        return self.get_time_end() - self.get_time_start()

    def get_relative_index(self, inplace=False):
        if inplace:
            ret_data = self
            ind = self.index
        else:
            ret_data = self.copy()
            ind = ret_data.index
        ind.set_levels(level='time', levels=(self.index.get_level_values('time') -
                                             self.index.get_level_values('time')[0]), inplace=True)
        ind.set_levels(level='timestamp', levels=(self.index.get_level_values('timestamp') -
                                                  self.index.get_level_values('timestamp')[0]), inplace=True)

        return ret_data

    def get_days(self):
        return self.index.get_level_values('day').unique()

    def update_default_bins(self, time_bin_size, inplace=False):
        if inplace:
            df = self
        else:
            df = self.copy()

        self.default_bin_size = time_bin_size

        bins = np.floor((self.index.get_level_values('timestamp') -
                         self.index.get_level_values('timestamp')[0]) / time_bin_size).astype('int')
        bins_start = (int(self.index.get_level_values('timestamp')[0] / time_bin_size) *
                          time_bin_size + bins * time_bin_size)
        df['bin'] = bins
        df['bin_start'] = bins_start

        #df.update_num_missing_future_bins(inplace=True)
        return df

    def get_resampled(self, bin_size):
        """
        
        Args:
            bin_size: size of time 
7
        Returns (LinearPositionContainer): copy of self with times resampled using backfill.

        """

        def epoch_rebin_func(df):
            day = df.index.get_level_values('day')[0]
            epoch = df.index.get_level_values('epoch')[0]

            new_timestamps = np.arange(df.index.get_level_values('timestamp')[0] +
                                       (bin_size - df.index.get_level_values('timestamp')[0] % bin_size),
                                       df.index.get_level_values('timestamp')[-1]+1, bin_size)
            new_times = new_timestamps / float(self.sampling_rate)

            new_indices = pd.MultiIndex.from_tuples(list(zip(
                new_timestamps, new_times)), names=['timestamp', 'time'])

            #df.set_index(df.index.get_level_values('timestamp'), inplace=True)
            pos_data_bin_ids = np.arange(0, len(new_timestamps), 1)

            pos_data_binned = df.loc[day, epoch].reindex(new_indices, method='ffill')
            #pos_data_binned.set_index(new_timestamps)
            pos_data_binned['bin'] = pos_data_bin_ids

            return pos_data_binned

        grp = self.groupby(level=['day', 'epoch'])

        pos_data_rebinned = grp.apply(epoch_rebin_func)
        return type(self)(pos_data_rebinned, history=self.history, **self.kwds)

    def get_irregular_resampled(self, index_source):
        """
        timestamps input is a dataframe with day, epoch, timestamp, time, as multiindex.
        """
        result = type(self)()
        if len(index_source) > 0:
            source_index_names = index_source.index.names[:-1]
            dest_index_names = self.index.names[:-1]
            exclude_index_names = [index_source.index.names[-1]]
            common_index_names = [name for name in dest_index_names if name in source_index_names]
            unique_index_names = ([name for name in dest_index_names if name not in source_index_names] +
                                  [name for name in source_index_names if name not in dest_index_names])
            timestampsort = index_source.reset_index().sort_values(common_index_names)
            linpossort = self.reset_index().sort_values(self.index.names[:-1]).drop(exclude_index_names, axis=1)
            result = (pd.merge_asof(timestampsort, linpossort,
                                    on=common_index_names[-1],
                                    by=common_index_names[:-1],
                                    direction='nearest').set_index(source_index_names, drop=True))
            result = result.set_index(exclude_index_names, append=True).loc[:, self.columns]
            return type(self).create_default(result, arm_coord=None, **self.kwds)
        else:
            return result
    
    def apply_time_event(self, time_ranges: DayEpochEvent, event_mask_name='event_grp'):
        grouping = np.full(len(self), -1)
        time_index = self.index.get_level_values('time')
        for event_id, (range_start, range_end) in zip(time_ranges.index.get_level_values('event'),
                                                       time_ranges.get_range_view().values):
            range_mask = (time_index > range_start) & (time_index < range_end)
            grouping[range_mask] = event_id

        self[event_mask_name] = grouping

        return self


class DayEpochElecTimeChannelSeries(DayEpochTimeSeries):

    _metadata = DayEpochTimeSeries._metadata

    def __init__(self, data=None, index=None, columns=None, dtype=None, copy=False, parent=None, history=None,
                 custom_uuid=None, user_key=None, desc='', sampling_rate=0, **kwds):

        if isinstance(data, pd.DataFrame):
            if not isinstance(data.index, pd.MultiIndex):
                raise DataFormatError("DataFrame index must use MultiIndex as index.")

            if not all([col in data.index.names for col in ['day', 'epoch', 'elec_grp_id',
                                                            'timestamp', 'time', 'channel']]):
                raise DataFormatError("DayEpochElecTimeChannelSeries must have index with 6 levels named: "
                                      "day, epoch, elec_grp_id, timestamp, time.")

        if index is not None and not isinstance(index, pd.MultiIndex):
            raise DataFormatError("Index to be set must be MultiIndex.")

        super().__init__(data=data, index=index, columns=columns, dtype=dtype,
                         copy=copy, parent=parent, history=history, custom_uuid=custom_uuid,
                         user_key=user_key, desc=desc, sampling_rate=sampling_rate, **kwds)



class DayEpochElecTimeSeries(DayEpochTimeSeries):

    _metadata = DayEpochTimeSeries._metadata

    def __init__(self, data=None, index=None, columns=None, dtype=None, copy=False, parent=None, history=None,
                 custom_uuid=None, user_key=None, desc='', sampling_rate=0, **kwds):

        if isinstance(data, pd.DataFrame):
            if not isinstance(data.index, pd.MultiIndex):
                raise DataFormatError("DataFrame index must use MultiIndex as index.")

            if not all([col in data.index.names for col in ['day', 'epoch', 'elec_grp_id', 'timestamp', 'time']]):
                raise DataFormatError("DayEpochTimeSeries must have index with 6 levels named: "
                                      "day, epoch, elec_grp_id, timestamp, time.")

        if index is not None and not isinstance(index, pd.MultiIndex):
            raise DataFormatError("Index to be set must be MultiIndex.")

        super().__init__(data=data, index=index, columns=columns, dtype=dtype,
                         copy=copy, parent=parent, history=history, custom_uuid=custom_uuid,
                         user_key=user_key, desc=desc, sampling_rate=sampling_rate, **kwds)


class EncodeSettings:
    """
    Mapping of encoding parameters from realtime configuration into class attributes for easy access.
    """
    def __init__(self, realtime_config):
        """
        
        Args:
            realtime_config (dict[str, *]): JSON realtime configuration imported as a dict
        """
        encoder_config = realtime_config['encoder']

        self.sampling_rate = encoder_config['sampling_rate']
        self.arm_coordinates = encoder_config['position']['arm_pos']
        self.tetrodes = realtime_config[realtime_config['datasource']]['nspike_animal_info']['tetrodes']
        
        self.pos_upper = encoder_config['position']['upper']
        self.pos_lower = encoder_config['position']['lower']
        self.pos_num_bins = encoder_config['position']['bins']
        self.pos_bin_delta = ((self.pos_upper - self.pos_lower) / self.pos_num_bins)
        self.pos_col_names = [pos_col_format(ii, self.pos_num_bins) for ii in range(self.pos_num_bins)]

        self.pos_bins = np.linspace(0, self.pos_bin_delta * (self.pos_num_bins - 1), self.pos_num_bins)
        self.pos_bin_edges = np.linspace(0, self.pos_bin_delta * self.pos_num_bins, self.pos_num_bins+1)

        self.pos_kernel_std = encoder_config['position_kernel']['std']

        self.pos_kernel = gaussian(self.pos_bins,
                                   self.pos_bins[int(len(self.pos_bins)/2)],
                                   self.pos_kernel_std)

        self.mark_kernel_mean = encoder_config['mark_kernel']['mean']
        self.mark_kernel_std = encoder_config['mark_kernel']['std']

        self.vel = encoder_config['vel']
        self.spk_amp = encoder_config['spk_amp']

    def pos_column_name(self, pos_ind):
        return pos_col_format(pos_ind, self.pos_num_bins)

    @property
    def pos_column_slice(self):
        return slice(self.pos_col_names[0], self.pos_col_names[-1])

    @pos_column_slice.setter
    def pos_column_slice(self, slice):
        raise NotImplementedError

    def __str__(self):
        new_str = '{'
        for ind, (key, en) in enumerate(self.__dict__.items()):
            new_str += '\'' + key + '\'' + ':' + print_util.pretty_str_list(en)
            if ind != len(self.__dict__.items()) - 1:
                new_str += '\n'
        new_str += '}'
        return new_str


class DecodeSettings:
    """
    Mapping of decoding parameters from realtime configuration into class attributes for easy access.
    """
    def __init__(self, realtime_config):
        """
        
        Args:
            realtime_config (dict[str, *]): JSON realtime configuration imported as a dict
        """
        self.time_bin_size = realtime_config['pp_decoder']['bin_size']     # Decode bin size in samples (usually 30kHz)
        self.trans_smooth_std = realtime_config['pp_decoder']['trans_mat_smoother_std']
        self.trans_uniform_gain = realtime_config['pp_decoder']['trans_mat_uniform_gain']

    def __str__(self):
        new_str = '{'
        for ind, (key, en) in enumerate(self.__dict__.items()):
            new_str += '\'' + key + '\'' + ':' + print_util.pretty_str_list(en)
            if ind != len(self.__dict__.items()) - 1:
                new_str += '\n'
        new_str += '}'
        return new_str


class SpikeWaves(DayEpochElecTimeChannelSeries):

    _metadata = DayEpochElecTimeChannelSeries._metadata

    def __init__(self, data=None, index=None, columns=None, dtype=None,
                 copy=False, parent=None, history=None, custom_uuid=None,
                 user_key=None, desc='', sampling_rate=0, **kwds):

        super().__init__(data=data, index=index, columns=columns, dtype=dtype,
                         copy=copy, parent=parent, history=history, custom_uuid=custom_uuid,
                         user_key=user_key, desc=desc, sampling_rate=sampling_rate, **kwds)

    @classmethod
    def create_default(cls, df, sampling_rate, parent=None, **kwds):
        if parent is None:
            parent = df

        return cls(sampling_rate=sampling_rate, df=df, parent=parent, **kwds)

    @classmethod
    def from_df(cls, df, enc_settings, parent=None, **kwds):
        df['time'] = df.index.get_level_values('timestamp') / float(enc_settings.sampling_rate)
        df.set_index('time', append=True, inplace=True)
        #df.index = df.index.swaplevel(4, 5)
        return cls(sampling_rate=enc_settings.sampling_rate, data=df, enc_settings=enc_settings,
                   parent=parent, **kwds)


class SpikeFeatures(DayEpochElecTimeSeries):

    _metadata = DayEpochElecTimeSeries._metadata

    def __init__(self, data=None, index=None, columns=None, dtype=None,
                 copy=False, parent=None, history=None, custom_uuid=None,
                 user_key=None, desc='', sampling_rate=0, **kwds):

        super().__init__(data=data, index=index, columns=columns, dtype=dtype,
                         copy=copy, parent=parent, history=history, custom_uuid=custom_uuid,
                         user_key=user_key, desc=desc, sampling_rate=sampling_rate, **kwds)

    @classmethod
    def create_default(cls, df, sampling_rate, parent=None, **kwds):
        if parent is None:
            parent = df

        return cls(sampling_rate=sampling_rate, data=df, parent=parent, **kwds)

    @classmethod
    def from_numpy_single_epoch_elec(cls, day, epoch, elec_grp, timestamp, amps, sampling_rate):
        ind = pd.MultiIndex.from_arrays([[day]*len(timestamp), [epoch]*len(timestamp), [elec_grp]*len(timestamp),
                                         timestamp, timestamp/float(sampling_rate)],
                                        names=['day', 'epoch', 'elec_grp_id', 'timestamp', 'time'])

        return cls(sampling_rate=sampling_rate, data=amps, index=ind)    

    @classmethod
    def from_numpy_single_epoch_elec_MEC(cls, day, epoch, elec_grp, timestamp, amps, sampling_rate):
        ind = pd.MultiIndex.from_arrays([day, epoch, elec_grp,
                                         timestamp, timestamp/float(sampling_rate)],
                                        names=['day', 'epoch', 'elec_grp_id', 'timestamp', 'time'])

        return cls(sampling_rate=sampling_rate, data=amps, index=ind)

    def get_above_threshold(self, threshold):
        ind = np.nonzero(np.any(self.values > threshold, axis=1))
        return self.iloc[ind]

    def get_simple_index(self):
        """
        Only use if MultiIndex has been selected for day, epoch, and tetrode.
        Returns:

        """

        return self.set_index(self.index.get_level_values('timestamp'))


class LinearPosition(DayEpochTimeSeries):
    """
    Container for Linearized position read from an AnimalInfo.  
    
    The linearized position can come from Frank Lab's Matlab data read by the NSpike data parser
    (spykshrk.realtime.simulator.nspike_data), using the AnimalInfo class to parse the directory
    structure and PosMatDataStream to parse the linearized position files.
    """

    _metadata = DayEpochTimeSeries._metadata + ['arm_coord']

    def __init__(self, data=None, index=None, columns=None, dtype=None,
                 copy=False, parent=None, history=None, custom_uuid=None, user_key=None, desc='',
                 sampling_rate=0, **kwds):

        super().__init__(data=data, index=index, columns=columns, dtype=dtype,
                         copy=copy, parent=parent, history=history, custom_uuid=custom_uuid,
                         user_key=user_key, desc=desc, sampling_rate=sampling_rate, **kwds)

        try:
            self.arm_coord = kwds['arm_coord']
        except KeyError:
            # Likely called with a blockmanager called, which shouldn't propagate
            self.arm_coord = None

    @classmethod
    def create_default(cls, df, sampling_rate, arm_coord, parent=None, **kwds):
        if parent is None:
            parent = df

        return cls(data=df, sampling_rate=sampling_rate, arm_coord=arm_coord, parent=parent, **kwds)

    @classmethod
    def from_nspike_posmat(cls, nspike_pos_data, enc_settings: EncodeSettings, parent=None):
        """
        
        Args:
            parent: 
            nspike_pos_data: The position panda table from an animal info.  Expects a specific multi-index format.
            enc_settings: Encoder settings, used to get the endpoints of the W track
        """
        if parent is None:
            parent = nspike_pos_data

        return cls(sampling_rate=enc_settings.sampling_rate, arm_coord=enc_settings.arm_coordinates,
                   data=nspike_pos_data, parent=parent, enc_settings=enc_settings)

        # make sure there's a time field

    # @property
    # def _constructor(self):
    #     #construct_func = super()._constructor(self)
    #     print(type(self), self.arm_coord)
    #     return functools.partial(type(self), history=self.history, **self.kwds)

        #return functools.partial(construct_func, arm_coord=self.arm_coord)

    def get_pd_no_multiindex(self):
        """
        Removes the MultiIndexes, for a simplier panda table. Maintains well distances, 
        reduces velocity to just center velocity, and removes day and epoch info from index.
        
        This should not be used for multi-day datasets where the timestamp resets.
        
        Returns: Copy of linearized panda table with MultiIndexes removed

        """

        pos_data_time = self.index.get_level_values('time')
        pos_data_timestamp = self.index.get_level_values('timestamp')

        pos_data_simple = self.loc[:, 'lin_dist_well'].copy()
        pos_data_simple.loc[:, 'lin_vel_center'] = self.loc[:, ('lin_vel', 'well_center')]
        pos_data_simple.loc[:, 'seg_idx'] = self.loc[:, ('seg_idx', 'seg_idx')]
        pos_data_simple.loc[:, 'time'] = pos_data_time
        pos_data_simple.loc[:, 'timestamp'] = pos_data_timestamp
        pos_data_simple = pos_data_simple.set_index('timestamp')

        return pos_data_simple

    def get_mapped_single_axis(self):
        """
        Returns linearized position converted into a segmented 1-D representation.
        
        Returns (pd.DataFrame): Segmented 1-D linear position.
        """

        invalid = self.query('@self.seg_idx.seg_idx == 0')
        invalid_flat = pd.DataFrame([[0, 0] for _ in range(len(invalid))],
                                    columns=['linpos_flat', 'linvel_flat'], index=invalid.index)

        center_flat = (self.query('@self.seg_idx.seg_idx == 1').
                       loc[:, [('lin_dist_well', 'well_center'),
                               ('lin_vel', 'well_center')]])
        center_flat[('lin_dist_well', 'well_center')] += self.arm_coord[0][0]
        left_flat = (self.query('@self.seg_idx.seg_idx == 2 | '
                                '@self.seg_idx.seg_idx == 3').
                     loc[:, [('lin_dist_well', 'well_left'),
                             ('lin_vel', 'well_left')]])
        left_flat[('lin_dist_well', 'well_left')] += self.arm_coord[1][0]
        right_flat = (self.query('@self.seg_idx.seg_idx == 4 | '
                                 '@self.seg_idx.seg_idx == 5').
                      loc[:, [('lin_dist_well', 'well_right'),
                              ('lin_vel', 'well_right')]])
        right_flat[('lin_dist_well', 'well_right')] += self.arm_coord[2][0]
        center_flat.columns = ['linpos_flat', 'linvel_flat']
        left_flat.columns = ['linpos_flat', 'linvel_flat']
        right_flat.columns = ['linpos_flat', 'linvel_flat']

        linpos_flat = pd.concat([invalid_flat, center_flat, left_flat, right_flat]) # type: pd.DataFrame
        linpos_flat = linpos_flat.sort_index()

        linpos_flat['seg_idx'] = self.seg_idx.seg_idx

        # reset history to remove intermediate query steps
        return FlatLinearPosition.create_default(linpos_flat, self.sampling_rate,
                                                 self.arm_coord, parent=self)

    def _demetris_get_mapped_single_axis(self):
        """
        Returns linearized position converted into a segmented 1-D representation, spaced out.
        
        Returns (pd.DataFrame): Segmented 1-D linear position.

        """

        invalid = -self.linpos_flat.max() #negative range for invalid pos
        centeroffset = self.arm_coord[0][0]
        leftoffset = centeroffset + (self.arm_coord[1][0]-self.arm_coord[0][1])
        rightoffset = leftoffset + (self.arm_coord[1][1]-self.arm_coord[1][0]) + (self.arm_coord[2][0]-self.arm_coord[1][1])

        linmap = {0:invalid,1:centeroffset,2:leftoffset,3:leftoffset,4:rightoffset,5:rightoffset}
        linpos_out = self.copy()
        linpos_out['linpos_flat_unmapped'] = self['linpos_flat'] #backup
        linpos_out['linpos_flat'] = self.linpos_flat + self.seg_idx.map(linmap)
        
        return FlatLinearPosition.create_default(linpos_out, self.sampling_rate, self.arm_coord, parent=self)

    
    def get_time_only_index(self):
        return self.reset_index(level=['day', 'epoch'])


class SpikeObservation(DayEpochTimeSeries):
    """
    The observations can be generated by the realtime system or from an offline encoding model.
    The observations consist of a panda table with one row for each spike being decoded.  Each
    spike (and observation) is identified by a unique timestamp and electrode id. The observation
    content is the estimated probability that the spike and its marks will be observed in the
    encoding model.
    """

    _metadata = DayEpochTimeSeries._metadata + ['enc_settings', 'observation_bin_size', 'parallel_bin_size']

    def __init__(self, data=None, index=None, columns=None, dtype=None, copy=False,
                 parent=None, history=None, custom_uuid=None, user_key=None, desc='',
                 sampling_rate=0, enc_settings=None, **kwds):

        super().__init__(data=data, index=index, columns=columns, dtype=dtype,
                         copy=copy, parent=parent, history=history, custom_uuid=custom_uuid,
                         user_key=user_key, desc=desc, sampling_rate=sampling_rate,
                         enc_settings=enc_settings, **kwds)

        self.enc_settings = enc_settings

        self.parallel_bin_size = 0
        self.observation_bin_size = 0

    @classmethod
    def create_default(cls, df, enc_settings, sampling_rate=None, parent=None, **kwds):
        if parent is None:
            parent = df

        if sampling_rate is None:
            sampling_rate = enc_settings.sampling_rate

        return cls(sampling_rate=sampling_rate, data=df, parent=parent, enc_settings=enc_settings, **kwds)

    @classmethod
    def from_realtime(cls, spike_dec, day, epoch, enc_settings, parent=None, **kwds):
        if parent is None:
            parent = spike_dec

        start_timestamp = spike_dec['timestamp'][0]
        spike_dec['time'] = spike_dec['timestamp'] / float(enc_settings.sampling_rate)
        return cls(sampling_rate=enc_settings.sampling_rate,
                   data=spike_dec.set_index(pd.MultiIndex.from_arrays([[day]*len(spike_dec), [epoch]*len(spike_dec),
                                                                       spike_dec['timestamp'], spike_dec['time']],
                                                                      names=['day', 'epoch', 'timestamp', 'time'])),
                   parent=parent, enc_settings=enc_settings, **kwds)

    def update_observations_bins(self, time_bin_size, inplace=False):
        if inplace:
            df = self
        else:
            df = self.copy()

        self.observation_bin_size = time_bin_size
        if self.parallel_bin_size % self.observation_bin_size != 0:
            raise DataFormatError('Parallel time bins must be a multiple of observation bin sizes.')

        dec_bins = np.floor((self.index.get_level_values('timestamp') -
                             self.index.get_level_values('timestamp')[0]) / time_bin_size).astype('int')
        dec_bins_start = (int(self.index.get_level_values('timestamp')[0] / time_bin_size) *
                          time_bin_size + dec_bins * time_bin_size)
        df['dec_bin'] = dec_bins
        df['dec_bin_start'] = dec_bins_start

        df.update_num_missing_future_bins(inplace=True)
        return df

    def update_parallel_bins(self, time_bin_size, inplace=True):
        if inplace:
            df = self
        else:
            df = self.copy()

        self.parallel_bin_size = time_bin_size

        if self.parallel_bin_size % self.observation_bin_size != 0:
            raise DataFormatError('Parallel time bins must be a multiple of observation bin sizes.')
        parallel_bins = np.floor((self.index.get_level_values('timestamp') -
                                  self.index.get_level_values('timestamp')[0]) / time_bin_size).astype('int')
        df['parallel_bin'] = parallel_bins

        return df

    def get_no_multi_index(self):
        return pd.DataFrame(self.set_index(self.index.get_level_values('timestamp')))

    def update_num_missing_future_bins(self, inplace=False):
        if inplace:
            df = self
        else:
            df = self.copy()
        df['num_missing_bins'] = np.concatenate([np.clip(np.diff(df['dec_bin'])-1, 0, None), [0]])

        return df

    def get_distribution_as_np(self):
        return self.loc[:, pos_col_format(0, self.kwds['enc_settings'].pos_num_bins):
                        pos_col_format(self.kwds['enc_settings'].pos_num_bins-1,
                                       self.kwds['enc_settings'].pos_num_bins)].values

    def set_distribution(self, dist):
        self.loc[:, pos_col_format(0, self.enc_settings.pos_num_bins):
                 pos_col_format(self.enc_settings.pos_num_bins-1, self.enc_settings.pos_num_bins)] = dist

    def get_distribution_view(self):
        return self.loc[:, pos_col_format(0, self.enc_settings.pos_num_bins):
                        pos_col_format(self.enc_settings.pos_num_bins-1, self.enc_settings.pos_num_bins)]


class Posteriors(DayEpochTimeSeries):

    _metadata = DayEpochTimeSeries._metadata + ['enc_settings', 'dec_settings']

    def __init__(self, data=None, index=None, columns=None, dtype=None, copy=False,
                 parent=None, history=None,custom_uuid=None, user_key=None, desc='',
                 sampling_rate=None, enc_settings=None, dec_settings=None, **kwds):

        self.enc_settings = enc_settings

        self.dec_settings = dec_settings

        self.sampling_rate = sampling_rate

        if issubclass(type(data), DataFrameClass):
            if self.enc_settings is not None:
                warnings.warn(OverridingAttributeWarning('enc_setting found in {} will be '
                                                         'overwritten by constructor value'.format(data)))
            else:
                self.enc_settings = data.kwds['enc_settings']

            if self.dec_settings is not None:
                warnings.warn(OverridingAttributeWarning('dec_setting found in {} will be '
                                                         'overwritten by constructor value'.format(data)))
            else:
                self.dec_settings = data.kwds['dec_settings']

            if self.sampling_rate is not None:
                warnings.warn(OverridingAttributeWarning('sampling_rate found in {} will be '
                                                         'overwritten by constructor value'.format(data)))

            else:
                self.sampling_rate = data.kwds['sampling_rate']

        super().__init__(data=data, index=index, columns=columns, dtype=dtype,
                         copy=copy, parent=parent, history=history, custom_uuid=custom_uuid,
                         user_key=user_key, desc=desc, sampling_rate=sampling_rate,
                         enc_settings=enc_settings, dec_settings=dec_settings, **kwds)

    #@property
    #def _constructor(self):
    #    return functools.partial(type(self), history=self.history, **self.kwds)

    @classmethod
    def create_default(cls, df, enc_settings, dec_settings, parent=None, user_key=None, **kwds):
        if parent is None:
            parent = df

        sampling_rate = enc_settings.sampling_rate

        return cls(df, sampling_rate=sampling_rate, parent=parent, enc_settings=enc_settings,
                   dec_settings=dec_settings, user_key= user_key, **kwds)

    @classmethod
    def from_dataframe(cls, posterior: pd.DataFrame, enc_settings, dec_settings,
                       index=None, columns=None, parent=None, user_key=None, **kwds):
        if parent is None:
            parent = posterior

        if index is not None:
            posterior.set_index(index)
        if columns is not None:
            posterior.columns = columns

        sampling_rate = enc_settings.sampling_rate
        return cls(data=posterior, sampling_rate=sampling_rate, parent=parent,
                   enc_settings=enc_settings, dec_settings=dec_settings, user_key=user_key, **kwds)

    @classmethod
    def from_numpy(cls, posterior, day, epoch, timestamps, times, columns=None, parent=None, sampling_rate=None,
                   enc_settings=None, user_key=None, **kwds):

        if (sampling_rate is None) and (enc_settings is None):
            raise DataFormatError('Posteriors.from_numpy() must specify either sampling_rate or encode_setting object.')
        if (sampling_rate is not None) and (enc_settings is not None):
            raise DataFormatError('Posteriors.from_numpy() only specify one, sampling rate or encode_setting.')

        if enc_settings is not None:
            sampling_rate = enc_settings.sampling_rate

        if parent is None:
            parent = posterior

        return cls(data=posterior, index=pd.MultiIndex.from_arrays([[day]*len(posterior), [epoch]*len(posterior),
                                                                    timestamps, times],
                                                                   names=['day', 'epoch', 'timestamp', 'time']),
                   columns=columns, parent=parent, enc_settings=enc_settings, sampling_rate=sampling_rate,
                   user_key=user_key, **kwds)

    @classmethod
    def from_realtime(cls, posterior: pd.DataFrame, day, epoch, columns=None, copy=False, parent=None,
                      enc_settings=None, sampling_rate=None, user_key=None, **kwds):

        if (sampling_rate is None) and (enc_settings is None):
            raise DataFormatError('Posteriors.from_numpy() must specify either sampling_rate or encode_setting object.')
        if (sampling_rate is not None) and (enc_settings is not None):
            raise DataFormatError('Posteriors.from_numpy() only specify one, sampling rate or encode_setting.')

        if enc_settings is not None:
            sampling_rate = enc_settings.sampling_rate

        if parent is None:
            parent = posterior

        if copy:
            posterior = posterior.copy()    # type: pd.DataFrame
        posterior.set_index(pd.MultiIndex.from_arrays([[day]*len(posterior), [epoch]*len(posterior),
                                                       posterior['timestamp'], posterior['timestamp']/
                                                       sampling_rate],
                                                      names=['day', 'epoch', 'timestamp', 'time']), inplace=True)

        if columns is not None:
            posterior.columns = columns

        return cls(data=posterior, parent=parent, enc_settings=enc_settings, sampling_rate=sampling_rate,
                   user_key=user_key, **kwds)

    def get_posteriors_as_np(self):
        return self[pos_col_format(0, self.kwds['enc_settings'].pos_num_bins):
                    pos_col_format(self.kwds['enc_settings'].pos_num_bins-1,
                                   self.kwds['enc_settings'].pos_num_bins)].values

    def set_posterior(self, post):
        self.loc[:, pos_col_format(0, self.enc_settings.pos_num_bins):
                 pos_col_format(self.enc_settings.pos_num_bins-1, self.enc_settings.pos_num_bins)] = post

    def get_distribution_view(self):
        return self.loc[:, pos_col_format(0, self.enc_settings.pos_num_bins):
                        pos_col_format(self.enc_settings.pos_num_bins-1, self.enc_settings.pos_num_bins)]

    def get_pos_range(self):
        return self.enc_settings.pos_bins[0], self.enc_settings.pos_bins[-1]

    def get_pos_start(self):
        return self.enc_settings.pos_bins[0]

    def get_pos_end(self):
        return self.enc_settings.pos_bins[-1]


class RippleTimes(DayEpochEvent):

    def __init__(self, data=None, index=None, columns=None, dtype=None, copy=False,
                 parent=None, history=None, custom_uuid=None, user_key=None, desc='', **kwds):

        super().__init__(data=data, index=index, columns=columns, dtype=dtype,
                         copy=copy, parent=parent, history=history, custom_uuid=custom_uuid,
                         user_key=user_key, desc=desc, **kwds)

        if isinstance(data, pd.DataFrame):
            if not 'maxthresh' in data.columns:
                raise DataFormatError("RippleTimes must have 'maxthresh' column.")

    @classmethod
    def create_default(cls, df, time_unit: UnitTime, parent=None, **kwds):
        if parent is None:
            parent = df

        return cls(data=df, time_unit=time_unit, parent=parent, **kwds)

    def get_above_maxthresh(self, threshold):
        return self.query('maxthresh >= @threshold')


class StimLockout(DataFrameClass):

    _metadata = DataFrameClass._metadata

    def __init__(self, data=None, index=None, columns=None, dtype=None, copy=False,
                 parent=None, history=None, custom_uuid=None, user_key=None, desc='', **kwds):

        super().__init__(data=data, index=index, columns=columns, dtype=dtype,
                         copy=copy, parent=parent, history=history, custom_uuid=custom_uuid,
                         user_key=user_key, desc=desc, **kwds)

    @classmethod
    def create_default(cls, df, sampling_rate, parent=None, **kwds):
        if parent is None:
            parent = df

        enc_settings = AttrDict({'sampling_rate': sampling_rate})
        return cls.from_realtime(df, enc_settings=enc_settings, parent=parent, **kwds)

    @classmethod
    def from_realtime(cls, stim_lockout, enc_settings, parent=None, **kwds):
        """
        Class factory to create stimulation lockout from realtime system.  
        Reshapes the structure to a more useful format (stim lockout intervals)
        Args:
            parent: 
            stim_lockout: Stim lockout pandas table from realtime records

        Returns: StimLockout

        """
        if parent is None:
            parent = stim_lockout

        stim_lockout_ranges = stim_lockout.pivot(index='lockout_num', columns='lockout_state', values='timestamp')
        stim_lockout_ranges = stim_lockout_ranges.reindex(columns=[1, 0])
        stim_lockout_ranges.columns = pd.MultiIndex.from_product([['timestamp'], ['on', 'off']])
        stim_lockout_ranges_sec = stim_lockout_ranges / float(enc_settings.sampling_rate)
        stim_lockout_ranges_sec.columns = pd.MultiIndex.from_product([['time'], ['on', 'off']])
        df = pd.concat([stim_lockout_ranges, stim_lockout_ranges_sec], axis=1)      # type: pd.DataFrame

        return cls(df, parent=parent, enc_settings=enc_settings, **kwds)

    def get_range_sec(self, low, high):
        sel = self.query('@self.time.off > @low and @self.time.on < @high')
        return type(self)(sel)


class FlatLinearPosition(LinearPosition):

    _metadata = LinearPosition._metadata + ['arm_coord']

    def __init__(self, data=None, index=None, columns=None, dtype=None,
                 copy=False, parent=None, history=None, custom_uuid=None,
                 user_key=None, desc='', sampling_rate=0, **kwds):
        super().__init__(data=data, index=index, columns=columns, dtype=dtype,
                         copy=copy, parent=parent, history=history, custom_uuid=custom_uuid,
                         user_key=user_key, desc=desc, sampling_rate=sampling_rate, **kwds)

        if isinstance(data, pd.DataFrame) and 'linvel_flat' not in data.columns:
            raise DataFormatError("Missing 'linvel_flat' column.")

    @classmethod
    def create_default(cls, df, sampling_rate, arm_coord, parent=None, **kwds):
        if parent is None:
            parent = df

        return cls(df, parent=parent, sampling_rate=sampling_rate, arm_coord=arm_coord, **kwds)

    @classmethod
    def from_nspike_posmat(cls, nspike_pos_data, enc_settings: EncodeSettings, parent=None):
        return LinearPosition.from_nspike_posmat(nspike_pos_data, enc_settings, parent).get_mapped_single_axis()


    @classmethod
    def from_numpy_single_epoch(cls, day, epoch, timestamp, lin_pos, lin_vel, sampling_rate, arm_coord):
        time = timestamp/float(sampling_rate)
        return cls(pd.DataFrame(list(zip(lin_pos, lin_vel)), columns=['linpos_flat', 'linvel_flat'],
                                index=pd.MultiIndex.from_arrays([[day]*len(timestamp), [epoch]*len(timestamp),
                                                                 timestamp, time],
                                                                names=['day', 'epoch', 'timestamp', 'time'])),
                   sampling_rate=sampling_rate, arm_coord=arm_coord)

    @classmethod
    def from_numpy_single_epoch_MEC(cls, day, epoch, timestamp, lin_pos, lin_vel, sampling_rate, arm_coord):
        time = timestamp/float(sampling_rate)
        return cls(pd.DataFrame(list(zip(lin_pos, lin_vel)), columns=['linpos_flat', 'linvel_flat'],
                                index=pd.MultiIndex.from_arrays([day, epoch,
                                                                 timestamp, time],
                                                                names=['day', 'epoch', 'timestamp', 'time'])),
                   sampling_rate=sampling_rate, arm_coord=arm_coord)

    def get_above_velocity(self, threshold):

        return self.query('abs(linvel_flat) >= @threshold')

    def get_pd_no_multiindex(self):
        self.set_index(pd.Index(self.index.get_level_values('timestamp'), name='timestamp'))

    def get_mapped_single_axis(self):
        return self
