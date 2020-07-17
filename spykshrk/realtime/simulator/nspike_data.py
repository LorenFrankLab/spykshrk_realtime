"""
Defines classes necessary to access and stream NSpike animal's data.

"""
import itertools
import os.path
from array import array
from glob import glob
from struct import unpack

import numpy as np
import pandas as pd
from scipy.io import loadmat

import spykshrk.franklab.filterframework_util as ff_util
from spykshrk.franklab.data_containers import RippleTimes, UnitTime
from spykshrk.realtime.datatypes import LFPPoint, LinearPosPoint, SpikePoint
from spykshrk.franklab.errors import DataReadError, DataContentError, ConfigurationError
idx = pd.IndexSlice

try:
    from IPython.terminal.debugger import TerminalPdb
    bp = TerminalPdb().set_trace
except AttributeError as err:
    #print('Warning: Attribute Error ({}), disabling IPython TerminalPdb.'.format(err))
    bp = lambda: None

# setup pandas string/print format for debugging/interactive
pd.set_option('float_format', '{:,.1f}'.format)
pd.set_option('display.max_rows', 20)
pd.set_option('display.width', 120)

# Constant variable definitions
TIMESTAMP_SCALE = 10000  # factor to convert timestamp to s


class AnimalInfo:
    """ animalinfo - stores details of the single animal to process and generates
    file lists that points to the data to process.
    
    """

    def __init__(self, base_dir, name, days, epochs, tetrodes, timescale=10000,
                 new_data=True):
        """ init function
        
        Args:
            base_dir: root directory of animal data
            name: name of animal
            days: array of which days of data to process
            tetrodes: array of which tetrodes to process
            epochs: list of epochs for encoding
        
        """
        self.base_dir = base_dir
        self.name = name
        self.days = days
        self.epochs = epochs
        self.tetrodes = tetrodes
        self.timescale = timescale
        self.times = self.parse_times()
        self.new_data = new_data

    def parse_times(self):
        """ parse_time - stores time data (times.mat) for each day to make sure
        each day is synchronized.  Should only be called from __init__

        Stores only the times for each epoch, 0 indexed.
        
        """
        times = {}
        for day in self.days:
            day_path = os.path.join(self.base_dir, self.name,
                                    '%s%02d/times.mat' % (self.name, day))
            mat = loadmat(day_path)
            time_ranges = mat['ranges']
            times.setdefault(day, time_ranges[1::, :].astype('int32').tolist())
        return times

    def get_spike_paths(self):
        """ get_spike_paths - returns a list of paths that points to the spike
        data ([animalname][day]-[tetrode].mat) of each tetrode and day.

        This function uses the matlab data structure because the reference dataset
        being used is Frank, whose raw data was lost.
        
        """
        path_list = []
        for day in self.days:
            day_path = os.path.join(self.base_dir, self.name,
                                    '%s%02d' % (self.name, day))
            for tet in self.tetrodes:
                tet_path_glob = os.path.join(day_path, '%02d*' % tet)
                tet_paths = glob(tet_path_glob)

                # only keep matching directories
                for tet_path in tet_paths:
                    if not os.path.isdir(tet_path):
                        del tet_paths[tet_paths.index(tet_path)]

                # Directory sanity checks
                if len(tet_paths) < 1:
                    print(('WARNING: %s day %02d does not have file for tetrode %02d,' +
                          ' skipping tetrode (%s)') %
                          (self.name, day, tet, tet_path_glob))
                    continue
                elif len(tet_paths) > 1:
                    print(('WARNING: %s day %02d has multiple directories for tetrode %02d,' +
                          'by default using first entry\n(%s)') %
                          (self.name, day, tet, tet_paths))

                spike_data_path = os.path.join(tet_paths[0], '%s%02d-%02d.mat'
                                               % (self.name, day, tet))

                path_list.append((day, tet, spike_data_path))

        return path_list

    def get_eeg_paths(self):
        """ returns a list of paths that points to the eeg data ([tetrode]-###.eeg) of each tetrode and day.

        This function uses the older \*.eeg because the reference dataset is frank.
        
        """
        path_list = []

        for day in self.days:
            day_path = os.path.join(self.base_dir, self.name, '%s%02d'
                                    % (self.name, day))
            for tet in self.tetrodes:
                tet_path_glob = os.path.join(day_path, '%02d*.eeg' % tet)
                tet_paths = glob(tet_path_glob)
                if len(tet_paths) < 1:
                    print(('WARNING: %s day %02d does not have eeg file for tetrode %02d,' +
                           ' skipping tetrode (%s)')
                          % (self.name, day, tet, tet_path_glob))
                    continue
                elif len(tet_paths) > 1:
                    print(('WARNING: %s day %02d has multiple eeg files for tetrode %02d,' +
                           'by default using first entry\n(%s)')
                          % (self.name, day, tet, tet_paths))
                path_list.append((day, tet, tet_paths[0]))
        return path_list

    def get_pos_paths(self):
        """ get_pos_paths - returns a list of paths that points to the pos
        data (matclust_[animalname][day]-[tetrode].mat) of each tetrode and day.

        This function uses the already processed matclust data to extract position
        because the raw data of the reference dataset used (frank) was lost.
        
        """
        path_list = []
        for day in self.days:
            day_path = os.path.join(self.base_dir, self.name, '%s%02d'
                                    % (self.name, day))
            for tet in self.tetrodes:
                tet_path_glob = os.path.join(day_path, '%02d*' % tet, '%s%02d-%02d_params.mat'
                                             % (self.name, day, tet))
                tet_paths = glob(tet_path_glob)

                # directory sanity check
                if len(tet_paths) < 1:
                    print(('WARNING: %s day %02d does not have file for tetrode %02d,' +
                           ' skipping tetrode (%s)')
                          % (self.name, day, tet, tet_path_glob))
                    continue
                elif len(tet_paths) > 1:
                    print(('WARNING: %s day %02d has multiple directories for tetrode %02d,' +
                           'by default using first entry\n(%s)')
                          % (self.name, day, tet, tet_paths))

                path_list.append((day, tet, tet_paths[0]))
        return path_list

    def get_posmat_paths(self):
        """ get_pathmat_paths - returns a list of paths that points to the post processed pos
        data ([post dir]/[3 char name prefix]pos[day].mat) of each day.

        This function uses the already processed video position data to extract position
        because the raw data of the reference dataset used (frank) was lost.
        
        """
        path_list = []
        for day in self.days:
            anim_prefix = self.name[0:3].title()
            day_glob = os.path.join(self.base_dir, anim_prefix.title()[0:3],
                                    '%slinpos%02d.mat' % (self.name[0:3].lower(), day))
            day_path = glob(day_glob)
            # directory sanity check
            if len(day_path) < 1:
                print('WARNING: %s day %02d does not have file %slinpos%02d.mat'
                      % (self.name, day, self.name[0:3].lower(), day))
                continue
            elif len(day_path) > 1:
                print(('WARNING: %s day %02d has multiple directories %slinpos%02d.mat' +
                       'which should not happen...')
                      % (self.name, day, self.name[0:3].lower(), day))

            path_list.append((day, day_path[0]))
        return path_list

    def get_ripplecons_paths(self):

        path_list = []
        for day in self.days:
            anim_prefix = self.name[0:3].title()
            day_glob = os.path.join(self.base_dir, anim_prefix.title()[0:3],
                                    '%sripplescons%02d.mat' % (self.name[0:3].lower(), day))
            day_path = glob(day_glob)
            # directory sanity check
            if len(day_path) < 1:
                print('WARNING: %s day %02d does not have file %sripplescons%02d.mat'
                      % (self.name, day, self.name[0:3].lower(), day))
                continue
            elif len(day_path) > 1:
                print(('WARNING: %s day %02d has multiple directories %sripplescons%02d.mat' +
                       'which should not happen...')
                      % (self.name, day, self.name[0:3].lower(), day))

            path_list.append((day, day_path[0]))
        return path_list

    def num_epochs(self, day):
        time_day = self.times[day]
        return np.size(time_day, 0)

    def calc_epoch_state(self, day, cur_time):
        """ Returns the encoded epoch that the time belongs to.  If the time
        is not in an epoch or if the epoch is not being encoded, return -1
        
        Args:
            day: The day to lookup
            cur_time: The current time to lookup
        
        """
        time = self.times[day]
        for ii in self.epochs:
            if time[ii][0] <= cur_time <= time[ii][1]:
                return ii
        return -1

    def get_epoch_time_range(self, day, epoch):
        time = self.times[day]
        if epoch not in self.epochs:
            raise ConfigurationError('Epoch requested not an '
                                     'epoch being processed')

        return time[epoch][0], time[epoch][1]


class RipplesConsData:

    def __init__(self, anim: AnimalInfo, ripcons_area_ind=0):
        self.data = pd.DataFrame()
        self.anim = anim
        self.days = set()
        self.mat_list = []
        self.raw_list = []
        path_list = anim.get_ripplecons_paths()

        for path in path_list:
            day = path[0]
            self.days.add(day)
            self.mat_list.append(loadmat(path[1])['ripplescons'])
            self.raw_list.append(ff_util.parse_filterframework_cells(self.mat_list[-1]))

        self.raw = ff_util.merge_filterframework_cells(self.raw_list)

        for day, raw_day in self.raw.items():
            for epoch in self.anim.epochs:
                raw_epoch = raw_day[epoch]
                rip_cons = raw_epoch[ripcons_area_ind]
                rip_cons_data = {'starttime': rip_cons.starttime,
                                 'endtime': rip_cons.endtime,
                                 'maxthresh': rip_cons.maxthresh}
                rip_cons_pd = pd.DataFrame(rip_cons_data,
                                           index=pd.MultiIndex.from_product([[day], [epoch],
                                                                             range(len(rip_cons.starttime))],
                                                                            names=['day', 'epoch', 'event']))
                rip_cons_pd = rip_cons_pd[['starttime', 'endtime', 'maxthresh']]
                self.data = self.data.append(rip_cons_pd)
                self.data_obj = RippleTimes.create_default(df=self.data, time_unit=UnitTime.SEC)


class SpkDataStream:
    """	spike_data - Opens all files specified by animalinfo and creates a generator
    that can be called/polled to return spike data in bundles of the set timestep
    
    Args:
        anim (AnimalInfo): Defines the animal data and locations
        timestep: The time interval for the generator to return on each call
    """

    DAY_IND = 0
    TET_IND = 1
    DATA_IND = 2
    TIME_IND = 2
    WAVE_IND = 3

    def __init__(self, anim):
        # self.timestamps = []
        self.data = pd.DataFrame()
        self.anim = anim
        self.timebin_uniq = {}
        path_list = anim.get_spike_paths()
        self.days = set([])

        for path in path_list:
            day = path[self.DAY_IND]
            tet = path[self.TET_IND]
            mat = loadmat(path[self.DATA_IND])
            self.days.add(day)

            day_timestamps = mat['timestamps']
            day_waves = mat['waves']
            # Make sure not empty
            if day_timestamps.any() and day_waves.any():
                for epoch in self.anim.epochs:
                    epoch_start, epoch_end = self.anim.get_epoch_time_range(day, epoch)
                    epoch_mask = ((day_timestamps >= epoch_start) &
                                  (day_timestamps <= epoch_end)).ravel()
                    epoch_numspks = np.nonzero(epoch_mask)[0].size

                    # hardcoded conversion from 3 samples/time
                    epoch_spk_time = day_timestamps[epoch_mask] * 3
                    epoch_spk_wave = day_waves[:, :, epoch_mask]
                    # convert nparray to a list so each waveform set can be
                    # added to the right index

                    epoch_spk_wave_reshape = epoch_spk_wave.swapaxes(1, 2).reshape(epoch_spk_wave.shape[0],
                                                                                   epoch_spk_wave.shape[2] *
                                                                                   epoch_spk_wave.shape[1]).T

                    epoch_spk_time_reshape = np.tile(epoch_spk_time, (1, 4)).reshape(4 * len(epoch_spk_time))
                    epoch_spk_wave_ind = pd.MultiIndex.from_product([[day], [epoch], [tet], epoch_spk_time_reshape],
                                                                    names=['day', 'epoch', 'elec_grp_id',
                                                                           'timestamp'])

                    epoch_spk_wave_df = pd.DataFrame(data=epoch_spk_wave_reshape, index=epoch_spk_wave_ind,
                                                     columns=pd.Index(['s{:02d}'.format(ii) for ii in
                                                                       range(epoch_spk_wave_reshape.shape[1])],
                                                                      name='sample'))

                    epoch_spk_time_channel = np.tile(['c{:02d}'.format(ii) for ii in
                                                      np.arange(0, epoch_spk_wave.shape[1])],
                                                     epoch_spk_wave.shape[2])

                    epoch_spk_wave_df['channel'] = epoch_spk_time_channel
                    epoch_spk_wave_df.set_index('channel', append=True, inplace=True)

                    self.data = self.data.append(epoch_spk_wave_df)

    def __call__(self):
        for day in self.days:
            for epoch in self.anim.epochs:
                # For day, epoch sort based on time
                epoch_data_sorted = self.data.loc[idx[day, epoch, :], :].sort_index(level='timestamp')
                # Waveform data
                epoch_data_sorted_raw = epoch_data_sorted.values
                # Indexing data (with tet_id info)
                epoch_data_sorted_index = epoch_data_sorted.index

                for ind in range(0, len(epoch_data_sorted_raw), 4):
                    timestamp = epoch_data_sorted_index[ind][3]
                    spk_data = epoch_data_sorted_raw[ind:ind+4, :]
                    elec_grp = epoch_data_sorted_index[ind][2]

                    yield SpikePoint(timestamp=timestamp, elec_grp_id=elec_grp, data=spk_data)


class EEGDataStream:
    """	Streams NSpike EEG Data specified by AnimalInfo.
    
    Opens all files specified by anim (AnimalInfo) and creates a generator
    that can be called/polled to return eeg data points.  EEG points are returned
    one at a time for each tetrode and in chronological order.
    
    Args:
        anim (AnimalInfo): Defines the animal data and locations
    """
    DAY_IND = 0
    TET_IND = 1
    TIME_IND = 2
    DATA_IND = 2
    EEG_REC_TIME_IND = 0
    EEG_REC_NUMSAMP_IND = 1
    EEG_REC_SAMPRATE_IND = 2
    EEG_REC_DATA_IND = 3

    def __init__(self, anim):
        """
        Init function.
        
        Args:
            anim (AnimalInfo): The AnimalInfo that defines the data to stream eeg data from.
        """

        self.data = {}
        self.anim = anim
        path_list = anim.get_eeg_paths()
        days = list(set(map(lambda x: x[self.DAY_IND], path_list)))
        open_files = {day: [] for day in days}
        # Dict of DataFrames for each day
        self.data = {}
        for path in path_list:
            day = path[self.DAY_IND]
            tet = path[self.TET_IND]
            # record reader iterator
            get_eeg_rec = self.read_eeg_rec(path[self.DATA_IND])

            try:
                rec = get_eeg_rec.__next__()
                cur_epoch = self.anim.calc_epoch_state(path[0], rec[0])
                # Assume first record's sample frequency is correct
                # and all files have the same sample frequency
                self.sampfreq = rec[2]

                self.dt = 1/self.sampfreq*anim.timescale

                # empty dataframe per day so each tetrode's dataframe
                # can be joined
                day_data = self.data.setdefault(day, pd.DataFrame())
                # print('preallocating')

                tet_eeg_data_list = []
                tet_eeg_time_list = []
                tet_eeg_rowcount = 0
                # keep reading records until iterator ends
                while True:
                    rec_start_time = rec[0]
                    cur_epoch = self.anim.calc_epoch_state(path[0], rec_start_time)
                    # if rec in valid epoch range
                    if cur_epoch != -1:
                        rec_num_samp = rec[1]
                        rec_data = rec[3]
                        # interpolate start time to the entire record
                        rec_times = np.arange(0,rec_num_samp)*self.dt+rec_start_time
                        tet_eeg_data_list.extend(rec_data)
                        tet_eeg_time_list.extend(rec_times)

                        tet_eeg_rowcount += rec_num_samp

                    rec = get_eeg_rec.__next__()

            except StopIteration as ex:
                pass

            # processing file done
            day_data = day_data.merge(pd.DataFrame(index=tet_eeg_time_list,
                                                   data=tet_eeg_data_list, columns=[tet]),
                                      how='outer', left_index=True, right_index=True, copy=False)
            # re add merged data to dict
            self.data[day] = day_data

    def __call__(self):
        # extract number of days into a unique day list
        day_list = self.data.keys()
        days = list(set(day_list))
        for day in days:

            day_data = self.data[day]   # type: pd.DataFrame
            # place holder to avoid pandas overhead on each call to attribute values
            day_data_values = day_data.values

            for epoch in self.anim.epochs:

                epoch_start_time = self.anim.times[day][epoch][0]
                epoch_end_time = self.anim.times[day][epoch][1]
                # Initialize starting row to the start of the epoch
                row_cursor = np.abs(day_data.index - epoch_start_time).argmin()

                timestamp = day_data.index[row_cursor]
                while timestamp < epoch_end_time:
                    # Gets the voltages values of all tetrodes at specific timepoint (cursor)
                    # bypassing pandas by using values to avoid creating Series
                    raw_timepoint = day_data_values[row_cursor]
                    for col_ind, tet_val in enumerate(raw_timepoint):
                        # test to make sure value is not NaN, this is a computationally efficient shortcut way.
                        if tet_val == tet_val:
                            tet_id = day_data.columns[col_ind]

                            # Hardcoded timestamp conversion to sample rate (10kHz to 30kHz)
                            yield LFPPoint(timestamp=int(timestamp * 3), ntrode_index=int(col_ind),
                                           elec_grp_id=int(tet_id), data=int(tet_val))

                    # Increment to next row
                    row_cursor += 1
                    # retrieve new row's timestamp
                    timestamp = day_data.index[row_cursor]

    def stream_eeg_rec(self, rec):
        """ A generator function that takes a single record and returns
        each data point in the record with an interpolated timestamp
        based on the sample frequency.  The record must conform with \*.eeg
        format (timestamp, numsamples, sample rate, [data array])
        Args:
            rec: eeg record (timestamp, numsamples, sample rate, [data array])
        
        Returns:
            interp_time: interpolated timestamp of data point being returned
            eeg_pt: eeg sample data
        """
        rec_data = rec[self.EEG_REC_DATA_IND]
        rec_timestamp = rec[self.EEG_REC_TIME_IND]
        rec_numsamp = rec[self.EEG_REC_NUMSAMP_IND]
        rec_sampfreq = rec[self.EEG_REC_SAMPRATE_IND]

        samp_count = 0

        for point in rec_data:
            point_timestamp = rec_timestamp + 1 / float(rec_sampfreq) * samp_count * 10000
            yield point_timestamp, point
            samp_count += 1
            if samp_count > rec_numsamp:
                raise DataReadError('Record being streamed has more samples than \
                        set in the number of samples field.')
        if samp_count < rec_numsamp:
            raise DataReadError('Record being streamed has fewer samples than \
                    set in the number of samples field.')

    @staticmethod
    def read_eeg_header(path):
        f = open(path, 'r')
        header_text = []
        try:
            # reading header, limit to looking at top few lines
            # before throwing exception
            for linenum in range(0, 10):
                line = f.readline()

                header_text.append(line)
                if line == '%%ENDHEADER\n':
                    break

        except EOFError as e:
            print('EOFError reading EEG file (%s) - (%s, %s)' % (f.name, repr(e), e.args))
            f.close()
            return

        return header_text

    @staticmethod
    def read_eeg_rec(path):
        """ A generator function that for the file string specified in path,
        will return single eeg record (\*.eeg format) in sequential order. The
        generator returns (StopIterator) at the EOF.
        
        Args:
            path: eeg file path string
        
        Returns:
            timestamp: timestamp for begining of record
            numsamples: number of samples in record
            sampfreq: frequency of sample
            data: raw data (short) of eeg record
        """
        with open(path, 'rb') as f:
            error_code = 0
            try:
                # Skip header
                while f.readline() != b'%%ENDHEADER\n':
                    pass

                while True:
                    timestamp_bytes = f.read(4)
                    if not timestamp_bytes:
                        f.close()
                        return
                    timestamp = unpack('I', timestamp_bytes)[0]
                    error_code = 1
                    numsamples_bytes = f.read(4)
                    if not numsamples_bytes:
                        f.close()
                        return
                    numsamples = unpack('i', numsamples_bytes)[0]
                    error_code = 2
                    sampfreq_bytes = f.read(8)
                    if not sampfreq_bytes:
                        f.close()
                        return
                    sampfreq = unpack('d', sampfreq_bytes)[0]

                    error_code = 3
                    data = array('h')
                    data.fromfile(f, numsamples)

                    yield (timestamp, numsamples, sampfreq, data)

            except EOFError as e:
                print('EOFError reading EEG file (%s) - (%s, %s): code %d' % (f.name, repr(e), e.args, error_code))
                f.close()
                return


class PosMatDataStream:
    DAY_IND = 0
    TIME_IND = 0
    LINPOS_IND = 1
    PATH_IND = 1
    SEGINDEX_IND = 2
    VELLIN_IND = 3

    def __init__(self, anim):
        self.anim = anim
        path_list = anim.get_posmat_paths()
        # Initialized to empty dataframe to be appended/concated to
        self.data = pd.DataFrame([])

        self.days = set([])
        for path in path_list:
            day = path[self.DAY_IND]
            self.days.add(day)
            mat = loadmat(path[self.PATH_IND])
            posdata = mat['linpos'][0, day - 1]

            self.timebin_uniq = {}
            for epoch in anim.epochs:
                postime_epoch = posdata[0, epoch]['statematrix'][0, 0]['time'][0, 0]
                postimestamp_epoch = postime_epoch * 30000          # hardcode convert to timestamps
                poslindist_epoch = posdata[0, epoch]['statematrix'][0, 0]['linearDistanceToWells'][0, 0]
                possegind_epoch = posdata[0, epoch]['statematrix'][0, 0]['segmentIndex'][0, 0]
                poslinvel_epoch = posdata[0, epoch]['statematrix'][0, 0]['linearVelocity'][0, 0]
                posdata_all = np.hstack((poslindist_epoch, poslinvel_epoch, possegind_epoch))

                pos_ind_tup = list(itertools.starmap(lambda d, e, t: np.hstack([d, e, t]),
                                                     itertools.product([day], [epoch],
                                                                       zip(postimestamp_epoch.flatten(),
                                                                           postime_epoch.flatten()))))

                pos_pd_idx = pd.MultiIndex.from_tuples(pos_ind_tup,
                                                       names=['day', 'epoch', 'timestamp', 'time'])

                pos_pd_col = pd.MultiIndex.from_tuples(
                    [('lin_dist_well', 'well_center'),
                     ('lin_dist_well', 'well_left'), ('lin_dist_well', 'well_right'),
                     ('lin_vel', 'well_center'), ('lin_vel', 'well_left'),
                     ('lin_vel', 'well_right'), ('seg_idx', 'seg_idx')])

                posdata_all_df = pd.DataFrame(posdata_all, index=pos_pd_idx, columns=pos_pd_col)
                posdata_all_df = posdata_all_df.sort_index(level='time')

                self.data = self.data.append(posdata_all_df)


        # make sure column multi-index is sorted for multi-slicing
        # self.data.sort_index(axis=1, inplace=True)

    def __call__(self):
        for day in self.days:
            for epoch in self.anim.epochs:
                # get the list of unique timebins
                day_epoch_data = self.data.loc[day, epoch]

                pos_raw = day_epoch_data.values
                pos_raw_timestamps = day_epoch_data.index.get_level_values('timestamp')

                for row_id in range(len(pos_raw)):
                    row = pos_raw[row_id]
                    segind = row[6]     # seg_idx
                    postimestamp = int(pos_raw_timestamps[row_id])    # timestamp

                    if segind == 1:
                        pos_conv = row[0]          # ['lin_dist_well', 'well_center']
                        veldata = row[3]           # ['lin_vel', 'well_center']
                    elif (segind == 2) or (segind == 3):
                        pos_conv = row[1] + 150    # ['lin_dist_well', 'well_left']
                        veldata = row[4]           # ['lin_vel', 'well_left']
                    elif (segind == 4) or (segind == 5):
                        pos_conv = row[2] + 300    # ['lin_dist_well', 'well_right']
                        veldata = row[5]           # ['lin_vel', 'well_right']
                    else:
                        raise DataContentError("Segment Index ({}) is invalid, expects 1 to 5 for W-Track.".
                                               format(segind))

                    yield LinearPosPoint(timestamp=postimestamp, x=pos_conv, vel=veldata)


class PosData:
    """	Returns blocks of 2D raw pos data specified by AnimalInfo
    
    Opens all files specified by animalinfo and creates a generator
    that can be called/polled to return pos data in bundles of the set timestep
    
    Args:
        anim (AnimalInfo): Defines the animal data and locations
        timestep	time interval for generator to return on each call
    """
    DAY_IND = 0
    DATA_IND = 1
    PATH_IND = 2

    def __init__(self, anim, timestep):
        self.data = []
        self.timestep = timestep
        self.anim = anim
        path_list = anim.get_pos_paths()

        posdata_days = {}
        for path in path_list:
            mat = loadmat(path[self.PATH_IND])
            tmp_posdata = posdata_days.setdefault(path[self.DAY_IND], [])

            filedata = mat['filedata']
            tmp_posdata.append(filedata['params'][0, 0][:, [0, 7, 8]])

        for key in posdata_days.keys():
            tmp_posdata = posdata_days[key]

            total_num = reduce(lambda x, y: x + y.shape[0], tmp_posdata, 0)

            posdata = np.empty([total_num, 3], dtype=np.dtype(np.float32))

            insert_cur = 0
            for pos_sub in tmp_posdata:
                posdata[insert_cur:insert_cur + pos_sub.shape[0], :] = pos_sub
                insert_cur += pos_sub.shape[0]

            # Sort the timestamps within each day
            posdata = posdata[np.argsort(posdata[:, 0]), :]
            self.data.append([key, posdata])

    def __call__(self):
        # extract number of days into a unique day list
        day_list = map(lambda x: x[self.DAY_IND], self.data)
        days = list(set(day_list))
        for day in days:
            day_data = filter(lambda x: x[self.DAY_IND] == day, self.data)
            for day_i in range(len(day_data)):
                posdata = day_data[day_i][self.DATA_IND]
                day_start_time = self.anim.times[day][0][0]
                day_end_time = self.anim.times[day][-1][-1]

                cur_time = day_start_time

                next_time = cur_time

                pos_cursor = 0

                while cur_time <= day_end_time:
                    next_time = cur_time + self.timestep
                    poslist = []
                    while pos_cursor < len(posdata) and posdata[pos_cursor, 0] <= next_time:
                        poslist.append((posdata[pos_cursor, 0], (posdata[pos_cursor, 1], posdata[pos_cursor, 2])))
                        pos_cursor += 1
                    epoch = self.anim.calc_epoch_state(day, cur_time)
                    new_timepoint = PosTimepoint(day, epoch, (cur_time, next_time), poslist)
                    cur_time = next_time
                    yield new_timepoint

