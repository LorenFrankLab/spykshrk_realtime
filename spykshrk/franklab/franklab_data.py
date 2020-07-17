import os
import time
import re
import pandas as pd
import warnings
import numpy as np

from spykshrk.franklab.data_containers import DataFrameClass

from abc import ABCMeta, abstractmethod

try:
    from IPython.terminal.debugger import TerminalPdb
    bp = TerminalPdb(color_scheme='linux').set_trace
except AttributeError as err:
    # print('Warning: Attribute Error ({}), disabling IPython TerminalPdb.'.format(err))
    bp = lambda: None

idx = pd.IndexSlice


class FrankFileFormatError(RuntimeError):
    pass


class FrankFileFormatWarning(RuntimeWarning):
    pass


class FrankHDFFormatError(RuntimeError):
    pass


class FrankFilenameParser:

    frank_filename_re = re.compile('^(?:(\d*)_){0,1}(?:([a-zA-Z0-9]*)_{0,1})(\w*)\.(\w*)$')

    def __init__(self, filename_str):
        self.filename = filename_str
        filename_match = self.frank_filename_re.match(self.filename)

        if filename_match is not None:
            filename_groups = filename_match.groups()
            self.date = filename_groups[0]
            if self.date is not None and self.date != '':
                try:
                    self.date_expand = FrankFilenameParser.expand_date_str(self.date)
                except ValueError:
                    raise FrankFileFormatError('Filename ({}) date field ({}) is not a true date.'.
                                               format(self.filename, self.date))

            self.anim_name = filename_groups[1]
            self.datatype = filename_groups[2]
            self.ext = filename_groups[3]

    @staticmethod
    def expand_date_str(date_str):
        return time.strptime(date_str, '%Y%m%d')


class FrankAnimalInfo:
    def __init__(self, base_dir, anim_name):
        self.base_dir = base_dir
        self.anim_name = anim_name

        self.data_dir = FrankAnimalInfo._get_analysis_dir(self.base_dir, self.anim_name)
        self.data_paths = FrankAnimalInfo._get_data_path_df(self.data_dir)

    def get_dates(self):
        return list(self.data_paths['date'].unique())

    def get_data_path(self, date, datatype):

        if date is None:
            path = self.data_paths[(self.data_paths['datatype'] == datatype) &
                                   (self.data_paths['date'].isna())]
        else:
            path = self.data_paths[(self.data_paths['datatype'] == datatype) &
                                   (self.data_paths['date'] == date)]

        if len(path) < 1:
            raise FrankFileFormatError('Animal ({}) missing ({}) data for date ({}).'.format(self.anim_name,
                                                                                             datatype,
                                                                                             date))

        if len(path) > 1:
            raise FrankFileFormatError(('Animal ({}) has more than one date ({}) for ({}) data, '
                                        'likely a bug or corruption of the animal info.').format(self.anim_name,
                                                                                                 datatype,
                                                                                                 date))

        return path.iloc[0].path

    def get_avaliable_datatypes(self):
        return self.data_paths['datatype'].unique()

    def get_all_datatype(self, datatype):
        return self.data_paths.query('datatype == @datatype')

    def get_datatype_info(self, datatype):
        return FrankDataInfo(self, datatype)

    @staticmethod
    def _get_analysis_dir(base_dir, anim_name):
        return os.path.join(base_dir, anim_name, 'analysis')

    @staticmethod
    def _get_processing_dir(base_dir, anim_name):
        return os.path.join(base_dir, anim_name, 'processing')

    @staticmethod
    def _get_data_path_df(data_path):
        data_path_entries = os.scandir(data_path)
        path_df = pd.DataFrame(columns=['animal', 'date', 'datatype', 'ext', 'path'])

        for path_entry in data_path_entries:    # type: os._DummyDirEntry
            filename_parser = FrankFilenameParser(path_entry.name)

            if filename_parser.ext != 'h5':
                warnings.warn('Found file that does not have h5 extension ({}), skipping.'.
                              format(filename_parser.filename))

            path_df = path_df.append(dict(zip(path_df.columns, [filename_parser.anim_name,
                                                                filename_parser.date,
                                                                filename_parser.datatype,
                                                                filename_parser.ext,
                                                                path_entry.path])),
                                     ignore_index=True)

        path_df = path_df.sort_values(['animal', 'date', 'datatype']).reset_index(drop=True)

        return path_df

    @staticmethod
    def _get_hdf_path_components(path):
        hdf_group_re = re.compile('^(/[a-zA-Z0-9]*)([a-zA-Z0-9/]*?)(/[a-zA-Z0-9]*)$')
        return hdf_group_re.match(path).groups()

    def get_base_name(self, date, datatype):
        return date + '_' + self.anim_name + '_' + datatype + '.h5'


class FrankDataInfo:
    def __init__(self, anim: FrankAnimalInfo, datatype):
        self.datatype_paths = anim.get_all_datatype(datatype)

        self.entries = None

        self.reload_entries()

    def reload_entries(self):
        datatype_key_list = []
        for path_row in self.datatype_paths.itertuples():
            with pd.HDFStore(path_row.path) as data_store:
                for key in data_store.keys():
                    entry = [path_row.Index]
                    entry.append(path_row.date)
                    entry.extend(self._split_hdf_group(key))
                    key_storer = data_store.get_storer(key)
                    try:
                        user_key = key_storer.attrs.user_key
                    except AttributeError:
                        user_key = None
                    entry.append(user_key)
                    classtype = key_storer.attrs.classtype
                    entry.append(classtype.__name__)
                    entry.append(classtype)
                    datatype_key_list.append(entry)

        self.entries = pd.DataFrame(datatype_key_list, columns=['file_ind', 'date', 'base', 'group', 'label',
                                                                'user_key', 'classtype_str', 'classtype'])

    def load_single_dataset_ind(self, ind):
        data_entry = self.entries.loc[ind]
        hdf_filename = self.datatype_paths.loc[data_entry['file_ind']]['path']
        data = DataFrameClass._from_hdf_store(hdf_filename, data_entry['base'],
                                              data_entry['group'], data_entry['label'])
        return data

    def save_single_data(self, base, group, label, data, overwrite=False):
        if isinstance(data, DataFrameClass):

            try:
                dates = data.get_days()
            except AttributeError:
                dates = None

            if dates is not None:
                if all([date in self.entries['date'].unique() for date in dates]):
                    print(dates, all([date in self.entries['date'].unique() for date in dates]))
                    # split dates into appropriate file
                    data_date_grp = data.groupby('day')
                    for date, data_date in data_date_grp:
                        file_path = self.datatype_paths.query('date == @date').path[0]
                        entry_collision = []
                        if overwrite:
                            data_date._to_hdf_store(file_path, base, group, label, overwrite=True)
                        else:
                            if self.has_entry(base, group, label, date):
                                entry_collision.append((date, base, group, label))
                            else:
                                data_date._to_hdf_store(file_path, base, group, label)
                elif [None] == self.entries['date'].unique():
                    #file_path = self.datatype_paths.query('date != date').path[0]
                    file_path = self.datatype_paths.iloc[0].path
                    data._to_hdf_store(file_path, base, group, label, overwrite=overwrite)
                else:
                    raise FrankHDFFormatError('Date from data ({}) do not match dates '
                                              'from file ({}).'.format(dates, self.entries['date'].unique()))


    def get_file_entry_from_ind(self, ind):
        return self.datatype_paths.loc[ind, :]

    def get_date_from_ind(self, ind):
        return self.get_file_entry_from_ind(ind).date[0]

    def has_entry(self, base, group, label, date=None):

        if date is None:
            en_match = self.entries.query('base == @os.path.normpath(@base) and '
                                          'group == @os.path.normpath(@group) and '
                                          'label == @os.path.normpath(@label) and '
                                          'date != date')
        else:
            en_match = self.entries.query('base == @os.path.normpath(@base) and '
                                          'group == @os.path.normpath(@group) and '
                                          'label == @os.path.normpath(@label) and '
                                          'date == @date')

        if len(en_match) == 1:
            return True
        elif len(en_match) == 0:
            return False
        else:
            raise FrankHDFFormatError('Dataset ({}) ({}) ({}) ({}) has duplicate entry in table.  FrankDataInfo or '
                                      'underlying may be corrupt,'.format(date, base, group, label))

    def get_entry(self, base, group, label, date=None):
        if date is None:
            en_match = self.entries.query('base == @os.path.normpath(@base) and '
                                          'group == @os.path.normpath(@group) and '
                                          'label == @os.path.normpath(@label) and '
                                          'date != date')
        else:
            en_match = self.entries.query('base == @os.path.normpath(@base) and '
                                          'group == @os.path.normpath(@group) and '
                                          'label == @os.path.normpath(@label) and '
                                          'date == @date')

        if len(en_match) == 1:
            return en_match
        elif len(en_match) == 0:
            return None
        else:
            raise FrankHDFFormatError('Dataset ({}) ({}) ({}) ({}) has duplicate entry in table.  FrankDataInfo or '
                                      'underlying may be corrupt,'.format(date, base, group, label))

    def get_entry_path(self, base, group, label, date=None):
        return self.get_entry(base, group, label, date).path[0]

    @staticmethod
    def _split_hdf_group(group_str):
        grp_split = re.compile('^(/[a-zA-Z0-9]*)/([a-zA-Z0-9/]*?)/([a-zA-Z0-9/_]*)$')
        grp_match = grp_split.match(group_str)
        if grp_match is None:
            raise FrankFileFormatError('HDF group ({}) cannot be parsed into [base]/[group]/[label].'.format(group_str))
        groups = grp_match.groups()
        groups = [os.path.normcase(path) for path in groups]

        return groups


class BaseData(metaclass=ABCMeta):

    def __enter__(self):
        return self

    def __del__(self):
        self.store.close()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.store.close()

    @property
    @abstractmethod
    def store(self):
        pass

    @store.setter
    @abstractmethod
    def store(self, value):
        pass

    @staticmethod
    @abstractmethod
    def get_hierarchy_base_str():
        return '/preprocessing/Module/'

    @staticmethod
    def get_hierarchy_epoch_str(epoch):
        return 'e{:02d}/'.format(epoch)


class RawPosData(BaseData):

    store = None

    def __init__(self, anim: FrankAnimalInfo, date):
        self.anim = anim
        self.date = date
        self.path = self.anim.get_data_path(date, 'rawpos')

        self.store = pd.HDFStore(self.path, mode='r')

    def get_data(self, epoch, recon_type):
        return self.store[self.get_hiearchy_str(epoch, recon_type) + 'data']

    @staticmethod
    def get_hierarchy_base_str():
        return '/preprocessing/Position/'

    @classmethod
    def get_hiearchy_str(cls, epoch, recon_type):
        return (cls.get_hierarchy_base_str() +
                cls.get_hierarchy_epoch_str(epoch) +
                recon_type + '/')


class LFPData(BaseData):

    store = None

    def __init__(self, anim: FrankAnimalInfo, date):
        self.anim = anim
        self.date = date
        self.path = self.anim.get_data_path(date, 'lfp')

        self.store = pd.HDFStore(self.path, mode='r')

    def get_data(self, epoch, ntrodes=None):
        lfp_df = self.store[self.get_hierarchy_str(epoch) + 'data']
        if ntrodes is None:
            return lfp_df
        else:
            return lfp_df.loc[:, idx[ntrodes, :]]

    @staticmethod
    def get_hierarchy_base_str():
        return '/preprocessing/LFP/'

    @classmethod
    def get_hierarchy_str(cls, epoch):
        return (cls.get_hierarchy_base_str() +
                cls.get_hierarchy_epoch_str(epoch))


class SpikeBaseData(BaseData, metaclass=ABCMeta):

    @staticmethod
    def get_hierarchy_ntrode_str(ntrode):
        return 't{:02d}/'.format(ntrode)

    def get_epochs(self):
        epoch_list = []
        epoch_re = re.compile('^e(\d*)$')
        for epoch_node in self.store.get_node(self.get_hierarchy_base_str()):
            epoch_list.append(int(epoch_re.match(epoch_node._v_name).groups()[0]))

        return epoch_list

    def get_ntrodes(self):

        ntrode_set = set()
        ntrode_re = re.compile('^t(\d*)$')
        for epoch_node in self.store.get_node(self.get_hierarchy_base_str()):
            for ntrode_node in epoch_node:
                ntrode_set.add(int(ntrode_re.match(ntrode_node._v_name).groups()[0]))

        ntrode_list = list(ntrode_set)
        ntrode_list.sort()
        return ntrode_list

    def get_data(self, epoch, ntrode):
        return self.store[self.get_hierarchy_str(epoch, ntrode) + 'data']

    @classmethod
    def get_hierarchy_str(cls, epoch, ntrode):
        return (cls.get_hierarchy_base_str() +
                cls.get_hierarchy_epoch_str(epoch) +
                cls.get_hierarchy_ntrode_str(ntrode))


class SpikeAmpData(SpikeBaseData):

    store = None

    def __init__(self, anim: FrankAnimalInfo, date):
        self.anim = anim
        self.date = date

        self.path = self.anim.get_data_path(date, 'spikeamp')

        self.store = pd.HDFStore(self.path, mode='r')

    @staticmethod
    def get_hierarchy_base_str():
        return '/preprocessing/FeatureExtraction/'


class SpikeWaveData(SpikeBaseData):

    store = None

    def __init__(self, anim: FrankAnimalInfo, date):
        self.anim = anim
        self.date = date

        self.path = self.anim.get_data_path(date, 'spikeamp')

        self.store = pd.HDFStore(self.path, mode='r')


    @staticmethod
    def get_hierarchy_base_str():
        return '/preprocessing/EventWaveform/'


