import logging
import os.path
import json
import struct
from collections import OrderedDict
from abc import ABCMeta, abstractmethod


class BinaryRecordsError(Exception):
    def __init__(self, value, **kwargs):
        self.value = value
        self.data = kwargs

    def __str__(self):
        return repr(self.value) + '\n' + repr(self.data)


class BinaryRecordCreateMessage:
    def __init__(self, manager_label, file_id, save_dir, file_prefix, file_postfix, rec_label_dict, rec_format_dict):
        self.manager_label = manager_label
        self.file_id = file_id
        self.save_dir = save_dir
        self.file_prefix = file_prefix
        self.file_postfix = file_postfix
        self.rec_label_dict = rec_label_dict
        self.rec_format_dict = rec_format_dict

    def __str__(self):
        return '{:}({:})'.format(self.__class__.__name__, self.__dict__)


class BinaryRecordTypeMessage:
    def __init__(self, manager_label, rec_id, rec_labels, rec_struct_fmt):
        self.manager_label = manager_label
        self.rec_id = rec_id
        self.rec_labels = rec_labels
        self.rec_struct_fmt = rec_struct_fmt

    def __str__(self):
        return '{:}({:})'.format(self.__class__.__name__, self.__dict__)


class RemoteBinaryRecordsManager:
    """Remote manager for file records.

    Only ONE of these classes should be instantiated per rank/process, per manager.

    This class can generate messages that can be used to register records
    with the root BinaryRecordsManager and can process create record file messages,
    but it does not handle communication to the root BinaryRecordsManager.
    Communication between remote and root managers must be explicitly performed by the developer."""
    def __init__(self, manager_label, local_rank, manager_rank=0):
        self._manager_label = manager_label
        self.rank = local_rank
        self._manager_rank = manager_rank

        self._local_files = None

    def create_register_rec_type_message(self, rec_id, rec_labels, rec_struct_fmt):
        if self._local_files is not None:
            raise BinaryRecordsError('Cannot add more record types after remote manager has created a file: '
                                     'rec_id {}, rec_label {}, rec_struct_fmt {}.'.format(rec_id,
                                                                                          rec_labels,
                                                                                          rec_struct_fmt))
        else:
            return BinaryRecordTypeMessage(manager_label=self._manager_label,
                                           rec_id=rec_id,
                                           rec_labels=rec_labels,
                                           rec_struct_fmt=rec_struct_fmt)

    def create_writer_from_message(self, create_message: BinaryRecordCreateMessage):

        if self._local_files is None:
            new_bin_writer = BinaryRecordsFileWriter(create_message=create_message, mpi_rank=self.rank)

            self._local_files = new_bin_writer

        else:
            # TODO Error checking to make sure message is compatible with existing file
            pass

        return self._local_files

    def close(self):
        for file_id, file in self._local_files.items():
            file.close()


class BinaryRecordsManager:
    """Managers FSData's file records"""
    def __init__(self, manager_label, save_dir='', file_prefix='', file_postfix=''):

        self._manager_label = manager_label
        self._save_dir = save_dir
        self._file_prefix = file_prefix
        self._file_postfix = file_postfix
        self._rec_format_dict = {}
        self._rec_label_dict = {}
        self._next_file_index = 1
        self._local_files = {}

    @property
    def manager_label(self):
        return self._manager_label

    @property
    def file_prefix(self):
        return self._file_prefix

    @file_prefix.setter
    def file_prefix(self, file_prefix):
        if self._next_file_index >= 2:
            raise BinaryRecordsError('Cannot modify filename (file_prefix) after manager has created a file!')
        else:
            self._file_prefix = file_prefix

    @property
    def save_dir(self):
        return self.save_dir

    @save_dir.setter
    def save_dir(self, save_dir):
        if self._next_file_index >= 2:
            raise BinaryRecordsError('Cannot modify save directory (save_dir) after manager has created a file!')
        else:
            self._save_dir = save_dir

    def register_rec_type_message(self, rec_type_message: BinaryRecordTypeMessage):
        if rec_type_message.manager_label != self._manager_label:
            raise BinaryRecordsError(('Trying to register record type with wrong manager, '
                                      'record manager_label={}, manager manager_label={}').
                                     format(rec_type_message.manager_label, self._manager_label))
        self.register_rec_type(rec_id=rec_type_message.rec_id,
                               rec_labels=rec_type_message.rec_labels,
                               rec_struct_fmt=rec_type_message.rec_struct_fmt)

    def register_rec_type(self, rec_id, rec_labels, rec_struct_fmt):
        if self._next_file_index >= 2:
            raise BinaryRecordsError('Cannot add more record types after manager has created a file. '
                                     'Tried to add {} {} {}'.format(rec_id, rec_labels, rec_struct_fmt))
        else:
            if rec_id in self._rec_format_dict:
                if (rec_labels != self._rec_label_dict[rec_id]) or (rec_struct_fmt != self._rec_format_dict[rec_id]):
                    raise BinaryRecordsError(('Record ID already exists and id or format does not match: '
                                              'old rec: ({}: {}, {}), new: ({}: {}, {})').
                                             format(rec_id,
                                                    self._rec_label_dict[rec_id],
                                                    self._rec_format_dict[rec_id],
                                                    rec_id,
                                                    rec_labels,
                                                    rec_struct_fmt))
            self._rec_format_dict.update({rec_id: rec_struct_fmt})
            self._rec_label_dict.update({rec_id: rec_labels})

    def new_local_writer(self):
        new_bin_writer = BinaryRecordsFileWriter(create_message=self.new_writer_message())

        self._local_files[self._next_file_index] = new_bin_writer
        self._next_file_index += 1
        return new_bin_writer

    def new_writer_message(self):
        create_message = BinaryRecordCreateMessage(manager_label=self._manager_label, file_id=self._next_file_index,
                                                   save_dir=self._save_dir, file_prefix=self._file_prefix,
                                                   file_postfix=self._file_postfix, rec_label_dict=self._rec_label_dict,
                                                   rec_format_dict=self._rec_format_dict)

        self._next_file_index += 1

        return create_message

    def close(self):
        for file_id, file in self._local_files.items():
            file.close()


class BinaryRecordsFileWriter:
    """File handler for a single Binary Records file.

    Current can only be created through a BinaryRecordCreateMessage (primarily for remote file creation).

    The file name will be defined by the BinaryRecordCreateMessage's attributes and the mpi_rank parameter if specified:
    <file_prefix>.<manager_label>.<mpi_rank>.<file_postfix>

    A Binary Records file consists of a JSON header prepended to the file that must define the following entries:
        file_prefix: The root file name that is shared if a data store spans multiple files
        file_id: A unique ID for the given file
        name: Descriptive label (shared across all files)
        rec_type_spec: The format (python struct - format character) of all possible recs

    What follows is a binary blob that contains a list of records with the following format:
        <rec_ind (uint32)> <rec_type (uint8)> <rec_data>

    Each record type with unique ID must be specified in rec_type_spec using python struct's format character syntax
    (don't prepend with a byte order character).

    Each record type has a fixed size that is implicitly defined by its format string.

    """

    @staticmethod
    def format_full_path(save_dir, file_prefix, mpi_rank, manager_label, file_postfix):
        file_path = os.path.join(save_dir, '{}.{:02d}.{}.{}'.format(file_prefix, mpi_rank,
                                                                    manager_label, file_postfix))

        return file_path

    def __init__(self, create_message: BinaryRecordCreateMessage, mpi_rank=None):
        self.manager_label = create_message.manager_label
        self._file_id = create_message.file_id
        self._mpi_rank = mpi_rank
        self._save_dir = create_message.save_dir
        self._file_prefix = create_message.file_prefix
        self._file_postfix = create_message.file_postfix

        self._file_path = self.format_full_path(save_dir=self._save_dir,
                                                file_prefix=self._file_prefix,
                                                mpi_rank=self._mpi_rank,
                                                manager_label=self.manager_label,
                                                file_postfix=self._file_postfix)

        self._file_handle = open(self._file_path, 'wb')
        self._rec_label_dict = create_message.rec_label_dict
        self._rec_format_dict = create_message.rec_format_dict
        self._header_json = json.dumps(OrderedDict([('file_prefix', self._file_prefix),
                                                    ('file_id', self._file_id),
                                                    ('mpi_rank', self._mpi_rank),
                                                    ('manager_label', self.manager_label),
                                                    ('rec_formats', self._rec_format_dict),
                                                    ('rec_labels', self._rec_label_dict)]))

        # write header to file
        self._file_handle.write(bytearray(self._header_json, encoding='utf-8'))
        print('wrote json header',self._mpi_rank)

        self._rec_counter = 0

    @property
    def rec_format_dict(self):
        return self._rec_format_dict

    @rec_format_dict.setter
    def rec_format_dict(self, rec_format_dict):
        raise BinaryRecordsError('Cannot modify rec_format_dict after object has been created!')

    def write_rec(self, rec_type_id, *args):
        try:
            rec_bytes = struct.pack('=QB' + self.rec_format_dict[rec_type_id], self._rec_counter, rec_type_id, *args)
            self._file_handle.write(rec_bytes)
            self._rec_counter += 1
        except struct.error as ex:
            raise BinaryRecordsError('Data does not match record {}\'s data format.'.format(rec_type_id),
                                     rec_type_id=rec_type_id, rec_type_fmt=self.rec_format_dict[rec_type_id],
                                     rec_data=args, orig_error=ex) from ex

    def __del__(self):
        self._file_handle.close()

    @property
    def closed(self):
        return self._file_handle.closed

    def close(self):
        self._file_handle.close()
