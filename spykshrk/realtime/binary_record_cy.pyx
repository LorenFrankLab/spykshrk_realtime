# cython: profile=True
# cython: linetrace=True

import struct
import json
import os
import pandas as pd
import numpy as np
import errno
import sys


class BinaryRecordsError(Exception):
    def __init__(self, value, **kwargs):
        self.value = value
        self.data = kwargs

    def __str__(self):
        return repr(self.value) + '\n' + repr(self.data)

class BinaryRecordsFileReader:
    def __init__(self, save_dir, file_prefix, mpi_rank, manager_label, file_postfix, filemeta_as_col=True):
        self._save_dir = save_dir
        self._file_prefix = file_prefix
        self._mpi_rank = mpi_rank
        self._manager_label = manager_label
        self._file_postfix = file_postfix
        self._file_path = self.format_full_path(save_dir=self._save_dir,
                                                file_prefix=self._file_prefix,
                                                mpi_rank=self._mpi_rank,
                                                manager_label=self._manager_label,
                                                file_postfix=self._file_postfix)
        self._filemeta_as_col = filemeta_as_col

        if not os.path.isfile(self._file_path):
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), self._file_path)

        self._header_bytes = None
        self._data_start_byte = None
        self._file_handle = None
        self._header = None

    def start_record_reading(self):
        self._file_handle = open(self._file_path, 'rb')
        self._extract_json_header()
        self._header = json.loads(self._header_bytes.decode('utf-8'))

    def getsize(self):
        return os.path.getsize(self._file_path)

    @staticmethod
    def format_full_path(save_dir, file_prefix, mpi_rank, manager_label, file_postfix):
        file_path = os.path.join(save_dir, '{}.{:02d}.{}.{}'.format(file_prefix, mpi_rank,
                                                                    manager_label, file_postfix))
        return file_path

    @staticmethod
    def c_bytes_to_string(c_bytes):
        return c_bytes.split(b'\0')[0].decode('utf-8')

    def _extract_json_header(self):
        self._file_handle.seek(0)
        self._header_bytes = bytearray()

        read_byte = self._file_handle.read(1)
        if read_byte != b'{':
            raise BinaryRecordsError('Not a Binary Records file, JSON header not found at first byte.',
                                     file_path=self._file_path)

        level = 0
        while read_byte:
            self._header_bytes.append(ord(read_byte))
            if read_byte == b'{':
                level += 1
            elif read_byte == b'}':
                level -= 1

            if level == 0:
                break
            elif len(self._header_bytes) >= 10000:
                raise BinaryRecordsError('Could not find end of JSON header before 10000 byte header limit.',
                                         file_path=self._file_path)

            # read next byte
            read_byte = self._file_handle.read(1)

        if level != 0:
            raise BinaryRecordsError('Could not find end of JSON header before end of file.',
                                     file_path=self._file_path)

        self._data_start_byte = self._file_handle.tell()

    def __iter__(self):
        return self

    def __next__(self):
        return_rec = self._read_record()
        if not return_rec:
            raise StopIteration
        else:
            return return_rec

    def get_rec_types(self):
        return [int(key) for key in self._header['rec_formats']]

    def get_rec_formats(self):
        return {int(key): value for key, value in self._header['rec_formats'].items()}

    def get_rec_labels(self):
        return {int(key): value for key, value in self._header['rec_labels'].items()}

    def _read_record(self):
        # Assuming file_handle pointer is aligned to the beginning of a message
        # read header
        cpdef unsigned long long rec_ind
        cpdef unsigned char rec_type_id
        rec_head_bytes = self._file_handle.read(struct.calcsize('=QB'))
        if not rec_head_bytes:
            return None

        try:
            rec_ind, rec_type_id = struct.unpack('=QB', rec_head_bytes)

            rec_fmt = self._header['rec_formats'][str(rec_type_id)]
            rec_data_bytes = self._file_handle.read(struct.calcsize('='+rec_fmt))
            rec_data = struct.unpack('='+rec_fmt, rec_data_bytes)
        except struct.error as ex:
            raise BinaryRecordsError('File might be corrupted, record does not match format or unexpected EOF.',
                                     file_path=self._file_path)

        return rec_ind, rec_type_id, rec_data

    def convert_pandas(self):
        # always return to start of data
        self._file_handle.seek(self._data_start_byte)

        columns = self.get_rec_labels()
        rec_data = {key: [] for key in columns.keys()}

        rec_count = 0
        if self._filemeta_as_col:
            for rec in self:
                rec_data[rec[1]].append((rec[0],) + (self._mpi_rank,) + (self._file_path,) + rec[2])

            panda_frames = {key: pd.DataFrame(data=rec_data[key],
                                              columns=['rec_ind', 'mpi_rank', 'file_path'] + columns[key])
                            for key in columns.keys()}
        else:
            for rec in self:
                rec_data[rec[1]].append((rec[0],) + rec[2])

            panda_frames = {key: pd.DataFrame(data=rec_data[key], columns=['rec_ind'] + columns[key])
                            for key in columns.keys()}

        # Converting bytes into strings
        for id, table in panda_frames.items():
            if len(table) > 0:
                for col_name in table:
                    if table[col_name].dtype == np.object:
                        if isinstance(table[col_name].iloc[0], bytes):
                            table[col_name] = table[col_name].apply(self.c_bytes_to_string)

        panda_numeric_frames = {key: df.apply(pd.to_numeric, errors='ignore') for key, df in panda_frames.items()}

        return panda_numeric_frames


