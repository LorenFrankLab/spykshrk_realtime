import json
import os
import struct
from collections import OrderedDict

import pandas as pd
from mpi4py import MPI

from spykshrk.realtime.realtime_logging import PrintableMessage


class TimingSystemError(Exception):
    def __init__(self, value, **kwargs):
        self.value = value
        self.data = kwargs

    def __str__(self):
        return repr(self.value) + '\n' + repr(self.data)


class TimingMessage(PrintableMessage):
    msgheader_format = '=12sqi'
    msgheader_format_size = struct.calcsize(msgheader_format)
    timept_format = '=id'
    timept_format_size = struct.calcsize(timept_format)

    def __init__(self, label, timestamp=0, start_rank=-1, timing_data=None):
        """

        Args:
            label:
            start_rank:
        """

        if len(label) > 12:
            raise TimingSystemError('Timing label {}: label must be 10 characters or less.'.format(label))

        if isinstance(label, str):
            self._label = bytes(label, 'utf-8')
        elif isinstance(label, bytes) or isinstance(label, bytearray):
            self._label = label

        self._timestamp = timestamp

        if timing_data is None:
            self._timing_data = [(start_rank, MPI.Wtime())]
        else:
            self._timing_data = timing_data

    def record_time(self, rank):
        self._timing_data.append((rank, MPI.Wtime()))

    @classmethod
    def unpack(cls, message_bytes):
        """

        Args:
            message_bytes:

        Returns:
            TimingMessage: The de-serialized TimingMessage object.
        """

        label, timestamp, num_time_pt = struct.unpack(cls.msgheader_format,
                                                      message_bytes[0:cls.msgheader_format_size])

        num_time_pt = int(num_time_pt)
        unpacked = struct.unpack(cls.timept_format[0] + cls.timept_format[1:] *
                                 num_time_pt, message_bytes[cls.msgheader_format_size:
                                                            cls.msgheader_format_size +
                                                            cls.timept_format_size*num_time_pt])

        timing_data = []
        for pt_ii in range(num_time_pt):
            timing_data.append((unpacked[pt_ii*2], unpacked[pt_ii*2+1]))

        return cls(label=label, timestamp=timestamp, timing_data=timing_data)

    def pack(self):
        return struct.pack(self.msgheader_format + self.timept_format[1:] * len(self._timing_data),
                           self._label, self._timestamp, len(self._timing_data),
                           *[it for sub in self._timing_data for it in sub])

    @property
    def label(self):
        return self._label.split(b'\0', 1)[0].decode()

    @label.setter
    def label(self, label):
        raise TimingSystemError('Cannot modify TimingMessage label.')

    @property
    def timestamp(self):
        return self._timestamp

    @timestamp.setter
    def timestamp(self, label):
        raise TimingSystemError('Cannot modify TimingMessage timestamp.')

    @property
    def timing_data(self):
        return self._timing_data

    @timing_data.setter
    def timing_data(self, timing_data):
        raise TimingSystemError('Cannot modify TimingMessage timing_data.')

    def __eq__(self, other):
        return ((self.label == other.label) and
                (self.timestamp == other.timestamp) and
                (self.timing_data == other.timing_data))


class CreateTimingFileMessage(PrintableMessage):
    def __init__(self, save_dir, file_prefix, file_postfix):
        self.save_dir = save_dir
        self.file_prefix = file_prefix
        self.file_postfix = file_postfix


class TimingFileWriter:

    header_format = '=Qi'

    @staticmethod
    def format_full_path(save_dir, file_prefix, mpi_rank, file_postfix):
        file_path = os.path.join(save_dir, '{}.{:02d}.{}'.format(file_prefix, mpi_rank,
                                                                 file_postfix))

        return file_path

    def __init__(self, save_dir, file_prefix, mpi_rank, file_postfix):
        self._save_dir = save_dir
        self._file_prefix = file_prefix
        self._mpi_rank = mpi_rank
        self._file_postfix = file_postfix

        self._file_path = self.format_full_path(save_dir=self._save_dir, file_prefix=self._file_prefix,
                                                mpi_rank=self._mpi_rank, file_postfix=self._file_postfix)

        self._header_json = json.dumps(OrderedDict([('file_type', 'timing'),
                                                    ('file_prefix', self._file_prefix),
                                                    ('mpi_rank', self._mpi_rank),
                                                    ('header_fmt', self.header_format),
                                                    ('msgheader_fmt', TimingMessage.msgheader_format),
                                                    ('timept_fmt', TimingMessage.timept_format)
                                                    ]))

        self._file_handle = open(self._file_path, 'wb')

        self._file_handle.write(bytearray(self._header_json, encoding='utf-8'))

        self._rec_counter = 0

    @classmethod
    def create_from_mesage(cls, create_message: CreateTimingFileMessage, rank):
        return cls(save_dir=create_message.save_dir,
                   file_prefix=create_message.file_prefix,
                   mpi_rank=rank,
                   file_postfix=create_message.file_postfix)

    def write_timing_message(self, timing_msg: TimingMessage):
        msg_bytes = timing_msg.pack()
        header_bytes = struct.pack(self.header_format, self._rec_counter, len(msg_bytes))
        self._file_handle.write(header_bytes + msg_bytes)

        self._rec_counter += 1

    def __del__(self):
        self._file_handle.close()

    @property
    def closed(self):
        return self._file_handle.closed

    def close(self):
        self._file_handle.close()


class TimingFileReader:
    def __init__(self, save_dir, file_prefix, mpi_rank, file_postfix):
        self._save_dir = save_dir
        self._file_prefix = file_prefix
        self._mpi_rank = mpi_rank
        self._file_postfix = file_postfix

        self._file_path = TimingFileWriter.format_full_path(save_dir=self._save_dir, file_prefix=self._file_prefix,
                                                            mpi_rank=self._mpi_rank, file_postfix=self._file_postfix)

        self._file_handle = open(self._file_path, 'rb')

        self._header_bytes = None
        self._data_start_byte = None
        self._extract_json_header()
        self._header = json.loads(self._header_bytes.decode('utf-8'))

    def _extract_json_header(self):
        self._file_handle.seek(0)
        self._header_bytes = bytearray()

        read_byte = self._file_handle.read(1)
        if read_byte != b'{':
            raise TimingSystemError('Not a Binary Records file, JSON header not found at first byte.',
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
            elif len(self._header_bytes) >= 1000:
                raise TimingSystemError('Could not find end of JSON header before 1000 byte header limit.',
                                        file_path=self._file_path)

            # read next byte
            read_byte = self._file_handle.read(1)

        if level != 0:
            raise TimingSystemError('Could not find end of JSON header before end of file.',
                                    file_path=self._file_path)

        self._data_start_byte = self._file_handle.tell()

        self.rec_id = -1

    def __iter__(self):
        return self

    def __next__(self):
        return_rec = self._read_record()
        if return_rec is None:
            raise StopIteration()
        else:
            return return_rec

    def _read_record(self):

        header_bytes = self._file_handle.read(struct.calcsize(self._header['header_fmt']))

        if not header_bytes:
            return None

        try:
            self.rec_id, msg_len = struct.unpack(self._header['header_fmt'], header_bytes)

            msg_bytes = self._file_handle.read(msg_len)
            msg = TimingMessage.unpack(msg_bytes)

        except struct.error as ex:
            raise TimingSystemError('File might be corrupted, record does not match format or unexpected EOF.',
                                    file_path=self._file_path, last_rec_id=self.rec_id)

        return self.rec_id, msg

    def convert_panda_labels(self):
        timing_labels = {}
        msg_id_labels = {}

        return_rec = self._read_record()
        while return_rec is not None:

            msg_id, msg = return_rec

            id_data = msg_id_labels.setdefault(msg.label, [])
            timing_data = timing_labels.setdefault(msg.label, [])

            id_data.append(msg_id)
            id_data_unrolled = [it for sublist in msg.timing_data for it in sublist]
            timing_data.append(id_data_unrolled)

            return_rec = self._read_record()

        timing_df_labels = {}
        for label in timing_labels.keys():
            data_id = msg_id_labels[label]
            timing_data = timing_labels[label]

            num_measurements = len(timing_data[0])/2
            if not num_measurements.is_integer():
                raise TimingSystemError('Timing data columns is not even (rank, time). Data is corrupted.')
            num_measurements = int(num_measurements)

            column_names = []
            for measurement_id in range(num_measurements):
                column_names.append('rank{}'.format(measurement_id))
                column_names.append('time{}'.format(measurement_id))

            timing_df_labels[label] = pd.DataFrame(data=timing_data, index=data_id, columns=column_names)


        return timing_df_labels
