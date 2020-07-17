import functools
import pandas as pd
import numpy as np
import struct


class TrodesBinaryFormatError(RuntimeError):
    pass


class TrodesBinaryReader:
    def __init__(self, path):
        self.path = path
        with open(path, 'rb') as file:
            # reading header
            # read first line to make sure its a trodes binary header
            line = file.readline()
            if line != b'<Start settings>\n':
                raise TrodesBinaryFormatError('File ({}) does not start with Trodes header <Start settings>'
                                            .format(path))
            # read header not including start and end tags
            self.header_params = {}
            for line_no, line in enumerate(file):
                if line == b'<End settings>\n':
                    break
                if line_no > 1000:
                    raise TrodesBinaryFormatError('File ({}) header over 1000 lines without <End settings>'
                                                .format(path))
                line_split = line.decode('utf-8').split(':', 1)
                self.header_params[line_split[0]] = line_split[1].strip()

            self.data_start_byte = file.tell()


class TrodesLFPBinaryLoader(TrodesBinaryReader):
    def __init__(self, path):
        super().__init__(path)

        # parse out basic header info
        self.rec_filename = self.header_params.get('Original_file')
        self.ntrode = self.header_params.get('nTrode_ID')
        self.channel = self.header_params.get('nTrode_channel')
        self.clockrate = self.header_params.get('Clock rate')
        self.voltage_scale = self.header_params.get('Voltage_scaling')
        self.decimation = self.header_params.get('Decimation')
        self.first_timestamp = self.header_params.get('First_timestamp')
        self.reference = self.header_params.get('Reference')
        self.low_pass_filter = self.header_params.get('Low_pass_filter')
        self.field_str = self.header_params.get('Fields')

        with open(self.path, 'rb') as file:
            file.seek(self.data_start_byte)
            byte_data = file.read()
            self.data = np.frombuffer(byte_data, dtype=np.dtype('int16'))


class TrodesTimestampBinaryLoader(TrodesBinaryReader):
    def __init__(self, path):
        super().__init__(path)

        # parse out basic header info
        self.byte_order = self.header_params.get('Byte_order')
        self.rec_filename = self.header_params.get('Original_file')
        self.clockrate = self.header_params.get('Clock rate')
        self.decimation = self.header_params.get('Decimation')
        self.time_offset = self.header_params.get('Time_offset')
        self.field_str = self.header_params.get('Fields')

        with open(self.path, 'rb') as file:
            file.seek(self.data_start_byte)
            byte_data = file.read()
            self.data = np.frombuffer(byte_data, dtype=np.dtype('uint32'))


class TrodesSpikeBinaryLoader(TrodesBinaryReader):
    def __init__(self, path):
        super().__init__(path)

        # parse out basic header info
        self.rec_filename = self.header_params.get('Original_file')
        self.ntrode = self.header_params.get('nTrode_ID')
        self.num_channels = int(self.header_params.get('num_channels'))
        self.clockrate = self.header_params.get('Clock rate')
        self.voltage_scale = self.header_params.get('Voltage_scaling')
        self.time_offset = self.header_params.get('Time_offset')
        self.threshold = self.header_params.get('Threshold')
        self.spike_invert = self.header_params.get('Spike_invert')
        self.reference = self.header_params.get('Reference')
        self.ref_ntrode = self.header_params.get('ReferenceNTrode')
        self.ref_chan = self.header_params.get('ReferenceChannel')
        self.filter = self.header_params.get('Filter')
        self.low_pass_filter = self.header_params.get('lowPassFilter')
        self.high_pass_filter = self.header_params.get('highPassFilter')
        self.field_str = self.header_params.get('Fields')

        # size of binary record size, assuming 40 samples per channel, 2 byte per sample, and 4 byte timestamp
        self.num_samples_per_spike = 40
        self.spike_rec_size = self.num_channels * self.num_samples_per_spike * 2 + 4
        self.unpack_format = 'I' + ('{:d}h'.format(self.num_samples_per_spike))*self.num_channels
        self.timestamps = []
        spikes_np = []
        with open(self.path, 'rb') as file:
            file.seek(self.data_start_byte)
            try:
                for spike_rec_bytes in iter(functools.partial(file.read, self.spike_rec_size), b''):
                    spike_rec = struct.unpack(self.unpack_format, spike_rec_bytes)
                    self.timestamps.append(spike_rec[0])
                    spikes_np.append(np.reshape(np.array(spike_rec[1:], dtype='int16'),
                                                (self.num_channels, self.num_samples_per_spike)))
            except struct.error:
                print(('TrodesSpikeBinaryLoader: for file {:} found an incomplete spike record, '
                       'truncating before record.').format(self.path))

        # convert list np.arrays into pure np.array
        spikes_np = np.array(spikes_np, dtype='int16')

        # transpose 3D array to fit intuitive pd.Panel notation
        # items (spike sample) x major_axis (spike timestamps) x minor_axis (channels)
        try:
            spikes_np = spikes_np.transpose(2, 0, 1)
        except ValueError:
            print(('TrodesSpikeBinaryLoader: for file {:} failed spike transpose, '
                   'likely no spikes in file?').format(self.path))

        try:
            # channels and ntrode are 1-indexed (really should be channel name), but samples are 0 index
            self.spikes = pd.Panel(spikes_np, items=pd.Index(range(self.num_samples_per_spike), name='spike_sample'),
                                   major_axis=pd.Index(self.timestamps, name='timestamp'),
                                   minor_axis=pd.Index(range(1, self.num_channels+1), name='channel'))
        except ValueError:
            print(('TrodesSpikeBinaryLoader: for file {:} failed panel creation, bad dimensions. '
                   'No spikes created from file.').format(self.path))
            self.spikes = pd.Panel(items=pd.Index([], name='spike_sample'),
                                   major_axis=pd.Index([], name='timestamp'),
                                   minor_axis=pd.Index([], name='channel'))
            


class TrodesPosBinaryLoader(TrodesBinaryReader):
    def __init__(self, path):
        super().__init__(path)

        # parse out basic header info
        self.threshold = self.header_params.get('threshold')
        self.dark = self.header_params.get('dark')
        self.clockrate = self.header_params.get('clockrate')
        self.field_str = self.header_params.get('Fields')

        # uint timestamp, 4 uint16 coordinates (x1, y1, x2, y2) for two diodes
        self.rec_size = 4 + 2 * 4
        self.unpack_format = 'IHHHH'

        pos_list = []

        with open(self.path, 'rb') as file:
            file.seek(self.data_start_byte)
            for pos_rec_bytes in iter(functools.partial(file.read, self.rec_size), b''):
                pos_rec = struct.unpack(self.unpack_format, pos_rec_bytes)
                pos_list.append(pos_rec)

        self.pos = pd.DataFrame(pos_list, columns=['timestamp', 'x1', 'y1', 'x2', 'y2']).set_index('timestamp')


class TrodesDIOBinaryLoader(TrodesBinaryReader):
    def __init__(self, path):

        super().__init__(path)

        # parse out basic header info
        self.original_file = self.header_params.get('Original_file')
        self.direction = self.header_params.get('Direction')
        self.id = self.header_params.get('ID')
        self.display_order = self.header_params.get('Display_order')
        self.clockrate = self.header_params.get('Clockrate')
        self.field_str = self.header_params.get('Fields')

        # uint32 timestamp, 1 byte state
        self.rec_size = 4 + 1
        self.unpack_format = 'IB'

        dio_list = []

        with open(self.path, 'rb') as file:
            file.seek(self.data_start_byte)
            for dio_rec_bytes in iter(functools.partial(file.read, self.rec_size), b''):
                dio_rec = struct.unpack(self.unpack_format, dio_rec_bytes)
                dio_list.append([dio_rec[0], bool(dio_rec[1])])

        self.dio = pd.DataFrame(dio_list, columns=['timestamp', 'state']).set_index('timestamp')

