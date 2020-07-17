import numpy as np
import pandas as pd
from collections import OrderedDict
from franklab.franklab_data import SpikeBaseData, LFPData, RawPosData
from spykshrk.realtime.datatypes import SpikePoint, LFPPoint, RawPosPoint


class RawPosDataStream:

    def __init__(self, pos_data: RawPosData, epoch, recon_type):
        self.pos_data = pos_data
        self.epoch = epoch
        self.recon_type = recon_type
        self.pos_df = pos_data.get_data(epoch, recon_type)

    def __iter__(self):
        return self

    def __call__(self):
        for pos_tuple in self.pos_df.itertuples():
            yield RawPosPoint(timestamp=pos_tuple.Index, x1=pos_tuple.x1, y1=pos_tuple.y1, x2=pos_tuple.x2,
                              y2=pos_tuple.y2, camera_id=None)


class LFPDataStream:

    def __init__(self, lfp_data: LFPData, epoch, ntrodes):
        self.lfp_data = lfp_data
        self.epoch = epoch
        self.ntrodes = ntrodes

        self.lfp_df = self.lfp_data.get_data(epoch=epoch, ntrodes=ntrodes)

    def __iter__(self):
        return self

    def __call__(self):

        data_values = self.lfp_df.values

        # create column index for only first channel
        col_ind_dict = OrderedDict()
        for col_ind, ntrode in enumerate(self.lfp_df.columns):
            col_ind_dict.setdefault(int(ntrode[0]), col_ind)

        for row_ind, timestamp in enumerate(self.lfp_df.index):
            for ntrode, col_ind in col_ind_dict.items():
                yield LFPPoint(timestamp, self.ntrodes.index(ntrode), data_values[row_ind, col_ind])


class SpikeDataStream:

    def __init__(self, spike_data: SpikeBaseData, epoch, ntrodes):
        self.spike_data = spike_data
        self.epoch = epoch
        self.ntrodes = ntrodes

        self.spike_df = pd.DataFrame()

        for ntrode in ntrodes:

            ntrode_spk_data = spike_data.get_data(epoch, ntrode)

            ntrode_spk_data.index = pd.MultiIndex.from_product(([ntrode], ntrode_spk_data.index),
                                                               names=['ntrode', 'timestamp'])

            self.spike_df = self.spike_df.append(ntrode_spk_data)

        self.spike_df = self.spike_df.sort_index(level='timestamp')

    def __iter__(self):
        return self

    def __call__(self):
        for spk_tuple in self.spike_df.itertuples():
            yield SpikePoint(spk_tuple[0][1], self.ntrodes.index(spk_tuple[0][0]), spk_tuple[1:])


class SpikeDataStreamSlow:

    def __init__(self, spike_data: SpikeBaseData, epoch, ntrodes):
        self.spike_data = spike_data
        self.epoch = epoch
        self.ntrodes = ntrodes

        self.data = dict([(ntrode, spike_data.get_data(epoch, ntrode)) for ntrode in ntrodes])

    def __iter__(self):
        return self

    def __call__(self):
        ntrode_times = np.array([self.data[ntrode].index[0] for ntrode in self.ntrodes], dtype='float64')
        ntrode_pointer = [0] * len(self.ntrodes)

        while not (~np.isfinite(ntrode_times)).all():
            ntrode_ind = np.argmin(ntrode_times)
            cur_ntrode = self.ntrodes[ntrode_ind]

            spk_entry = self.data[cur_ntrode].iloc[ntrode_pointer[ntrode_ind]]

            ntrode_pointer[ntrode_ind] += 1
            if ntrode_pointer[ntrode_ind] >= len(self.data[cur_ntrode]):
                ntrode_times[ntrode_ind] = float('inf')
            else:
                ntrode_times[ntrode_ind] = self.data[cur_ntrode].index[ntrode_pointer[ntrode_ind]]

            yield SpikePoint(spk_entry.name, ntrode_ind, spk_entry.values.tolist())


