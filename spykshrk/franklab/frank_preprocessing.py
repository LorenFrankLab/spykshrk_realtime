import multiprocessing
import os

import pandas as pd
from IPython.terminal.debugger import TerminalPdb

import franklab.franklab_data as fd

bp = TerminalPdb(color_scheme='linux').set_trace


class SpikeParameterGenerator:

    def __init__(self, anim: fd.FrankAnimalInfo):
        self.anim = anim

    def extract_amplitudes_from_waves(self, dates, parallel_instances=1):
        if parallel_instances == 1:
            for date in dates:
                self._single_date_extract_amplitudes_from_waves(date)

        elif parallel_instances > 1:
            p = multiprocessing.Pool(parallel_instances)
            p.map(self._single_date_extract_amplitudes_from_waves, dates, chunksize=1)
            p.close()

    def _single_date_extract_amplitudes_from_waves(self, date):

        with fd.SpikeWaveData(self.anim, date) as spk_wave_data, \
                pd.HDFStore(os.path.join(self.anim.data_dir,
                                         self.anim.get_base_name(date, 'spikeamp'))) as date_store:

            # Right now getting the epoch and ntrodes from the spike hdf file,
            # in the future should get that from the animal metadata
            for epoch in spk_wave_data.get_epochs():
                for ntrode in spk_wave_data.get_ntrodes():
                    spkwave_data = spk_wave_data.get_data(epoch, ntrode)
                    spkamp_data = spkwave_data.max(axis=0)

                    date_store[SpikeParameterGenerator.get_hierarchy_str(epoch, ntrode) + 'data'] = spkamp_data

    @staticmethod
    def get_hierarchy_str(epoch, ntrode):
        return (SpikeParameterGenerator.get_hierarchy_base_str() +
                SpikeParameterGenerator.get_hierarchy_epoch_str(epoch) +
                SpikeParameterGenerator.get_hierarchy_ntrode_str(ntrode))

    @staticmethod
    def get_hierarchy_base_str():
        return '/preprocessing/FeatureExtraction/'

    @staticmethod
    def get_hierarchy_epoch_str(epoch):
        return 'e{:02d}/'.format(epoch)

    @staticmethod
    def get_hierarchy_ntrode_str(ntrode):
        return 't{:02d}/'.format(ntrode)


