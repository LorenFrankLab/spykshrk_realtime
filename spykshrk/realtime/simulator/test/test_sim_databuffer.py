from unittest import TestCase

from spykshrk.realtime.simulator.nspike_data import AnimalInfo, EEGDataTimeBlockStream, SpkDataStream, PosMatDataStream
from spykshrk.realtime.simulator.sim_databuffer import SimDataBuffer

from spykshrk.realtime.datatypes import LFPPoint

from franklab.franklab_data import FrankAnimalInfo, LFPData, RawPosData, SpikeAmpData
from spykshrk.realtime.simulator.franklab_datastream import LFPDataStream, RawPosDataStream, SpikeDataStream


class TestNSpikeSimDataBuffer(TestCase):

    def setUp(self):
        anim_dir = '/home/daliu/data/'

        timescale = 10000

        animal_name = 'test'
        days = [2]
        tetrodes = [5, 11, 12, 14, 19]

        epoch_encode = [1]
        new_data = True
        self.anim = AnimalInfo(base_dir=anim_dir,
                               name=animal_name,
                               days=days,
                               tetrodes=tetrodes,
                               epochs=epoch_encode,
                               timescale=timescale,
                               new_data=new_data)

    def test_nspike_SimDataBuffer(self):

        eeg = EEGDataTimeBlockStream(self.anim, 100)
        spk = SpkDataStream(self.anim, 100)
        pos = PosMatDataStream(self.anim, 1000)

        databuffer = SimDataBuffer([eeg(), pos()])

        last_timestamp = 0
        for datacount, data in enumerate(databuffer()):
            self.assertTrue(data.timestamp >= last_timestamp)
            last_timestamp = data.timestamp

            if isinstance(data, LFPPoint):
                print(data)
                self.assertTrue(data.ntrode_id in self.anim.tetrodes,
                                msg='ntrode {:} not in anim tetrode list.'.format(data.ntrode_id, self.anim.tetrodes))

        self.assertTrue(datacount == 452007, msg='Datacount is actually {:}'.format(datacount))


class TestTrodesSimDataBuffer(TestCase):

    def test_franklab_SimDataBuffer(self):
        anim = FrankAnimalInfo('/opt/data36/jason/', 'kanye')

        lfp_data = LFPData(anim, '20160426')
        lfp_stream = LFPDataStream(lfp_data=lfp_data, epoch=2, ntrodes=list(range(1, 17)))

        pos_data = RawPosData(anim, '20160426')
        pos_stream = RawPosDataStream(pos_data=pos_data, epoch=2, recon_type='online')

        databuffer = SimDataBuffer([lfp_stream(), pos_stream()])

        last_timestamp = 0
        for datacount, data in enumerate(databuffer()):
            self.assertTrue(data.timestamp >= last_timestamp)
            last_timestamp = data.timestamp

        print(datacount)
