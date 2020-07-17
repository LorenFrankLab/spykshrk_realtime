from unittest import TestCase

from spykshrk.realtime.simulator.franklab_datastream import LFPDataStream, SpikeDataStream, \
    SpikeDataStreamSlow, RawPosDataStream
from franklab.franklab_data import SpikeAmpData, FrankAnimalInfo, LFPData, RawPosData


class TestRawPosDataStream(TestCase):

    def test_iter(self):
        anim = FrankAnimalInfo('/opt/data36/jason/', 'kanye')
        pos_data = RawPosData(anim, '20160426')
        pos_stream = RawPosDataStream(pos_data, 1, 'online')

        pos_itr = pos_stream()

        count = 0
        for pos in pos_itr:
            count += 1

        self.assertEqual(count, 48017)


class TestLFPDataStream(TestCase):

    def test_iter(self):
        anim = FrankAnimalInfo('/opt/data36/jason/', 'kanye')
        lfp_data = LFPData(anim, '20160426')
        lfp_stream = LFPDataStream(lfp_data, 1, list(range(1, 17)))

        lfp_itr = lfp_stream()

        count = 0
        for lfp in lfp_itr:
            count += 1

        print(count)

        self.assertEqual(count, 40176640)


class TestSpikeDataStream(TestCase):

    def test_iter(self):
        anim = FrankAnimalInfo('/opt/data36/jason/', 'kanye')
        spkamp_data = SpikeAmpData(anim, '20160426')
        spkamp_stream = SpikeDataStream(spkamp_data, 1, range(1, 17))

        spkamp_itr = spkamp_stream()

        count = 0
        for spk in spkamp_itr:
            count += 1

        self.assertEqual(count, 1707000)


class TestSpikeDataStreamSlow(TestCase):

    def test_iter(self):
        anim = FrankAnimalInfo('/opt/data36/jason/', 'kanye')
        spkamp_data = SpikeAmpData(anim, '20160426')
        spkamp_stream = SpikeDataStreamSlow(spkamp_data, 1, range(1, 17))

        spkamp_itr = spkamp_stream()

        count = 0
        for spk in spkamp_itr:
            count += 1

        self.assertEqual(count, 1707000)

