from unittest import TestCase
from spykshrk.realtime.timing_system import TimingMessage, TimingFileWriter, TimingFileReader

class TestTimingMessage(TestCase):

    def setUp(self):
        self.msg = TimingMessage(label='Test', timestamp=100, start_rank=1)
        self.msg.record_time(rank=2)
        self.msg.record_time(rank=3)

    def test_TimingMessage_serialization(self):

        msg_bytes = self.msg.pack()
        msg_unpack = TimingMessage.unpack(message_bytes=msg_bytes)

        self.assertEqual(self.msg.label, msg_unpack.label,
                         'Serialization failed, labels do not match ({}, {})'.
                         format(self.msg.label, msg_unpack.label))

        self.assertEqual(self.msg.timestamp, msg_unpack.timestamp,
                         'Serialization failed, timestamps do not match ({}, {})'.
                         format(self.msg.timestamp, msg_unpack.timestamp))

        self.assertEqual(self.msg.timing_data, msg_unpack.timing_data,
                         'Serialization failed, timing_data does not match ({}, {})'.
                         format(self.msg.timing_data, msg_unpack.timing_data))

    def test_TimingMessage_eq(self):
        msg_bytes = self.msg.pack()
        msg_unpack = TimingMessage.unpack(message_bytes=msg_bytes)

        new_msg = TimingMessage(label='Test', timestamp=100, start_rank=1)
        new_msg.record_time(rank=2)
        new_msg.record_time(rank=3)

        self.assertIsNot(self.msg, msg_unpack, 'The msg and the deserialized form '
                                               'should not be the same object: {}, {}.'.format(self.msg, msg_unpack))
        self.assertEqual(self.msg, msg_unpack, 'The msg and the deserialized form '
                                               'should be equal: {}, {}.'.format(self.msg, msg_unpack))
        self.assertNotEqual(self.msg, new_msg, 'Two messages created at different times '
                                               'should be different: {}, {}'.format(self.msg, new_msg))


class TestTimingFileWriter(TestCase):

    def setUp(self):
        self.msgs_to_save = []
        msg = TimingMessage(label='Test', timestamp=100, start_rank=1)
        msg.record_time(rank=2)
        msg.record_time(rank=3)
        self.msgs_to_save.append(msg)
        self.msgs_to_save.append(msg)
        msg2 = TimingMessage(label='Test2', timestamp=101, start_rank=1)
        self.msgs_to_save.append(msg2)

        self.save_dir = '/tmp'
        self.prefix = 'test'
        self.mpi_rank = 0
        self.postfix = 'timing'

    def test_TimingFileWriter(self):
        self.writer = TimingFileWriter(save_dir=self.save_dir,
                                       file_prefix=self.prefix,
                                       mpi_rank=self.mpi_rank,
                                       file_postfix=self.postfix)

        for msg in self.msgs_to_save:
            self.writer.write_timing_message(msg)
        self.writer.close()

        self.reader = TimingFileReader(save_dir=self.save_dir,
                                       file_prefix=self.prefix,
                                       mpi_rank=self.mpi_rank,
                                       file_postfix=self.postfix)
        last_rec = -1
        for rec_id, read_msg in self.reader:
            self.assertEqual(last_rec, rec_id - 1, 'The rec id did not start at 0 and increment 1 per message: '
                                                   'last {}, new {}.'.format(last_rec, rec_id))

            self.assertEqual(read_msg, self.msgs_to_save[rec_id], 'Rec in file is different from rec saved: '
                             'from file {}, from memory {}'.format(read_msg, self.msgs_to_save[rec_id]))
            # print(rec_id)
            # print(read_msg)
            # print(self.msgs_to_save[rec_id])

            last_rec = rec_id

