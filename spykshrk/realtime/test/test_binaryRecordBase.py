from unittest import TestCase

from spykshrk.realtime.binary_record import RemoteBinaryRecordsManager, BinaryRecordsManager, BinaryRecordsError, \
    BinaryRecordsFileReader
from spykshrk.realtime.realtime_base import BinaryRecordBase, BinaryRecordBaseWithTiming

class TestBinaryRecordBase(TestCase):

    def setUp(self):
        self.rec_manager = BinaryRecordsManager(manager_label='test',
                                                save_dir='/tmp',
                                                file_prefix='test',
                                                file_postfix='dat')
        self.local_rec_manager = RemoteBinaryRecordsManager(manager_label='test', local_rank=0, manager_rank=0)


    def test_simple(self):
        rec_base = BinaryRecordBase(rank=0, local_rec_manager=self.local_rec_manager,
                                    rec_ids=[1], rec_labels=[['int', 'int']], rec_formats=['dd'])
        rec_messages = rec_base.get_record_register_messages()
        for msg in rec_messages:
            self.rec_manager.register_rec_type_message(msg)

        rec_base.set_record_writer_from_message(self.rec_manager.new_writer_message())

        rec_base.write_record(1, 1, 1)
        rec_base.start_record_writing()
        rec_base.write_record(1, 2, 2)
        rec_base.stop_record_writing()
        rec_base.write_record(1, 3, 3)

        rec_base.close_record()

        reader = BinaryRecordsFileReader(save_dir='/tmp',
                                         file_prefix='test',
                                         mpi_rank=0,
                                         manager_label='test',
                                         file_postfix='dat')

        # Check and see if read file contains the correct single record
        self.assertEqual(reader.__next__(), (0, 1, (2.0, 2.0)))
        # Check to make sure iterator stops at end of file
        with self.assertRaises(StopIteration):
            reader.__next__()

    def test_no_records(self):
        rec_base = BinaryRecordBase(rank=0, local_rec_manager=self.local_rec_manager,
                                    rec_ids=[], rec_labels=[], rec_formats=[])
        rec_messages = rec_base.get_record_register_messages()
        for msg in rec_messages:
            self.rec_manager.register_rec_type_message(msg)

        rec_base.set_record_writer_from_message(self.rec_manager.new_writer_message())

        # Should error if writing record that has never been registered
        with self.assertRaises(BinaryRecordsError):
            rec_base.write_record(1, 1, 1)

        # Should have no effect
        rec_base.start_record_writing()
        rec_base.stop_record_writing()

        # closes the empty file
        rec_base.close_record()


class TestBinaryRecordBaseWithTiming(TestCase):

    def setUp(self):
        self.rec_manager = BinaryRecordsManager(manager_label='test',
                                                save_dir='/tmp',
                                                file_prefix='test',
                                                file_postfix='dat')
        self.local_rec_manager = RemoteBinaryRecordsManager(manager_label='test', local_rank=0, manager_rank=0)


    def test_simple(self):
        rec_base = BinaryRecordBaseWithTiming(rank=0, local_rec_manager=self.local_rec_manager,
                                              rec_ids=[1], rec_labels=[['int', 'int']], rec_formats=['dd'])
        rec_messages = rec_base.get_record_register_messages()
        for msg in rec_messages:
            self.rec_manager.register_rec_type_message(msg)

        rec_base.set_record_writer_from_message(self.rec_manager.new_writer_message())

        rec_base.write_record(1, 1, 1)
        rec_base.start_record_writing()
        rec_base.write_record(1, 2, 2)
        rec_base.record_timing(100, 10, 1, 'label')
        rec_base.stop_record_writing()
        rec_base.write_record(1, 3, 3)

        rec_base.close_record()

        reader = BinaryRecordsFileReader(save_dir='/tmp',
                                         file_prefix='test',
                                         mpi_rank=0,
                                         manager_label='test',
                                         file_postfix='dat')

        # Check and see if read file contains the correct single record
        self.assertEqual(reader.__next__(), (0, 1, (2.0, 2.0)))
        print(reader.__next__())
        # Check to make sure iterator stops at end of file
        #with self.assertRaises(StopIteration):
            #reader.__next__()
