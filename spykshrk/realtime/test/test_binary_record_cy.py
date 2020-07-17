from unittest import TestCase

import pickle

from spykshrk.realtime.binary_record_cy import BinaryRecordsFileReader
from spykshrk.realtime.binary_record import  BinaryRecordsFileWriter, BinaryRecordCreateMessage


class TestBinaryRecordsFileReader(TestCase):

    def setUp(self):
        self.create_message = BinaryRecordCreateMessage(manager_label='test',
                                                        file_id=-1,
                                                        save_dir='/tmp',
                                                        file_prefix='test',
                                                        file_postfix='bin_rec',
                                                        rec_label_dict={1:['test']},
                                                        rec_format_dict={1:'i'})

        self.writer = BinaryRecordsFileWriter(self.create_message, mpi_rank=0)
        self.writer.write_rec(1, 1)
        self.writer.write_rec(1, 2)
        self.writer.write_rec(1, 3)
        self.writer.close()

    def test_cython_BinaryRecordReader(self):
        reader = BinaryRecordsFileReader(save_dir='/tmp',
                                         file_prefix='test',
                                         mpi_rank=0,
                                         manager_label='test',
                                         file_postfix='bin_rec',
                                         filemeta_as_col=False)
        reader.start_record_reading()
        pd = reader.convert_pandas()

        # check if data is correct for rec_id 1's column 'test'
        pd_integrity = (pd[1]['test'] == [1, 2, 3])
        self.assertTrue(pd_integrity.all())

    def test_cython_BinaryRecordReader_pickle(self):

        reader = BinaryRecordsFileReader(save_dir='/tmp',
                                         file_prefix='test',
                                         mpi_rank=0,
                                         manager_label='test',
                                         file_postfix='bin_rec',
                                         filemeta_as_col=False)

        pickle.dumps(reader)
