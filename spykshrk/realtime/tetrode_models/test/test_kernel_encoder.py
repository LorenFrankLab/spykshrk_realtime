from unittest import TestCase
from spykshrk.realtime.tetrode_models.kernel_encoder import RSTKernelEncoder, RSTParameter, PosBinStruct
import sys
import spykshrk.realtime.rst.RSTPython as RST
import numpy as np


class TestRSTKernelEncoder(TestCase):

    def setUp(self):
        kernel = RST.kernel_param(0, 2, -1024, 1024, 1)
        pos_bin_struct = PosBinStruct([0, 10], 20)
        self.rst_param = RSTParameter(kernel, pos_bin_struct)

    def test_basic(self):
        rst_encoder = RSTKernelEncoder('/tmp/test', True, self.rst_param)
        rst_encoder.update_covariate(10)
        rst_encoder.new_mark([1, 1, 1, 1])
        rst_encoder.update_covariate(5)
        rst_encoder.new_mark([3, 3, 3, 3])
        results = rst_encoder.query_mark([1, 1, 1, 1])

        self.assertAlmostEqual(results[0][0], 0.2, 2, 'Closer point ({:}, {:}) should be almost 0.2.'.
                               format(results[0][0], results[1][0]))
        self.assertAlmostEqual(results[0][1], 0.025, 2, 'Further point ({:} {:}) should be almost 0.025.'.
                               format(results[0][1], results[1][1]))

    def test_query_mark_hist(self):
        rst_encoder = RSTKernelEncoder('/tmp/test', True, self.rst_param)
        rst_encoder.update_covariate(10)
        rst_encoder.new_mark([1, 1, 1, 1])
        rst_encoder.update_covariate(5)
        rst_encoder.new_mark([3, 3, 3, 3])
        results = rst_encoder.query_mark_hist([1, 1, 1, 1], 100, 1)
        print(results.query_hist)
