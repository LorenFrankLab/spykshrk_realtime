import numpy as np
import sys
from time import time


class SimDataBuffer:
    """ This class implements the interleaving of multiple data streams
    based on each data stream sample's timestamps.

    Data streams are currently iterators that have a preset
    data return format.

    Each sample is then packaged in a unique message that can be iteratively
    streams in chronological order by setting up a generator.
    """

    def __init__(self, iter_list):
        self.iter_list = iter_list

    def __call__(self):
        data_point_list = [None] * len(self.iter_list)
        timestamp_list = [sys.maxsize] * len(self.iter_list)

        cur_iter_list = self.iter_list

        for ind, iter_item in enumerate(cur_iter_list):
            try:
                data_point_list[ind] = iter_item.__next__()
                timestamp_list[ind] = data_point_list[ind].timestamp
            except StopIteration:
                print('Iterator #{} finished.'.format(ind))
                del cur_iter_list[ind]
                del data_point_list[ind]
                del timestamp_list[ind]

        while cur_iter_list:
            try:
                min_ind = timestamp_list.index(min(timestamp_list))
                yield data_point_list[min_ind]

                data_point_list[min_ind] = cur_iter_list[min_ind].__next__()
                timestamp_list[min_ind] = data_point_list[min_ind].timestamp

            except StopIteration:
                print('Iterator #{} finished.'.format(min_ind))
                del cur_iter_list[min_ind]
                del data_point_list[min_ind]
                del timestamp_list[min_ind]


