import spykshrk.realtime.rst.RSTPython as RST
import struct
from spykshrk.realtime.realtime_logging import PrintableMessage
import numpy as np


def gaussian(x, mu, sig):
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))


class PosBinStruct:
    def __init__(self, pos_range, num_bins):
        self.pos_range = pos_range
        self.num_bins = num_bins
        self.pos_bin_edges = np.linspace(pos_range[0], pos_range[1], num_bins + 1, endpoint=True, retstep=False)
        self.pos_bin_center = (self.pos_bin_edges[:-1] + self.pos_bin_edges[1:]) / 2
        self.pos_bin_delta = self.pos_bin_center[1] - self.pos_bin_center[0]

    def which_bin(self, pos):
        return np.nonzero(np.diff(self.pos_bin_edges > pos))


class RSTParameter:
    def __init__(self, kernel, pos_hist_struct, pos_kernel_std):
        self.kernel = kernel
        self.pos_hist_struct = pos_hist_struct
        self.pos_kernel_std = pos_kernel_std
        #print('pos_hist_struct: ',self.pos_hist_struct)


class RSTKernelEncoderQuery(PrintableMessage):
    _header_byte_fmt = '=qiii'
    _header_byte_len = struct.calcsize(_header_byte_fmt)

    def __init__(self, query_time, elec_grp_id, query_weights, query_positions, query_hist):
        self.query_time = query_time
        self.elec_grp_id = elec_grp_id
        self.query_weights = query_weights
        self.query_positions = query_positions
        self.query_hist = query_hist

    def pack(self):
        query_len = len(self.query_weights)
        query_byte_len = query_len * struct.calcsize('=f')
        query_hist_len = len(self.query_hist)
        query_hist_byte_len = query_hist_len * struct.calcsize('=d')

        message_bytes = struct.pack(self._header_byte_fmt,
                                    self.query_time,
                                    self.elec_grp_id,
                                    query_byte_len,
                                    query_hist_byte_len)

        message_bytes = message_bytes + self.query_weights.tobytes() + \
                        self.query_positions.tobytes() + self.query_hist.tobytes()

        return message_bytes

    @classmethod
    def unpack(cls, message_bytes):
        query_time, elec_grp_id, query_len, query_hist_len = struct.unpack(cls._header_byte_fmt,
                                                                         message_bytes[0:cls._header_byte_len])

        query_weights = np.frombuffer(message_bytes[cls._header_byte_len: cls._header_byte_len+query_len],
                                      dtype='float32')

        query_positions = np.frombuffer(message_bytes[cls._header_byte_len+query_len:
                                                      cls._header_byte_len+2*query_len],
                                       dtype='float32')

        query_hist = np.frombuffer(message_bytes[cls._header_byte_len+2*query_len:
                                                 cls._header_byte_len+2*query_len+query_hist_len])

        return cls(query_time=query_time, elec_grp_id=elec_grp_id, query_weights=query_weights,
                   query_positions=query_positions, query_hist=query_hist)


class RSTKernelEncoder:
    def __init__(self, filename, new_tree, param, config):
        self.param = param
        self.kernel = param.kernel
        self.filename = filename
        self.new_tree = new_tree
        self.config = config

        self.tree = RST.RSTPython(filename.encode('utf-8'), new_tree, param.kernel)
        self.covariate = 0
        # initialize to one's to prevent divide by zero when normalizing by occupancy
        self.pos_hist = np.ones(param.pos_hist_struct.num_bins)

        pos_bin_center_tmp = self.param.pos_hist_struct.pos_bin_center
        #currently not using pos_kernel because i turned off the convolution step below
        self.pos_kernel = gaussian(pos_bin_center_tmp,
                                   pos_bin_center_tmp[int(len(pos_bin_center_tmp)/2)],
                                   self.param.pos_kernel_std)

        self.occupancy_counter = 0

        # define arm_coords for occupancy
        self.arm_coords = np.array([[0,8],[13,24],[29,40],[45,56],[61,72],[77,88],[93,104],[109,120],[125,136]])
        self.max_pos = self.arm_coords[-1][-1] + 1
        self.pos_bins = np.arange(0,self.max_pos,1)

        #print('num bins: ',param.pos_hist_struct.num_bins)
        #print('range: ',param.pos_hist_struct.pos_range)
        #print('bin edges: ',param.pos_hist_struct.pos_bin_edges)
        #print('bin center: ',param.pos_hist_struct.pos_bin_center)
        #print('bin delta: ',param.pos_hist_struct.pos_bin_delta)

    def apply_no_anim_boundary(self, x_bins, arm_coor, image, fill=0):
        # from util.py script in offline decoder folder

        # calculate no-animal boundary
        arm_coor = np.array(arm_coor, dtype='float64')
        arm_coor[:,0] -= x_bins[1] - x_bins[0]
        bounds = np.vstack([[x_bins[-1], 0], arm_coor])
        bounds = np.roll(bounds, -1)

        boundary_ind = np.searchsorted(x_bins, bounds, side='right')
        #boundary_ind[:,1] -= 1

        for bounds in boundary_ind:
            if image.ndim == 1:
                image[bounds[0]:bounds[1]] = fill
            elif image.ndim == 2:
                image[bounds[0]:bounds[1], :] = fill
                image[:, bounds[0]:bounds[1]] = fill
        return image

    def update_covariate(self, covariate, current_vel=None):
        self.covariate = covariate
        #print('position in update position: ',self.covariate)
        self.current_vel = current_vel
        # bin_idx = np.nonzero((self.param.pos_hist_struct.pos_bin_edges - covariate) > 0)[0][0] - 1
        bin_idx = self.param.pos_hist_struct.which_bin(self.covariate)
        #only want to add to pos_hist during movement times - aka vel > 8
        if abs(self.current_vel) >= self.config['encoder']['vel']:
            self.pos_hist[bin_idx] += 1
            #print('occupancy before',self.pos_hist)
            #print('update_covariate current_vel: ',self.current_vel)
            # put NaNs into arm gaps
            self.apply_no_anim_boundary(self.pos_bins, self.arm_coords, self.pos_hist, np.nan)
            #print('occupancy',self.pos_hist)

            self.occupancy_counter += 1
        if self.occupancy_counter % 10000 == 0:
            #print('encoder_query_occupancy: ',self.pos_hist)
            print('number of position entries encoder: ',self.occupancy_counter)      

    def new_mark(self, mark, new_cov=None):
        # update new covariate if specified, otherwise use previous covariate state
        # it doesnt look this is currently being used
        if new_cov:
            self.update_covariate(new_cov)

        self.tree.insert_rec(mark[0], mark[1], mark[2],
                             mark[3], self.covariate)
        #print('position in new mark: ',self.covariate)

    # MEC 7-10-19 try going from 5 to 3, because 3 stdev in 4D space will still get 95% of the points
    def query_mark(self, mark):
        x1 = mark[0]
        x2 = mark[1]
        x3 = mark[2]
        x4 = mark[3]
        x1_l = x1 - self.kernel.stddev * self.config['encoder']['RStar_edge_length_factor']
        x2_l = x2 - self.kernel.stddev * self.config['encoder']['RStar_edge_length_factor']
        x3_l = x3 - self.kernel.stddev * self.config['encoder']['RStar_edge_length_factor']
        x4_l = x4 - self.kernel.stddev * self.config['encoder']['RStar_edge_length_factor']
        x1_h = x1 + self.kernel.stddev * self.config['encoder']['RStar_edge_length_factor']
        x2_h = x2 + self.kernel.stddev * self.config['encoder']['RStar_edge_length_factor']
        x3_h = x3 + self.kernel.stddev * self.config['encoder']['RStar_edge_length_factor']
        x4_h = x4 + self.kernel.stddev * self.config['encoder']['RStar_edge_length_factor']
        query_weights, query_positions = self.tree.query_rec(x1_l, x2_l, x3_l, x4_l,
                                                             x1_h, x2_h, x3_h, x4_h,
                                                             x1, x2, x3, x4)
        return query_weights, query_positions

    def query_mark_hist(self, mark, time, elec_grp_id):
        # to turn off RStar Tree query uncomment next 2 lines and comment out next line after
        #query_weights = np.zeros((1,137))+0.1
        #query_positions = np.zeros((1,137))+0.5

        query_weights, query_positions = self.query_mark(mark)
        query_hist, query_hist_edges = np.histogram(
            a=query_positions, bins=self.param.pos_hist_struct.pos_bin_edges,
            weights=query_weights, normed=False)
        # print observations before offset
        #print('weights',query_weights)
        #print('position',query_positions)
        #print('observations',query_hist)

        # Offset from zero - this could be a problem for the gaps between arms
        # gaps will have high firing rate because of this offset
        # we may want to remove this, and/or we will put NaNs in the gaps for self.pos_hist
        query_hist += 0.0000001

        # occupancy normalize
        # MEC: added NaNs in the gaps between arms in self.pos_hist
        # MEC: normalize self.pos_hist to match offline decoder 
        query_hist = query_hist / (self.pos_hist/np.nansum(self.pos_hist))
        query_hist[np.isnan(query_hist)]=0.0
        #print(query_weights.shape)
        #print('obs after normalize',query_hist)

        # MEC - turned off convolution because we are using 5cm position bins
        #query_hist = np.convolve(query_hist, self.pos_kernel, mode='same')
        #print(query_hist)

        # normalized PDF
        # MEC: replace sum with nansum - this seems okay now
        # note: pos_bin_delta is currently 1
        #print('query hist sum',np.nansum(query_hist))
        query_hist = query_hist / (np.sum(query_hist) * self.param.pos_hist_struct.pos_bin_delta)
        #print('observation:',query_hist)
        #print('observ sum',np.nansum(query_hist))

        return RSTKernelEncoderQuery(query_time=time,
                                     elec_grp_id=elec_grp_id,
                                     query_weights=query_weights,
                                     query_positions=query_positions,
                                     query_hist=query_hist)

