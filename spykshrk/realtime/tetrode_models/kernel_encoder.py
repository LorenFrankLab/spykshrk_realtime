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
        self.firing_rate = np.ones_like(self.pos_hist)

        pos_bin_center_tmp = self.param.pos_hist_struct.pos_bin_center
        #currently not using pos_kernel because i turned off the convolution step below
        self.pos_kernel = gaussian(pos_bin_center_tmp,
                                   pos_bin_center_tmp[int(len(pos_bin_center_tmp)/2)],
                                   self.param.pos_kernel_std)

        self.occupancy_counter = 1
        self.display_occupancy = True
        self.taskState = 1
        self.task3_counter = 0
        self.load_encoding = self.config['ripple_conditioning']['load_encoding']
        self.spike_counter = 0

        # define arm_coords for occupancy
        self.number_arms = self.config['pp_decoder']['number_arms']
        
        # define arm coords directly from config
        self.arm_coords = np.array(self.config['encoder']['arm_coords'])

        self.max_pos = self.arm_coords[-1][-1] + 1
        self.pos_bins = np.arange(0,self.max_pos,1)

        self.bin_idx = -1

        #print('num bins: ',param.pos_hist_struct.num_bins)
        #print('range: ',param.pos_hist_struct.pos_range)
        #print('bin edges: ',param.pos_hist_struct.pos_bin_edges)
        #print('bin center: ',param.pos_hist_struct.pos_bin_center)
        #print('bin delta: ',param.pos_hist_struct.pos_bin_delta)

        ############################################################################
        # for KDE computation
        N = 10000
        sigma = self.config['encoder']['mark_kernel']['std']
        self.marks = np.zeros((N, 4), dtype=np.float32)
        self.positions = np.zeros(N, dtype=np.float32)
        self.mark_idx = 0
        self.k1 = 1 / (np.sqrt(2*np.pi) * sigma)
        self.k2 = -0.5 / (sigma**2)
        ############################################################################

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

    def update_covariate(self, covariate, current_vel=None, taskState=None):
        self.covariate = covariate
        #print('position in update position: ',self.covariate)
        self.current_vel = current_vel
        self.taskState = taskState
        # bin_idx = np.nonzero((self.param.pos_hist_struct.pos_bin_edges - covariate) > 0)[0][0] - 1
        self.bin_idx = self.param.pos_hist_struct.which_bin(self.covariate)
        #only want to add to pos_hist during movement times - aka vel > 8
        if (abs(self.current_vel) >= self.config['encoder']['vel'] and self.taskState == 1
            and not self.load_encoding):
            self.pos_hist[self.bin_idx] += 1
            #print('occupancy before',self.pos_hist)
            #print('update_covariate current_vel: ',self.current_vel)
            # put NaNs into arm gaps
            self.apply_no_anim_boundary(self.pos_bins, self.arm_coords, self.pos_hist, np.nan)
            #print('occupancy',self.pos_hist)

            self.occupancy_counter += 1

        if self.occupancy_counter % 10000 == 0:
            #print('encoder_query_occupancy: ',self.pos_hist)
            print('number of position entries encoder: ',self.occupancy_counter)
        
    # remember to change this eventually to use encoding spikes, not decoding
    # only for testing at the moment to minimize changes
    def add_spike_to_decode(self):
        self.firing_rate[self.bin_idx] += 1

    def new_mark(self, mark, new_cov=None):
        ## update new covariate if specified, otherwise use previous covariate state
        ## it doesnt look this is currently being used
        #if new_cov:
        #    self.update_covariate(new_cov)

        #self.tree.insert_rec(mark[0], mark[1], mark[2],
        #                     mark[3], self.covariate)
        #print('position in new mark: ',self.covariate)

        ############################################################################
        if self.mark_idx == self.marks.shape[0]:
            self.marks = np.vstack((self.marks, np.zeros_like(self.marks)))
            self.positions = np.hstack((self.positions, np.zeros_like(self.positions)))

        self.marks[self.mark_idx] = mark
        self.positions[self.mark_idx] = self.covariate
        self.mark_idx += 1
        ############################################################################

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

        # rstar tree version
        #query_weights, query_positions = self.query_mark(mark)

        # load marks and position if needed
        if self.load_encoding and self.spike_counter == 0:
            self.marks = np.load(f'/tmp/marks{elec_grp_id}.npy')
            self.positions = np.load(f'/tmp/position{elec_grp_id}.npy')
            self.pos_hist = np.load(f'/tmp/occupancy{elec_grp_id}.npy')
            self.mark_idx = np.load(f'/tmp/marks_idx{elec_grp_id}.npy')[0]
            print('loaded marks and position for tet',elec_grp_id)
            #print(self.pos_hist)
            #print(self.mark_idx)
            self.spike_counter += 1

        # if start of taskstate3 save marks and position
        if self.taskState == 3 and self.task3_counter == 0 and not self.load_encoding:
            np.save(f'/tmp/marks{elec_grp_id}.npy',self.marks)
            np.save(f'/tmp/position{elec_grp_id}.npy',self.positions)
            np.save(f'/tmp/occupancy{elec_grp_id}.npy',self.pos_hist)
            np.save(f'/tmp/marks_idx{elec_grp_id}.npy',np.array((self.mark_idx,0)))
            #print('marks size',self.mark_idx,self.marks.shape[0])
            print('saved marks and position for tet',elec_grp_id)
            self.task3_counter += 1

        compute_histogram = True
        # apply filter if requested
        if self.config['encoder']['mark_kernel']['enable_filter'] == 1:
            std = self.config['encoder']['mark_kernel']['std']
            n_std = self.config['encoder']['mark_kernel']['n_std']
            in_range = np.ones(self.mark_idx, dtype=bool)
            for ii in range(self.marks.shape[1]):
                in_range = np.logical_and(
                    np.logical_and(
                        self.marks[:self.mark_idx, ii] > mark[ii] - n_std * std,
                        self.marks[:self.mark_idx, ii] < mark[ii] + n_std * std),
                    in_range)
            if np.sum(in_range) < self.config['encoder']['mark_kernel']['n_marks_min']:
                compute_histogram = False

        if compute_histogram:

            ############################################################################
            # evaluate Gaussian kernel on distance in mark space
            squared_distance = np.sum(
                np.square(self.marks[:self.mark_idx] - mark),
                axis=1)
            query_weights = self.k1 * np.exp(squared_distance * self.k2)
            query_positions = self.positions[:self.mark_idx]
            ############################################################################

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
            if self.taskState == 2 and self.display_occupancy:
                print(self.pos_hist)
                print((self.pos_hist/np.nansum(self.pos_hist)))
                self.display_occupancy = False

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
        else:
            return None

