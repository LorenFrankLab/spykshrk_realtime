import os
import fcntl
import math
import struct

import numpy as np
from scipy.ndimage.interpolation import shift

import spykshrk.realtime.realtime_logging as rt_logging
from mpi4py import MPI
#from spykshrk.franklab.pp_decoder.util import apply_no_anim_boundary
from spykshrk.realtime import (binary_record, datatypes, encoder_process,
                               main_process, realtime_base)
#from spykshrk.realtime import realtime_logging as rt_logging
from spykshrk.realtime import ripple_process
from spykshrk.realtime.camera_process import (LinearPositionAssignment,
                                              VelocityCalculator)
from spykshrk.realtime.simulator import simulator_process


class PosteriorSum(rt_logging.PrintableMessage):
    """"Message containing summed posterior from decoder_process.

    This message has helper serializer/deserializer functions to be used to speed transmission.
    """
    _byte_format = 'IIdddddddddiii'

    def __init__(self, bin_timestamp, spike_timestamp, box, arm1, arm2, arm3, arm4, arm5, arm6, arm7, arm8, spike_count, crit_ind, posterior_max):
        self.bin_timestamp = bin_timestamp
        self.spike_timestamp = spike_timestamp
        self.box = box
        self.arm1 = arm1
        self.arm2 = arm2
        self.arm3 = arm3
        self.arm4 = arm4
        self.arm5 = arm5
        self.arm6 = arm6
        self.arm7 = arm7
        self.arm8 = arm8
        self.spike_count = spike_count
        self.crit_ind = crit_ind
        self.posterior_max = posterior_max

    def pack(self):
        return struct.pack(self._byte_format, self.bin_timestamp, self.spike_timestamp, self.box,
                           self.arm1, self.arm2, self.arm3, self.arm4, self.arm5, self.arm6, self.arm7,
                           self.arm8, self.spike_count, self.crit_ind, self.posterior_max)

    @classmethod
    def unpack(cls, message_bytes):
        bin_timestamp, spike_timestamp, box, arm1, arm2, arm3, arm4, arm5, arm6, arm7, arm8, spike_count, crit_ind, posterior_max = struct.unpack(
            cls._byte_format, message_bytes)
        return cls(bin_timestamp=bin_timestamp, spike_timestamp=spike_timestamp, box=box, arm1=arm1, arm2=arm2,
                   arm3=arm3, arm4=arm4, arm5=arm5, arm6=arm6, arm7=arm7, arm8=arm8, spike_count=spike_count,
                   crit_ind=crit_ind, posterior_max=posterior_max)


class VelocityPosition(rt_logging.PrintableMessage):
    """"Message containing velocity and linearized position from decoder_process.

    This message has helper serializer/deserializer functions to be used to speed transmission.
    """
    _byte_format = 'Iid'

    def __init__(self, bin_timestamp, pos, vel):
        self.bin_timestamp = bin_timestamp
        self.pos = pos
        self.vel = vel

    def pack(self):
        return struct.pack(self._byte_format, self.bin_timestamp, self.pos, self.vel)

    @classmethod
    def unpack(cls, message_bytes):
        bin_timestamp, pos, vel = struct.unpack(
            cls._byte_format, message_bytes)
        return cls(bin_timestamp=bin_timestamp, pos=pos, vel=vel)


class DecoderMPISendInterface(realtime_base.RealtimeMPIClass):
    def __init__(self, comm: MPI.Comm, rank, config):
        super(DecoderMPISendInterface, self).__init__(
            comm=comm, rank=rank, config=config)

    def send_record_register_messages(self, record_register_messages):
        self.class_log.debug("Sending record register messages.")
        for message in record_register_messages:
            self.comm.send(obj=message, dest=self.config['rank']['supervisor'],
                           tag=realtime_base.MPIMessageTag.COMMAND_MESSAGE.value)

    # def sending posterior message to supervisor with POSTERIOR tag
    def send_posterior_message(self, bin_timestamp, spike_timestamp, box, arm1, arm2, arm3, arm4, arm5, arm6, arm7, arm8, spike_count, crit_ind, posterior_max):
        message = PosteriorSum(bin_timestamp, spike_timestamp, box,
                               arm1, arm2, arm3, arm4, arm5, arm6, arm7, arm8, spike_count,
                               crit_ind, posterior_max)
        #print('stim_message: ',message)

        self.comm.Send(buf=message.pack(),
                       dest=self.config['rank']['supervisor'],
                       tag=realtime_base.MPIMessageTag.POSTERIOR.value)
        #print('stim_message: ',message,self.config['rank']['decoder'],self.rank)

    # def sending velocity&position message to supervisor with VEL_POS tag
    def send_vel_pos_message(self, bin_timestamp, pos, vel):
        message = VelocityPosition(bin_timestamp, pos, vel)
        #print('vel_message: ',message)

        self.comm.Send(buf=message.pack(),
                       dest=self.config['rank']['supervisor'],
                       tag=realtime_base.MPIMessageTag.VEL_POS.value)
        #print('vel_message: ',message,self.config['rank']['decoder'],self.rank)

    def send_time_sync_report(self, time):
        self.comm.send(obj=realtime_base.TimeSyncReport(time),
                       dest=self.config['rank']['supervisor'],
                       tag=realtime_base.MPIMessageTag.COMMAND_MESSAGE)

    def all_barrier(self):
        self.comm.Barrier()


class SpikeDecodeRecvInterface(realtime_base.RealtimeMPIClass):
    def __init__(self, comm: MPI.Comm, rank, config):
        super(SpikeDecodeRecvInterface, self).__init__(
            comm=comm, rank=rank, config=config)

        self.msg_buffer = bytearray(50000)
        self.req = self.comm.Irecv(
            buf=self.msg_buffer, tag=realtime_base.MPIMessageTag.SPIKE_DECODE_DATA)

    def __next__(self):
        rdy = self.req.Test()
        if rdy:

            msg = encoder_process.SpikeDecodeResultsMessage.unpack(
                self.msg_buffer)
            self.req = self.comm.Irecv(
                buf=self.msg_buffer, tag=realtime_base.MPIMessageTag.SPIKE_DECODE_DATA)
            #print('decoded spike message',msg.pos_hist)
            return msg

        else:
            return None

# make receiver to take in threshold message from ripple node - use same setup as in main_process
# to use this LFP timekeeper compare the timestamp of the lfp message to the timestamp of the last spike
# if greter than 5 msec then trigger calcuating the posterior


class LFPTimekeeperRecvInterface(realtime_base.RealtimeMPIClass):
    def __init__(self, comm: MPI.Comm, rank, config):
        super(LFPTimekeeperRecvInterface, self).__init__(
            comm=comm, rank=rank, config=config)

        self.mpi_status = MPI.Status()

        self.feedback_bytes = bytearray(16)
        self.timing_bytes = bytearray(100)

        self.mpi_reqs = []
        self.mpi_statuses = []

        req_feedback = self.comm.Irecv(buf=self.feedback_bytes,
                                       tag=realtime_base.MPIMessageTag.FEEDBACK_DATA.value)
        self.mpi_statuses.append(MPI.Status())
        self.mpi_reqs.append(req_feedback)

    # def __iter__(self):
    #    return self

    def __next__(self):
        rdy = MPI.Request.Testall(
            requests=self.mpi_reqs, statuses=self.mpi_statuses)

        if rdy:
            #if self.mpi_statuses[0].source in self.config['rank']['ripples']:
            # NOTE: this will only work if rank 2 is a ripple process
            if self.mpi_statuses[0].source in [2]:    
                message = ripple_process.RippleThresholdState.unpack(
                    message_bytes=self.feedback_bytes)
                self.mpi_reqs[0] = self.comm.Irecv(buf=self.feedback_bytes,
                                                   tag=realtime_base.MPIMessageTag.FEEDBACK_DATA.value)
                #print('lfp message in decoder',message)
                return message

            else:
                return None


class PointProcessDecoder(rt_logging.LoggingClass):

    def __init__(self, pos_range, pos_bins, time_bin_size, arm_coor, config, uniform_gain=0.01):
        self.pos_range = pos_range
        self.pos_bins = pos_bins
        self.time_bin_size = time_bin_size
        self.arm_coor = arm_coor
        self.config = config
        #self.uniform_gain = uniform_gain
        self.uniform_gain = self.config['pp_decoder']['trans_mat_uniform_gain']

        # get number outer arms from config
        self.number_arms = self.config['pp_decoder']['number_arms']
        self.ntrode_list = []
        self.ntrode_list_array = []

        self.cur_pos_time = -1
        self.cur_pos = -1
        self.cur_pos_ind = 0
        self.pos_delta = (self.pos_range[1] -
                          self.pos_range[0]) / self.pos_bins

        # Initialize major PP variables
        self.observation = np.ones(self.pos_bins)
        self.observation_next = np.ones(self.pos_bins)
        self.occ = np.ones(self.pos_bins)
        self.likelihood = np.ones(self.pos_bins)
        self.posterior = np.ones(self.pos_bins)
        self.prev_posterior = np.ones(self.pos_bins)
        self.firing_rate = {}
        # not using Dan's transition matrix
        # self.transition_mat = PointProcessDecoder._create_transition_matrix(self.pos_delta,
        #                                             self.pos_bins,self.arm_coor,self.uniform_gain)

        self.current_spike_count = 0
        self.spike_count_next = 0
        self.total_decoded_spike_count = 0
        self.pos_counter = 0
        self.current_vel = 0

        self._ripple_thresh_states = {}

        self.post_sum_bin_length = 20
        self.posterior_sum_time_bin = np.zeros((self.post_sum_bin_length, 9))
        self.posterior_sum_result = np.zeros((1, 9))

        # make arm_coords conditional on number of arms
        if self.number_arms == 8:
            self.arm_coords = np.array([[0, 8], [13, 24], [29, 40], [45, 56], [61, 72], [
                                   77, 88], [93, 104], [109, 120], [125, 136]])
        elif self.number_arms == 4:
            self.arm_coords = np.array([[0,8],[13,24],[29,40],[45,56],[61,72]])
        elif self.number_arms == 2:
            #sun god
            #self.arm_coords = np.array([[0,8],[13,24],[29,40]])
            #tree track
            self.arm_coords = np.array([[0,12],[17,41],[46,70]])

        self.max_pos = self.arm_coords[-1][-1] + 1
        self.pos_bins_1 = np.arange(0, self.max_pos, 1)

        # create sungod transition matrix
        self.transition_mat = PointProcessDecoder._sungod_transition_matrix(self,self.uniform_gain,self.arm_coords,
                                                                            self.max_pos,self.pos_bins_1,
                                                                            self.number_arms)

    @staticmethod
    def _create_transition_matrix(pos_delta, num_bins, arm_coor, uniform_gain=0.01):

        def gaussian(x, mu, sig):
            return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))

            # Setup transition matrix
        x_bins = np.linspace(0, pos_delta * (num_bins - 1), num_bins)

        transition_mat = np.ones([num_bins, num_bins])
        for bin_ii in range(num_bins):
            transition_mat[bin_ii, :] = gaussian(x_bins, x_bins[bin_ii], 3)

        # uniform offset
        uniform_dist = np.ones(transition_mat.shape)

        # apply no-animal boundary

        transition_mat = apply_no_anim_boundary(
            x_bins, arm_coor, transition_mat)
        uniform_dist = apply_no_anim_boundary(x_bins, arm_coor, uniform_dist)

        # normalize transition matrix
        transition_mat = transition_mat / (transition_mat.sum(axis=0)[None, :])
        transition_mat[np.isnan(transition_mat)] = 0

        # normalize uniform offset
        uniform_dist = uniform_dist / (uniform_dist.sum(axis=0)[None, :])
        uniform_dist[np.isnan(uniform_dist)] = 0

        # apply uniform offset
        transition_mat = transition_mat * \
            (1 - uniform_gain) + uniform_dist * uniform_gain

        return transition_mat

    @staticmethod
    def _sungod_transition_matrix(self,uniform_gain,arm_coords,max_pos,pos_bins_1,number_arms):
        # updated for 2 pixels 8-14-19
        # arm_coords updated for 8 pixels 8-15-19
        # NOTE: by rounding up for binning position of outer arms, we get no position in first bin of each arm
        # we could just move the first position here in arm coords and then each arm will start 1 bin higher
        # based on looking at counts from position this should work, so each arm is 11 units

        # make all these steps conditional on config number of arms

        # 8 arm version
        # see if we can use self.arm_coords instead and self.max_pos and self.pos_bins_1

        #arm_coords = np.array([[0, 8], [13, 24], [29, 40], [45, 56], [61, 72], [
        #                      77, 88], [93, 104], [109, 120], [125, 136]])
        # 4 arm version
        #arm_coords = np.array([[0,8],[13,24],[29,40],[45,56],[61,72]])
        #max_pos = arm_coords[-1][-1] + 1
        #pos_bins = np.arange(0, max_pos, 1)

        uniform_gain = uniform_gain
        arm_coords = arm_coords
        max_pos = max_pos
        pos_bins = pos_bins_1
        number_arms = number_arms

        from scipy.sparse import diags
        n = len(pos_bins)
        transition_mat = np.zeros([n, n])
        k = np.array([(1/3) * np.ones(n - 1), (1/3) *
                      np.ones(n), (1 / 3) * np.ones(n - 1)])
        offset = [-1, 0, 1]
        transition_mat = diags(k, offset).toarray()
        box_end_bin = arm_coords[0, 1]

        if number_arms == 8:
            for x in arm_coords[:, 0]:
                transition_mat[int(x), int(x)] = (5/9)
                transition_mat[box_end_bin, int(x)] = (1/9)
                transition_mat[int(x), box_end_bin] = (1/9)

        elif number_arms == 4:
            for x in arm_coords[:,0]:
                transition_mat[int(x),int(x)] = (7/15)
                transition_mat[box_end_bin,int(x)] = (1/5)
                transition_mat[int(x),box_end_bin] = (1/5)

        elif number_arms == 2:
            for x in arm_coords[:,0]:
                transition_mat[int(x),int(x)] = (1/3)
                transition_mat[box_end_bin,int(x)] = (1/3)
                transition_mat[int(x),box_end_bin] = (1/3)


        for y in arm_coords[:, 1]:
            transition_mat[int(y), int(y)] = (2 / 3)

        transition_mat[box_end_bin, 0] = 0
        transition_mat[0, box_end_bin] = 0
        transition_mat[box_end_bin, box_end_bin] = 0
        transition_mat[0, 0] = (2 / 3)

        if number_arms == 8:
            transition_mat[box_end_bin - 1, box_end_bin - 1] = (5/9)
            transition_mat[box_end_bin - 1, box_end_bin] = (1/9)
            transition_mat[box_end_bin, box_end_bin - 1] = (1/9)

        elif number_arms == 4:
            transition_mat[box_end_bin-1, box_end_bin-1] = (7/15)
            transition_mat[box_end_bin-1,box_end_bin] = (1/5)
            transition_mat[box_end_bin, box_end_bin-1] = (1/5)

        elif number_arms == 2:
            transition_mat[box_end_bin-1, box_end_bin-1] = (1/3)
            transition_mat[box_end_bin-1,box_end_bin] = (1/3)
            transition_mat[box_end_bin, box_end_bin-1] = (1/3)

        # uniform offset (gain, currently 0.0001)
        # 9-1-19 this is now taken from config file
        #uniform_gain = 0.0001
        uniform_dist = np.ones(transition_mat.shape) * uniform_gain

        # apply uniform offset
        transition_mat = transition_mat + uniform_dist

        # apply no animal boundary - make gaps between arms
        transition_mat = self.apply_no_anim_boundary(
            pos_bins, arm_coords, transition_mat)

        # to smooth: take the transition matrix to a power
        transition_mat = np.linalg.matrix_power(transition_mat, 1)

        # normalize transition matrix
        transition_mat = transition_mat / (transition_mat.sum(axis=0)[None, :])

        transition_mat[np.isnan(transition_mat)] = 0

        return transition_mat

    def apply_no_anim_boundary(self, x_bins, arm_coor, image, fill=0):
        # from util.py script in offline decoder folder

        # calculate no-animal boundary
        arm_coor = np.array(arm_coor, dtype='float64')
        arm_coor[:, 0] -= x_bins[1] - x_bins[0]
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

    def select_ntrodes(self, ntrode_list):
        self.ntrode_list = ntrode_list
        self.firing_rate = {elec_grp_id: np.ones(self.pos_bins)
                            for elec_grp_id in self.ntrode_list}
        #print('tetrode list',self.ntrode_list,len(self.ntrode_list))
        self.tetrodes_with_spikes = np.zeros(
            (1, len(self.ntrode_list)), dtype=np.bool)
        self.tets_with_spikes_next = np.zeros(
            (1, len(self.ntrode_list)), dtype=np.bool)
        self.ntrode_list_array = np.asarray(self.ntrode_list)

    # MEC: added encoding velocity filter and taskstate
    # firing rate is later used to calculate prob_no_spike

    # if you want to do correct prob_no_spike calculation then you
    # need to keep track of tetrodes here each time you add a spike
    def add_observation(self, spk_elec_grp_id, spk_pos_hist, vel_data, taskState):
        self.taskState = taskState
        if abs(vel_data) >= self.config['encoder']['vel'] and self.taskState == 1:
            #print('firing rate vel thresh',vel_data,'tetrode',spk_elec_grp_id)
            self.firing_rate[spk_elec_grp_id][self.cur_pos_ind] += 1

        self.observation *= spk_pos_hist
        #print('decoded spike',spk_pos_hist)
        # print('observation',self.observation)
        # MEC: i think this should be normalized, not divided by max????
        self.observation = self.observation / np.max(self.observation)
        # print('observation',self.observation)
        self.current_spike_count += 1
        self.total_decoded_spike_count += 1
        # add marker for tet with observation
        #print('tetrode',spk_elec_grp_id,'tetrode list',self.ntrode_list_array)
        self.tetrodes_with_spikes[0][np.where(
            self.ntrode_list_array == spk_elec_grp_id)] = True

    def update_position(self, pos_timestamp, pos_data, vel_data, taskState):
        # Convert position to bin index in histogram count
        # MEC: added NaN mask with no_anim_boundary
        self.cur_pos_time = pos_timestamp
        self.cur_pos = pos_data
        self.cur_vel = vel_data
        self.taskState = taskState
        #print('update position result:',self.cur_pos)
        self.cur_pos_ind = int((self.cur_pos - self.pos_range[0]) /
                               self.pos_delta)
        #print('current position',self.cur_pos)
        #print('pos index added to occupancy',self.cur_pos_ind)

        if abs(self.cur_vel) >= self.config['encoder']['vel'] and self.taskState == 1:
            # MEC test: add all positions to occupancy to compare to offline
            # if abs(self.cur_vel) >= 0:
            self.occ[self.cur_pos_ind] += 1
            self.apply_no_anim_boundary(self.pos_bins_1, self.arm_coords, self.occ, np.nan)

            # originally this was set to 10000
            self.pos_counter += 1
            if self.pos_counter % 10000 == 0:
                print('decoder occupancy: ',self.occ)
                print(' occupancy shape: ',self.occ.shape)
                print('number of position entries decode: ', self.pos_counter)
                #print('firing rates', self.firing_rate)
                #print('total decoded spikes', self.total_decoded_spike_count)
        
        # if re-loading previous run, get occ from config
        elif self.taskState == 0:
            self.occ = np.asarray(self.config['encoder']['occupancy'])[0]
            self.occ = self.occ.astype('float64')
            self.apply_no_anim_boundary(self.pos_bins_1, self.arm_coords, self.occ, np.nan)
            #print(self.occ)

        return self.occ

    def increment_no_spike_bin(self):

        prob_no_spike = {}
        global_prob_no = np.ones(self.pos_bins)
        for tet_id, tet_fr in self.firing_rate.items():
            # Normalize firing rate
            tet_fr_norm = tet_fr / tet_fr.sum()
            # MEC normalize self.occ to match calcuation in offline decoder
            # MEC 9-3-19 to turn off prob_no_spike
            #prob_no_spike[tet_id] = np.ones(self.pos_bins)
            prob_no_spike[tet_id] = np.exp(-self.time_bin_size / self.config['encoder']['sampling_rate'] *
                                           tet_fr_norm / (self.occ / np.nansum(self.occ)))
            prob_no_spike[tet_id][np.isnan(prob_no_spike[tet_id])] = 0.0
            #print('prob no spike',prob_no_spike[tet_id])

            global_prob_no *= prob_no_spike[tet_id]
        global_prob_no /= global_prob_no.sum()

        # MEC print statement added
        # if self.pos_counter % 10 == 0:
        #    print('global prob no spike: ',global_prob_no)
        #    print('norm occupancy',self.occ / np.nansum(self.occ))

        # Compute likelihood for all previous 0 spike bins
        # update last posterior
        self.prev_posterior = self.posterior

        # Compute no spike likelihood
        # for prob_no in prob_no_spike.values():
        #    self.likelihood *= prob_no
        self.likelihood = global_prob_no

        # Compute posterior for no spike
        self.posterior = self.likelihood * (
            self.transition_mat @ self.prev_posterior)
        # Normalize
        self.posterior = self.posterior / self.posterior.sum()

        # print('likelihood',self.likelihood,np.sum(self.likelihood))

        # we can save the no spike likelihood here
        # QUESTION: what happens to the likelihood and the posterior during long times of no spike??

        return self.posterior, self.likelihood

    # original version
    def increment_bin(self):

        # Compute conditional intensity function (probability of no spike)
        tets_with_no_spikes = self.ntrode_list_array[~self.tetrodes_with_spikes[0]]
        #print('tets with no spikes',tets_with_no_spikes)
        prob_no_spike = {}
        global_prob_no = np.ones(self.pos_bins)
        # MEC: calculate global_prob_no only for missing tets
        for tet_id, tet_fr in self.firing_rate.items():
            if tet_id in tets_with_no_spikes:
                #print('tetrode with no spikes',tet_id)
                # Normalize firing rate
                tet_fr_norm = tet_fr / tet_fr.sum()
                # MEC normalize self.occ to match calcuation in offline decoder
                # MEC 9-3-19 to turn off prob_no_spike
                #prob_no_spike[tet_id] = np.ones(self.pos_bins)
                prob_no_spike[tet_id] = np.exp(-self.time_bin_size / self.config['encoder']['sampling_rate'] *
                                               tet_fr_norm / (self.occ / np.nansum(self.occ)))
                prob_no_spike[tet_id][np.isnan(prob_no_spike[tet_id])] = 0.0

                # MEC: replace with prob_no_spike only for missing tets
                global_prob_no *= prob_no_spike[tet_id]
        global_prob_no /= global_prob_no.sum()

        # MEC print statement added
        # if self.pos_counter % 10000 == 0:
        #    print('global prob no spike: ',global_prob_no)

        # Update last posterior
        self.prev_posterior = self.posterior

        # where should we introduce occupancy normalization of observation????

        # Compute likelihood for previous bin with spikes
        self.likelihood = self.observation * global_prob_no
        # MEC: i think this should also be normalized
        self.likelihood = self.likelihood / self.likelihood.sum()
        # print('observation',self.observation)

        # Compute posterior
        # MEC: switch to nansum
        self.posterior = self.likelihood * \
            (self.transition_mat @ self.prev_posterior)
        #self.posterior = self.likelihood * np.nansum(self.transition_mat * self.prev_posterior,axis=1)
        # Normalize
        # MEC: switch to nansum
        self.posterior = self.posterior / self.posterior.sum()
        #self.posterior = self.posterior / np.nansum(self.posterior)

        # print('likelihood',self.likelihood,np.sum(self.likelihood))
        # Save resulting posterior
        # self.record.write_record(realtime_base.RecordIDs.DECODER_OUTPUT,
        #                          self.current_time_bin * self.time_bin_size,
        #                          *self.posterior)

        # reset values for next observation
        self.current_spike_count = 0
        # np.ones is resetting the observation array for the next time bin
        # observation is filled with deocoded spikes above in add_observation
        self.observation = np.ones(self.pos_bins)
        self.tetrodes_with_spikes = np.zeros(
            (1, len(self.ntrode_list)), dtype=np.bool)

        return self.posterior, self.likelihood

    def calculate_posterior_arm_sum(self, posterior, ripple_time_bin):

        # hmmmm we have defined arm_coords many places - try to just define it once
        # this needs to take into account number of arms

        # 8 arm version - i think this should match the transition matrix (before was all values-1, not sure why)
        #arm_coords_rt = [[0, 8], [13, 24], [29, 40], [45, 56], [
        #    61, 72], [77, 88], [93, 104], [109, 120], [125, 136]]
        # we should be able to use self.arm_coords

        # 4 arm version
        #arm_coords_rt = [[0,8],[13,24],[29,40],[45,56],[61,72]]

        #post_sum_bin_length = 20
        posterior = posterior
        ripple_time_bin = ripple_time_bin

        # calculate the sum of the decode for each arm (box, then arms 1-8)
        # posterior is just an array 136 items long, so this should work

        # for here just calculate sum for current posterior - do cumulative sum in main_process
        # to turn off posterior sum, comment out for loop below
        self.posterior_sum_result = np.zeros((1, 9))
        #print('zeros shape: ',self.posterior_sum_result)

        for region_ind, (start_ind, stop_ind) in enumerate(self.arm_coords):
            self.posterior_sum_result[0,region_ind] = posterior[start_ind:stop_ind + 1].sum()
            # print(self.posterior_sum_result)
            #print('whole posterior sum',posterior.sum())
        # posterior sum vector seems good - always adds to 1
        # yes, i can find a ripple that doesnt sum to 1, but this line didnt display anything
        if self.posterior_sum_result.sum() < 0.99:
            print('posterior sum vector sum', self.posterior_sum_result.sum())
        # print('posterior',posterior)

        return self.posterior_sum_result


class PPDecodeManager(realtime_base.BinaryRecordBaseWithTiming):
    def __init__(self, rank, config, local_rec_manager, send_interface: DecoderMPISendInterface,
                 spike_decode_interface: SpikeDecodeRecvInterface, pos_interface: realtime_base.DataSourceReceiver,
                 lfp_interface: LFPTimekeeperRecvInterface):
        super(PPDecodeManager, self).__init__(rank=rank,
                                              local_rec_manager=local_rec_manager,
                                              send_interface=send_interface,
                                              rec_ids=[realtime_base.RecordIDs.DECODER_OUTPUT,
                                                       realtime_base.RecordIDs.LIKELIHOOD_OUTPUT,
                                                       realtime_base.RecordIDs.DECODER_MISSED_SPIKES,
                                                       realtime_base.RecordIDs.OCCUPANCY],
                                              rec_labels=[['bin_timestamp', 'wall_time', 'velocity', 'real_pos',
                                                           'raw_x', 'raw_y', 'smooth_x', 'smooth_y', 'spike_count', 'next_bin',
                                                           'taskState','ripple', 'ripple_number', 'ripple_length', 'shortcut_message',
                                                           'box', 'arm1', 'arm2', 'arm3', 'arm4', 'arm5', 'arm6', 'arm7', 'arm8',
                                                           'cred_int'] +
                                                          ['x{:0{dig}d}'.
                                                           format(x, dig=len(str(config['encoder']
                                                                                 ['position']['bins'])))
                                                           for x in range(config['encoder']['position']['bins'])],
                                                          ['bin_timestamp', 'wall_time', 'real_pos', 'spike_count'] +
                                                          ['x{:0{dig}d}'.
                                                           format(x, dig=len(str(config['encoder']
                                                                                 ['position']['bins'])))
                                                           for x in range(config['encoder']['position']['bins'])],
                                                          ['timestamp', 'elec_grp_id',
                                                              'real_bin', 'late_bin'],
                                                          ['bin_timestamp', 'bin', 'raw_x', 'raw_y',
                                                           'segment', 'pos_on_seg',
                                                           'linear_pos', 'velocity'] + ['x{:0{dig}d}'.
                                                                                        format(x, dig=len(str(config['encoder']
                                                                                                              ['position']['bins'])))
                                                                                        for x in range(config['encoder']['position']['bins'])]],
                                              rec_formats=['qdddddddqqqqqqqdddddddddd' + 'd' * config['encoder']['position']['bins'],
                                                           'qddq' + 'd' *
                                                           config['encoder']['position']['bins'],
                                                           'qiii',
                                                           'qqqqqddd' + 'd' * config['encoder']['position']['bins']])
        # i think if you change second q to d above, then you can replace real_pos_time
        # with velocity
        # NOTE: q is symbol for integer, d is symbol for decimal

        self.config = config
        self.mpi_send = send_interface
        self.spike_dec_interface = spike_decode_interface
        self.pos_interface = pos_interface
        self.lfp_interface = lfp_interface

        # initialize velocity calc and linear position assignment functions
        self.raw_x = 0
        self.raw_y = 0
        self.current_vel = 0
        self.smooth_x = 0
        self.smooth_y = 0
        self.smooth_vel = 0
        self.velCalc = VelocityCalculator(self.config)
        self.linPosAssign = LinearPositionAssignment(self.config)

        # Send binary record register message
        # self.mpi_send.send_record_register_messages(self.get_record_register_messages())

        self.msg_counter = 0
        self.pos_msg_counter = 0
        self.ntrode_list = []

        self.current_time_bin = 0
        self.time_bin_size = self.config['pp_decoder']['bin_size']
        self.pp_decoder = PointProcessDecoder(pos_range=[self.config['encoder']['position']['lower'],
                                                         self.config['encoder']['position']['upper']],
                                              pos_bins=self.config['encoder']['position']['bins'],
                                              time_bin_size=self.time_bin_size,
                                              arm_coor=self.config['encoder']['position']['arm_pos'],
                                              uniform_gain=config['pp_decoder']['trans_mat_uniform_gain'],
                                              config=self.config)
        # 7-2-19, added spike count for each decoding bin
        self.spike_count = 0
        self.ripple_thresh_decoder = False
        self.ripple_time_bin = 0
        self.no_ripple_time_bin = 0
        self.replay_target_arm = self.config['ripple_conditioning']['replay_target_arm']
        self.posterior_arm_sum = np.zeros((1, 9))
        self.num_above = 0
        self.ripple_number = 0
        self.shortcut_message_sent = False
        self.dropped_spikes = 0
        self.previous_spike_timestamp = 0
        self.lfp_timekeeper_counter = 1
        self.lfp_msg_counter = 0
        self.decode_loop_counter = 1
        self.used_next_bin = False
        self.decoded_spike = 0
        self.tetrodes_with_spikes = []
        self.taskState = 1

        # circular buffer for decoder
        self.decoded_spike_counter = 0
        self.spike_buffer_size = self.config['pp_decoder']['circle_buffer']
        self.decoded_spike_array = np.zeros((self.spike_buffer_size, self.config['encoder']['position']['bins']+3))
        self.decoder_bin_delay = self.config['pp_decoder']['bin_delay']

        # credible interval and posterior max
        self.spxx = []
        self.crit_ind = 0
        self.posterior_max = 0

    def register_pos_interface(self):
        # Register position, right now only one position channel is supported
        self.pos_interface.register_datatype_channel(-1)
        if self.config['datasource'] == 'trodes':
            self.trodessource = True
            # done, MEC 6-30-19
            #self.class_log.warning("*****Position data subscribed, but update_position() needs to be changed to fit CameraModule position data. Delete this message when implemented*****")
        else:
            self.trodessource = False

    def turn_on_datastreams(self):
        self.pos_interface.start_all_streams()

    def select_ntrodes(self, ntrode_list):
        self.ntrode_list = ntrode_list
        self.pp_decoder.select_ntrodes(ntrode_list)

    def process_next_data(self):
        spike_dec_msg = self.spike_dec_interface.__next__()
        lfp_timekeeper = self.lfp_interface.__next__()
        time = MPI.Wtime()

        if spike_dec_msg is not None:
            self.msg_counter += 1
            if self.msg_counter % 1000 == 0:
                self.class_log.debug(
                    'Received {} decoded messages.'.format(self.msg_counter))

        # this counts every time an lfp timestamp comes in and will be used below so that the posterior
        # loop only runs every 6 msec (9 LFP values)
        if lfp_timekeeper is not None:
            if lfp_timekeeper.elec_grp_id == self.config['trodes_network']['ripple_tetrodes'][0]:
                self.lfp_timekeeper_counter += 1
                #print(lfp_timekeeper.timestamp)

        # this is just a check of the lfp_timekeeper and it seems to work as expected, counts up in between spikes
        # if spike_dec_msg is not None or (self.msg_counter > 0 and lfp_timekeeper is not None and
        #                                 lfp_timekeeper.timestamp > self.previous_spike_timestamp+
        #                                 (self.config['pp_decoder']['bin_size']*2*self.lfp_timekeeper_counter)):
        #    print('5 msec space between decoded spikes. number of empty bins:',self.lfp_timekeeper_counter,
        #          lfp_timekeeper.timestamp,self.previous_spike_timestamp)
        #    self.lfp_timekeeper_counter +=1

        # also want to run this if too much time has passed based on lfp_timekeeper
        # this seems to run now based on the lfp timekeeper, but there are many more dropped spikes

        # circular buffer of last 100 decoded spikes
        if spike_dec_msg is not None:
            #print('spike timestamp',spike_dec_msg.timestamp)

            self.decoded_spike_array[np.mod(self.decoded_spike_counter,self.spike_buffer_size), 0] = spike_dec_msg.timestamp
            self.decoded_spike_array[np.mod(self.decoded_spike_counter,self.spike_buffer_size), 1] = spike_dec_msg.elec_grp_id
            self.decoded_spike_array[np.mod(self.decoded_spike_counter,self.spike_buffer_size), 2:-1] = spike_dec_msg.pos_hist
            self.decoded_spike_array[np.mod(self.decoded_spike_counter,self.spike_buffer_size), -1] = 0

            if np.mod(self.decoded_spike_counter,self.spike_buffer_size) == 0 and self.decoded_spike_counter > 0:
                # count number of 0's in last column, add this to dropped spike count
                self.dropped_spikes += (self.spike_buffer_size - self.decoded_spike_array[:,-1].sum())
                # NOTE: we are not yet saving the dropped spikes!
                # can put dan's saving function here and loop through all the dropped spikes - too slow??
            if self.lfp_timekeeper_counter % 1000 == 0:
                print('number of dropped spikes: ', self.dropped_spikes)
                #print('ripple tet',self.config['trodes_network']['ripple_tetrodes'][0])
                #print(self.spike_buffer_size, self.decoded_spike_array[:,-1].sum())
                #print(self.decoded_spike_array[:,-1])

            self.decoded_spike_counter += 1

            if self.msg_counter % 100 == 0:
                self.record_timing(timestamp=spike_dec_msg.timestamp, elec_grp_id=spike_dec_msg.elec_grp_id,
                                   datatype=datatypes.Datatypes.SPIKES, label='dec_recv')            

        # new version of decoder that runs with every LFP timestamp
        # note we only want this to run every 6 msec, so need something to check the lfp timestamp
        # we could start this only after 10 spikes have been received
        if (lfp_timekeeper is not None and self.lfp_timekeeper_counter % 9 == 0 and
             lfp_timekeeper.elec_grp_id == self.config['trodes_network']['ripple_tetrodes'][0]):
            #self.config['trodes_network']['ripple_tetrodes'][0]
            #print('posterior loop',self.lfp_timekeeper_counter,lfp_timekeeper.timestamp,(lfp_timekeeper.timestamp-2*6*30))

            if self.lfp_timekeeper_counter % 10 == 0:
                self.record_timing(timestamp=lfp_timekeeper.timestamp-2*self.time_bin_size, elec_grp_id=1,
                                   datatype=datatypes.Datatypes.SPIKES, label='post_start')

            # find rows in array that are -12 to -6 msec behind current timestamp
            posterior_spikes = self.decoded_spike_array[(self.decoded_spike_array[:,0]>
                                                        (lfp_timekeeper.timestamp-self.decoder_bin_delay*self.time_bin_size))&
                                                        (self.decoded_spike_array[:,0]<
                                                        (lfp_timekeeper.timestamp-(self.decoder_bin_delay-1)*self.time_bin_size))]
            if posterior_spikes.shape[0] > 0:
                #print(self.lfp_timekeeper_counter)
                #print(lfp_timekeeper.timestamp/30)
                #if self.lfp_timekeeper_counter % 100 == 0:
                #print('posterior spikes',posterior_spikes.shape[0])
                self.spike_count = posterior_spikes.shape[0]
                # set last column in decoded_spike_array to 1 - might be able to do this directly to posterior_spikes
                self.decoded_spike_array[np.where((self.decoded_spike_array[:,0]>
                                                (lfp_timekeeper.timestamp-self.decoder_bin_delay*self.time_bin_size))&
                                                (self.decoded_spike_array[:,0]<
                                                (lfp_timekeeper.timestamp-(self.decoder_bin_delay-1)*self.time_bin_size)))[0][:],-1] = 1

                # check for multiple timestamps here - if tet list is split in 2 we can just remove any duplicates
                # ideally we might want to only remove triplicates or more, but not sure how to do this
                vals, inverse, count = np.unique(posterior_spikes[:,0], return_inverse=True,return_counts=True)

                idx_vals_repeated = np.where(count > 1)[0]
                vals_repeated = vals[idx_vals_repeated]

                rows, cols = np.where(inverse == idx_vals_repeated[:, np.newaxis])
                _, inverse_rows = np.unique(rows, return_index=True)
                res = np.split(cols, inverse_rows[1:])

                if res[0].shape[0] > 0:
                    posterior_spikes = posterior_spikes[~res[0],:]
                    print('removed duplicate spikes in decoder')
                else:
                    posterior_spikes = posterior_spikes

                # run add observation
                for i in range(0, posterior_spikes.shape[0]):
                    self.pp_decoder.add_observation(spk_elec_grp_id=posterior_spikes[i,1],
                                                spk_pos_hist=posterior_spikes[i,2:-1],
                                                vel_data=self.current_vel, taskState=self.taskState)
               
                # run increment bin: to calculate like and posterior
                posterior, likelihood = self.pp_decoder.increment_bin()
                
                # record timing
                if self.lfp_timekeeper_counter % 10 == 0:
                    self.record_timing(timestamp=lfp_timekeeper.timestamp-self.decoder_bin_delay*self.time_bin_size, elec_grp_id=1,
                                   datatype=datatypes.Datatypes.SPIKES, label='post_end')                

                # calculate arm sum, max, and cred interval
                self.posterior_arm_sum = self.pp_decoder.calculate_posterior_arm_sum(
                    posterior, self.ripple_time_bin)
                #self.posterior_arm_sum = np.zeros((1,9))

                # add credible interval here and add to message
                self.spxx = np.sort(posterior)[::-1]
                self.crit_ind = (np.nonzero(np.diff(np.cumsum(self.spxx) >= 0.95, prepend=False))[0] + 1)[0]
                #if spike_dec_msg is not None and self.msg_counter % 10000 == 0:
                #    print('credible interval',self.crit_ind)

                # calculate max position of posterior
                self.posterior_max = posterior.argmax()

                # send posterior message to main process
                # timestamp is the beginning of the bin: lfp_timekeeper.timestamp-2*5*30
                self.mpi_send.send_posterior_message(lfp_timekeeper.timestamp-self.decoder_bin_delay*self.time_bin_size,
                                                     lfp_timekeeper.timestamp-self.decoder_bin_delay*self.time_bin_size, 
                                                     self.posterior_arm_sum[0][0],
                                                     self.posterior_arm_sum[0][1], self.posterior_arm_sum[0][2],
                                                     self.posterior_arm_sum[0][3], self.posterior_arm_sum[0][4],
                                                     self.posterior_arm_sum[0][5], self.posterior_arm_sum[0][6],
                                                     self.posterior_arm_sum[0][7], self.posterior_arm_sum[0][8],
                                                     self.spike_count,self.crit_ind,self.posterior_max)

                # save posterior and likelihood
                self.write_record(realtime_base.RecordIDs.LIKELIHOOD_OUTPUT,
                                  lfp_timekeeper.timestamp-self.decoder_bin_delay*self.time_bin_size, time,
                                  self.pp_decoder.cur_pos, self.spike_count,
                                  *likelihood)

                self.write_record(realtime_base.RecordIDs.DECODER_OUTPUT,
                                  lfp_timekeeper.timestamp-self.decoder_bin_delay*self.time_bin_size, time,
                                  self.current_vel,
                                  self.pp_decoder.cur_pos, self.raw_x, self.raw_y, self.smooth_x, self.smooth_y,
                                  self.spike_count, self.used_next_bin, self.taskState,
                                  self.ripple_thresh_decoder, self.ripple_number, 
                                  self.ripple_time_bin, self.shortcut_message_sent, self.crit_ind,
                                  self.posterior_arm_sum[0][0], self.posterior_arm_sum[0][1],
                                  self.posterior_arm_sum[0][2], self.posterior_arm_sum[0][3], self.posterior_arm_sum[0][4],
                                  self.posterior_arm_sum[0][5], self.posterior_arm_sum[0][6], self.posterior_arm_sum[0][7],
                                  self.posterior_arm_sum[0][8],
                                  *posterior)                
                
            elif posterior_spikes.shape[0] == 0:
                self.spike_count = 0
                # run increment no spike
                posterior, likelihood = self.pp_decoder.increment_no_spike_bin()

                # calculate arm sum, max, and cred interval
                self.posterior_arm_sum = self.pp_decoder.calculate_posterior_arm_sum(
                    posterior, self.ripple_time_bin)
                #self.posterior_arm_sum = np.zeros((1,9))

                # add credible interval here and add to message
                self.spxx = np.sort(posterior)[::-1]
                self.crit_ind = (np.nonzero(np.diff(np.cumsum(self.spxx) >= 0.95, prepend=False))[0] + 1)[0]
                #if spike_dec_msg is not None and self.msg_counter % 10000 == 0:
                #    print('credible interval',self.crit_ind)

                # calculate max position of posterior
                self.posterior_max = posterior.argmax()

                # send posterior message to main process
                # timestamp is the beginning of the bin: lfp_timekeeper.timestamp-2*5*30
                self.mpi_send.send_posterior_message(lfp_timekeeper.timestamp-self.decoder_bin_delay*self.time_bin_size,
                                                     lfp_timekeeper.timestamp-self.decoder_bin_delay*self.time_bin_size, 
                                                     self.posterior_arm_sum[0][0],
                                                     self.posterior_arm_sum[0][1], self.posterior_arm_sum[0][2],
                                                     self.posterior_arm_sum[0][3], self.posterior_arm_sum[0][4],
                                                     self.posterior_arm_sum[0][5], self.posterior_arm_sum[0][6],
                                                     self.posterior_arm_sum[0][7], self.posterior_arm_sum[0][8],
                                                     self.spike_count,self.crit_ind,self.posterior_max)

                # save posterior and likelihood
                self.write_record(realtime_base.RecordIDs.LIKELIHOOD_OUTPUT,
                                  lfp_timekeeper.timestamp-self.decoder_bin_delay*self.time_bin_size, time,
                                  self.pp_decoder.cur_pos, self.spike_count,
                                  *likelihood)

                self.write_record(realtime_base.RecordIDs.DECODER_OUTPUT,
                                  lfp_timekeeper.timestamp-self.decoder_bin_delay*self.time_bin_size, time,
                                  self.current_vel,
                                  self.pp_decoder.cur_pos, self.raw_x, self.raw_y, self.smooth_x, self.smooth_y,
                                  self.spike_count, self.used_next_bin, self.taskState,
                                  self.ripple_thresh_decoder, self.ripple_number, 
                                  self.ripple_time_bin, self.shortcut_message_sent, self.crit_ind,
                                  self.posterior_arm_sum[0][0], self.posterior_arm_sum[0][1],
                                  self.posterior_arm_sum[0][2], self.posterior_arm_sum[0][3], self.posterior_arm_sum[0][4],
                                  self.posterior_arm_sum[0][5], self.posterior_arm_sum[0][6], self.posterior_arm_sum[0][7],
                                  self.posterior_arm_sum[0][8],
                                  *posterior)                
                
        # position and velocity loop
        pos_msg = self.pos_interface.__next__()

        if pos_msg is not None:
            # self.class_log.debug("Pos msg received.")
            pos_data = pos_msg[0]
            # print(pos_data)
            if not self.trodessource:
                self.pp_decoder.update_position(
                    pos_timestamp=pos_data.timestamp, pos_data=pos_data.x, taskState=self.taskState)
            else:
                #self.pp_decoder.update_position(pos_timestamp=pos_data.timestamp, pos_data=pos_data.x)
                # we want to use linearized position here
                # try calculating velocity with smoothed position, see if it looks better
                #self.raw_x = pos_data.x
                #self.raw_y = pos_data.y
                #self.current_vel = self.velCalc.calculator(pos_data.x, pos_data.y)

                # smooth position not velocity
                # now try smooth position and then velocity
                self.smooth_x = self.velCalc.smooth_x_position(pos_data.x)
                self.smooth_y = self.velCalc.smooth_y_position(pos_data.y)
                #self.current_vel = self.velCalc.calculator_no_smooth(self.smooth_x, self.smooth_y)
                self.current_vel = self.velCalc.calculator(
                    self.smooth_x, self.smooth_y)
                current_pos = self.linPosAssign.assign_position(
                    pos_data.segment, pos_data.position)

                # try turning off all of these calculations
                #self.smooth_x = pos_data.x
                #self.smooth_y = pos_data.y
                #self.current_vel = 0
                #self.smooth_vel = 0
                #current_pos = 0

                # MEC added: return occupancy
                occupancy = self.pp_decoder.update_position(pos_timestamp=pos_data.timestamp,
                                                            pos_data=current_pos, vel_data=self.current_vel,
                                                            taskState=self.taskState)

                # save record with occupancy
                # TO DO: save raw X and raw Y also
                self.write_record(realtime_base.RecordIDs.OCCUPANCY,
                                  pos_data.timestamp, self.current_time_bin,
                                  pos_data.x, pos_data.y,
                                  pos_data.segment, pos_data.position,
                                  current_pos, self.current_vel, *occupancy)

                # send message VEL_POS to main_process so that shortcut message can by filtered by velocity and position
                # note: used to send bin_timestamp, but main process doesnt use the timestamp for anything, so 
                # we can use the pos data timestamp instead
                self.mpi_send.send_vel_pos_message(pos_data.timestamp,
                                                   current_pos, self.current_vel)

                self.pos_msg_counter += 1
                # this prints position and velocity every 5 sec (150)
                if self.pos_msg_counter % 150 == 0:
                    print('position = ', current_pos, ' and velocity = ', np.around(self.current_vel, decimals=2),
                          'smooth velocity = ', np.around(
                              self.smooth_vel, decimals=2), 'segment = ', pos_data.segment,
                          'smooth_x', np.around(self.smooth_x, decimals=2), 'smooth_y', np.around(self.smooth_y, decimals=2))


                # read taskstate.txt for taskState - needs to be updated manually at begin of session
                # 1 = first cued arm trials, 2 = content trials, 3 = second cued arm trials
                # lets try 15 instead of 60
                if self.pos_msg_counter % 15 == 0:
                  #if self.vel_pos_counter % 1000 == 0:
                      #print('thresh_counter: ',self.thresh_counter)
                      with open('config/taskstate.txt') as taskState_file:
                          fd = taskState_file.fileno()
                          fcntl.fcntl(fd, fcntl.F_SETFL, os.O_NONBLOCK)
                          # read file
                          for taskState_file_line in taskState_file:
                              pass
                          new_taskState = taskState_file_line
                      self.taskState = np.int(new_taskState[0:1])
                      #print('taskState in decoder',self.taskState)

                #print(pos_data.x, pos_data.segment)
                # TODO implement trodes cameramodule update position function
                # If data source is trodes, then pos_data is of class CameraModulePoint, in datatypes.py
                #   pos_data.timestamp: trodes timestamp
                #   pos_data.segment:   linear track segment animal is on (0 if none defined)
                #   pos_data.position:  position along segment (0 if no linear tracks defined)
                #   pos_data.x and y:   raw coordinates of animal, (0,0) is top left of image
                #                        bottom right is full resolution of image
                # Update position function implementation is left
                pass


class BayesianDecodeManager(realtime_base.BinaryRecordBaseWithTiming):
    def __init__(self, rank, config, local_rec_manager, send_interface: DecoderMPISendInterface,
                 spike_decode_interface: SpikeDecodeRecvInterface):
        super(BayesianDecodeManager, self).__init__(rank=rank,
                                                    local_rec_manager=local_rec_manager,
                                                    send_interface=send_interface,
                                                    rec_ids=[
                                                        realtime_base.RecordIDs.DECODER_OUTPUT],
                                                    rec_labels=[['timestamp'] +
                                                                ['x' + str(x) for x in
                                                                 range(config['encoder']['position']['bins'])]],
                                                    rec_formats=['q' + 'd' * config['encoder']['position']['bins']])

        self.config = config
        self.mpi_send = send_interface
        self.spike_interface = spike_decode_interface

        # Send binary record register message
        # self.mpi_send.send_record_register_messages(self.get_record_register_messages())

        self.msg_counter = 0

        self.current_time_bin = 0
        self.current_est_pos_hist = np.ones(
            self.config['encoder']['position']['bins'])
        self.current_spike_count = 0
        self.ntrode_list = []

    def turn_on_datastreams(self):
        # Do nothing, no datastreams for this decoder
        pass

    def select_ntrodes(self, ntrode_list):
        self.ntrode_list = ntrode_list

    def process_next_data(self):
        spike_dec_msg = self.spike_interface.__next__()

        if spike_dec_msg is not None:

            if self.msg_counter % 100 == 0:
                self.record_timing(timestamp=spike_dec_msg.timestamp, elec_grp_id=spike_dec_msg.elec_grp_id,
                                   datatype=datatypes.Datatypes.SPIKES, label='dec_recv')

            if self.current_time_bin == 0:
                self.current_time_bin = math.floor(
                    spike_dec_msg.timestamp / self.config['bayesian_decoder']['bin_size'])
                spike_time_bin = self.current_time_bin
            else:
                spike_time_bin = math.floor(
                    spike_dec_msg.timestamp / self.config['bayesian_decoder']['bin_size'])

            if spike_time_bin == self.current_time_bin:
                # Spike is in current time bin
                self.current_est_pos_hist *= spike_dec_msg.pos_hist
                self.current_est_pos_hist = self.current_est_pos_hist / \
                    np.max(self.current_est_pos_hist)
                self.current_spike_count += 1

            elif spike_time_bin > self.current_time_bin:
                # Spike is in next time bin, advance to tracking next time bin
                self.write_record(realtime_base.RecordIDs.DECODER_OUTPUT,
                                  self.current_time_bin *
                                  self.config['bayesian_decoder']['bin_size'],
                                  *self.current_est_pos_hist)
                self.current_spike_count = 1
                self.current_est_pos_hist = spike_dec_msg.pos_hist
                self.current_time_bin = spike_time_bin

            elif spike_time_bin < self.current_time_bin:
                # Spike is in an old time bin, discard and mark as missed
                self.class_log.debug(
                    'Spike was excluded from Bayesian decode calculation, arrived late.')
                pass

            self.msg_counter += 1
            if self.msg_counter % 1000 == 0:
                self.class_log.debug(
                    'Received {} decoded messages.'.format(self.msg_counter))

            pass
            # self.class_log.debug(spike_dec_msg)


class DecoderRecvInterface(realtime_base.RealtimeMPIClass):
    def __init__(self, comm: MPI.Comm, rank, config, decode_manager: BayesianDecodeManager):
        super(DecoderRecvInterface, self).__init__(
            comm=comm, rank=rank, config=config)

        self.dec_man = decode_manager

        self.req = self.comm.irecv(
            tag=realtime_base.MPIMessageTag.COMMAND_MESSAGE)

    def __next__(self):
        rdy, msg = self.req.test()
        if rdy:
            self.process_request_message(msg)

            self.req = self.comm.irecv(
                tag=realtime_base.MPIMessageTag.COMMAND_MESSAGE)

    def process_request_message(self, message):
        if isinstance(message, realtime_base.TerminateMessage):
            self.class_log.debug("Received TerminateMessage")
            raise StopIteration()

        elif isinstance(message, binary_record.BinaryRecordCreateMessage):
            self.dec_man.set_record_writer_from_message(message)

        elif isinstance(message, realtime_base.ChannelSelection):
            self.class_log.debug(
                "Received NTrode channel selection {:}.".format(message.ntrode_list))
            self.dec_man.select_ntrodes(message.ntrode_list)

        elif isinstance(message, realtime_base.TimeSyncInit):
            self.class_log.debug("Received TimeSyncInit.")
            self.dec_man.sync_time()

        elif isinstance(message, realtime_base.TurnOnDataStream):
            self.class_log.debug("Turn on data stream")
            self.dec_man.turn_on_datastreams()

        elif isinstance(message, realtime_base.TimeSyncSetOffset):
            self.dec_man.update_offset(message.offset_time)

        elif isinstance(message, realtime_base.StartRecordMessage):
            self.dec_man.start_record_writing()

        elif isinstance(message, realtime_base.StopRecordMessage):
            self.dec_man.stop_record_writing()


class DecoderProcess(realtime_base.RealtimeProcess):
    def __init__(self, comm: MPI.Comm, rank, config):
        super(DecoderProcess, self).__init__(
            comm=comm, rank=rank, config=config)

        self.local_rec_manager = binary_record.RemoteBinaryRecordsManager(manager_label='state', local_rank=rank,
                                                                          manager_rank=config['rank']['supervisor'])

        self.terminate = False
        self.main_loop_counter = 1

        self.mpi_send = DecoderMPISendInterface(
            comm=comm, rank=rank, config=config)
        self.spike_decode_interface = SpikeDecodeRecvInterface(
            comm=comm, rank=rank, config=config)
        self.lfp_interface = LFPTimekeeperRecvInterface(
            comm=comm, rank=rank, config=config)

        if config['datasource'] == 'simulator':
            self.pos_interface = simulator_process.SimulatorRemoteReceiver(comm=self.comm,
                                                                           rank=self.rank,
                                                                           config=self.config,
                                                                           datatype=datatypes.Datatypes.LINEAR_POSITION)
        elif config['datasource'] == 'trodes':
            self.pos_interface = simulator_process.TrodesDataReceiver(comm=self.comm,
                                                                      rank=self.rank,
                                                                      config=self.config,
                                                                      datatype=datatypes.Datatypes.LINEAR_POSITION)

        if config['decoder'] == 'bayesian_decoder':
            self.dec_man = BayesianDecodeManager(rank=rank, config=config,
                                                 local_rec_manager=self.local_rec_manager,
                                                 send_interface=self.mpi_send,
                                                 spike_decode_interface=self.spike_decode_interface)
        elif config['decoder'] == 'pp_decoder':
            self.dec_man = PPDecodeManager(rank=rank, config=config,
                                           local_rec_manager=self.local_rec_manager,
                                           send_interface=self.mpi_send,
                                           spike_decode_interface=self.spike_decode_interface,
                                           pos_interface=self.pos_interface,
                                           lfp_interface=self.lfp_interface)

        self.mpi_recv = DecoderRecvInterface(
            comm=comm, rank=rank, config=config, decode_manager=self.dec_man)

        # First Barrier to finish setting up nodes

        self.class_log.debug("First Barrier")
        self.comm.Barrier()

    def trigger_termination(self):
        self.terminate = True

    def main_loop(self):

        self.dec_man.setup_mpi()
        self.dec_man.register_pos_interface()

        try:
            while not self.terminate:
                #self.main_loop_counter += 1
                # 1,000,000 is about every 2 sec
                #if self.main_loop_counter % 100000 == 0:
                #    #print('decoder main loop')
                #    self.record_timing(timestamp=self.main_loop_counter, elec_grp_id=1,
                #                   datatype=datatypes.Datatypes.SPIKES, label='dec_loop_1')
                self.dec_man.process_next_data()
                #if self.main_loop_counter % 100000 == 0:
                #    #print('decoder main loop')
                #    self.record_timing(timestamp=self.main_loop_counter, elec_grp_id=1,
                #                   datatype=datatypes.Datatypes.SPIKES, label='dec_loop_2')                    
                self.mpi_recv.__next__()

        except StopIteration as ex:
            self.class_log.info(
                'Terminating DecoderProcess (rank: {:})'.format(self.rank))

        # can try to save firing rate dict to a json file here
        self.class_log.info("Decoding Process reached end, exiting.")
