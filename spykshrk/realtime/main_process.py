import fcntl
import os
import struct
import sys
import time

import numpy as np

import spykshrk.realtime.binary_record as binary_record
import spykshrk.realtime.datatypes as datatypes
import spykshrk.realtime.decoder_process as decoder_process
import spykshrk.realtime.realtime_base as realtime_base
import spykshrk.realtime.realtime_logging as rt_logging
import spykshrk.realtime.ripple_process as ripple_process
import spykshrk.realtime.simulator.simulator_process as simulator_process
import spykshrk.realtime.timing_system as timing_system
import spykshrk.realtime.utils as utils
from spykshrk.realtime.gui_process import GuiMainParameterMessage
from mpi4py import MPI

from trodesnetwork.trodes import TrodesAcquisitionSubscriber, TrodesHardware
from zmq import ZMQError
import logging
# from spikegadgets import trodesnetwork as tnp

# try:
#     __IPYTHON__
#     from IPython.terminal.debugger import TerminalPdb
#     bp = TerminalPdb(color_scheme='linux').set_trace
# except NameError as err:
#     print('Warning: NameError ({}), not using ipython (__IPYTHON__ not set), disabling IPython TerminalPdb.'.
#           format(err))
#     bp = lambda: None
# except AttributeError as err:
#     print('Warning: Attribute Error ({}), disabling IPython TerminalPdb.'.format(err))
#     bp = lambda: None

##########################################################################
# Messages (not currently in use)
##########################################################################
class StimulationDecision(rt_logging.PrintableMessage):
    """"Message containing whether or not at a given timestamp a ntrode's ripple filter threshold is crossed.

    This message has helper serializer/deserializer functions to be used to speed transmission.
    """
    _byte_format = 'Ii'

    def __init__(self, timestamp, stim_decision):
        self.timestamp = timestamp
        self.stim_decision = stim_decision

    def pack(self):
        return struct.pack(self._byte_format, self.timestamp, self.stim_decision)

    @classmethod
    def unpack(cls, message_bytes):
        timestamp, stim_decision = struct.unpack(
            cls._byte_format, message_bytes)
        return cls(timestamp=timestamp, stim_decision=stim_decision)

##########################################################################
# Interfaces
##########################################################################
class MainMPISendInterface(realtime_base.RealtimeMPIClass):
    def __init__(self, comm: MPI.Comm, rank, config):

        super().__init__(comm=comm, rank=rank, config=config)

    def send_num_ntrode(self, rank, num_ntrodes):
        self.class_log.debug(
            "Sending number of ntrodes to rank {:}".format(rank))
        self.comm.send(realtime_base.NumTrodesMessage(num_ntrodes), dest=rank,
                       tag=realtime_base.MPIMessageTag.COMMAND_MESSAGE)

    def send_channel_selection(self, rank, channel_selects):
        #print('sending channel selection',rank,channel_selects)
        # print('object',spykshrk.realtime.realtime_base.ChannelSelection(channel_selects))
        self.comm.send(obj=realtime_base.ChannelSelection(channel_selects), dest=rank,
                       tag=realtime_base.MPIMessageTag.COMMAND_MESSAGE)

    # MEC added
    def send_ripple_channel_selection(self, rank, channel_selects):
        self.comm.send(obj=realtime_base.RippleChannelSelection(channel_selects), dest=rank,
                       tag=realtime_base.MPIMessageTag.COMMAND_MESSAGE)

    def send_new_writer_message(self, rank, new_writer_message):
        self.comm.send(obj=new_writer_message, dest=rank,
                       tag=realtime_base.MPIMessageTag.COMMAND_MESSAGE)

    def send_start_rec_message(self, rank):
        self.comm.send(obj=realtime_base.StartRecordMessage(), dest=rank,
                       tag=realtime_base.MPIMessageTag.COMMAND_MESSAGE)

    def send_turn_on_datastreams(self, rank):
        self.comm.send(obj=realtime_base.TurnOnDataStream(), dest=rank,
                       tag=realtime_base.MPIMessageTag.COMMAND_MESSAGE)

    def send_ripple_parameter(self, rank, param_message):
        self.comm.send(obj=param_message, dest=rank,
                       tag=realtime_base.MPIMessageTag.COMMAND_MESSAGE)

    def send_ripple_baseline_mean(self, rank, mean_dict):
        self.comm.send(obj=ripple_process.CustomRippleBaselineMeanMessage(mean_dict=mean_dict), dest=rank,
                       tag=realtime_base.MPIMessageTag.COMMAND_MESSAGE)

    def send_ripple_baseline_std(self, rank, std_dict):
        self.comm.send(obj=ripple_process.CustomRippleBaselineStdMessage(std_dict=std_dict), dest=rank,
                       tag=realtime_base.MPIMessageTag.COMMAND_MESSAGE)

    def send_time_sync_simulator(self):
        if self.config['datasource'] == 'trodes':
            ranks = list(range(self.comm.size))
            ranks.remove(self.rank)
            for rank in ranks:
                self.comm.send(obj=realtime_base.TimeSyncInit(), dest=rank,
                               tag=realtime_base.MPIMessageTag.COMMAND_MESSAGE)
        else:
            self.comm.send(obj=realtime_base.TimeSyncInit(), dest=self.config['rank']['simulator'],
                           tag=realtime_base.MPIMessageTag.COMMAND_MESSAGE)

    def all_barrier(self):
        self.comm.Barrier()

    def send_time_sync_offset(self, rank, offset_time):
        self.comm.send(obj=realtime_base.TimeSyncSetOffset(offset_time), dest=rank,
                       tag=realtime_base.MPIMessageTag.COMMAND_MESSAGE)

    def terminate_all(self):
        terminate_ranks = list(range(self.comm.size))
        terminate_ranks.remove(self.rank)
        for rank in terminate_ranks:
            self.comm.send(obj=realtime_base.TerminateMessage(), dest=rank,
                           tag=realtime_base.MPIMessageTag.COMMAND_MESSAGE)

    def send_setup_complete(self):
        self.comm.send(obj=realtime_base.SetupComplete(),
                       dest=self.config['rank']['gui'],
                       tag=realtime_base.MPIMessageTag.COMMAND_MESSAGE)


class StimDeciderMPISendInterface(realtime_base.RealtimeMPIClass):
    def __init__(self, comm: MPI.Comm, rank, config):
        super(StimDeciderMPISendInterface, self).__init__(
            comm=comm, rank=rank, config=config)
        self.comm = comm
        self.rank = rank
        self.config = config

    def start_stimulation(self):
        pass

    def send_record_register_messages(self, record_register_messages):
        self.class_log.debug("Sending binary record registration messages.")
        for message in record_register_messages:
            self.comm.send(obj=message, dest=self.config['rank']['supervisor'],
                           tag=realtime_base.MPIMessageTag.COMMAND_MESSAGE)


class StimDeciderMPIRecvInterface(realtime_base.RealtimeMPIClass):
    def __init__(self, comm: MPI.Comm, rank, config, stim_decider, networkclient):
        super(StimDeciderMPIRecvInterface, self).__init__(
            comm=comm, rank=rank, config=config)

        self.stim = stim_decider
        self.networkclient = networkclient

        self.mpi_status = MPI.Status()

        self.feedback_bytes = bytearray(16)
        self.timing_bytes = bytearray(100)

        self.mpi_reqs = []
        self.mpi_statuses = []

        req_feedback = self.comm.Irecv(buf=self.feedback_bytes,
                                       tag=realtime_base.MPIMessageTag.FEEDBACK_DATA.value)
        self.mpi_statuses.append(MPI.Status())
        self.mpi_reqs.append(req_feedback)

    def __iter__(self):
        return self

    def __next__(self):
        rdy = MPI.Request.Testall(
            requests=self.mpi_reqs, statuses=self.mpi_statuses)

        if rdy:
            if self.mpi_statuses[0].source in self.config['rank']['ripples']:
                # MEC: we need to add ripple size to this messsage
                message = ripple_process.RippleThresholdState.unpack(
                    message_bytes=self.feedback_bytes)
                self.stim.update_ripple_threshold_state(timestamp=message.timestamp,
                                                        elec_grp_id=message.elec_grp_id,
                                                        threshold_state=message.threshold_state,
                                                        conditioning_thresh_state=message.conditioning_thresh_state,
                                                        networkclient=self.networkclient)

                self.mpi_reqs[0] = self.comm.Irecv(buf=self.feedback_bytes,
                                                   tag=realtime_base.MPIMessageTag.FEEDBACK_DATA.value)


class MainSimulatorMPIRecvInterface(realtime_base.RealtimeMPIClass):

    def __init__(self, comm: MPI.Comm, rank, config, main_manager):
        super().__init__(comm=comm, rank=rank, config=config)
        self.main_manager = main_manager

        self.mpi_status = MPI.Status()

        self.req_cmd = self.comm.irecv(
            tag=realtime_base.MPIMessageTag.COMMAND_MESSAGE.value)

    def __iter__(self):
        return self

    def __next__(self):

        (req_rdy, msg) = self.req_cmd.test(status=self.mpi_status)

        if req_rdy:
            self.process_request_message(msg)

            self.req_cmd = self.comm.irecv(
                tag=realtime_base.MPIMessageTag.COMMAND_MESSAGE.value)

    def process_request_message(self, message):

        if isinstance(message, simulator_process.SimTrodeListMessage):
            self.main_manager.handle_ntrode_list(message.trode_list)
            print('decoding tetrodes message', message.trode_list)

        # MEC added
        if isinstance(message, simulator_process.RippleTrodeListMessage):
            self.main_manager.handle_ripple_ntrode_list(
                message.ripple_trode_list)
            print('ripple tetrodes message', message.ripple_trode_list)

        elif isinstance(message, binary_record.BinaryRecordTypeMessage):
            self.class_log.debug("BinaryRecordTypeMessage received for rec id {} from rank {}".
                                 format(message.rec_id, self.mpi_status.source))
            self.main_manager.register_rec_type_message(message)

        elif isinstance(message, realtime_base.TimeSyncReport):
            self.main_manager.send_calc_offset_time(
                self.mpi_status.source, message.time)

        elif isinstance(message, realtime_base.TerminateMessage):
            self.class_log.info('Received TerminateMessage from rank {:}, now terminating all.'.
                                format(self.mpi_status.source))

            self.main_manager.trigger_termination()

        elif isinstance(message, realtime_base.BinaryRecordSendComplete):
            self.main_manager.update_all_rank_setup_status(self.mpi_status.source)


class VelocityPositionRecvInterface(realtime_base.RealtimeMPIClass):
    def __init__(self, comm: MPI.Comm, rank, config, stim_decider, networkclient):
        super(VelocityPositionRecvInterface, self).__init__(
            comm=comm, rank=rank, config=config)

        self.stim = stim_decider
        self.networkclient = networkclient
        # NOTE: if you dont know how large the buffer should be, set it to a large number
        # then you will get an error saying what it should be set to
        self.msg_buffer = bytearray(68)
        self.req = self.comm.Irecv(
            buf=self.msg_buffer, tag=realtime_base.MPIMessageTag.VEL_POS.value)

    def __next__(self):
        rdy = self.req.Test()
        mpi_time = MPI.Wtime()
        if rdy:

            message = decoder_process.VelocityPosition.unpack(self.msg_buffer)
            self.req = self.comm.Irecv(
                buf=self.msg_buffer, tag=realtime_base.MPIMessageTag.VEL_POS.value)

            # okay so we are receiving the message! but now it needs to get into the stim decider
            self.stim.velocity_position(
                bin_timestamp=message.bin_timestamp, raw_x=message.raw_x, raw_y=message.raw_y,
                raw_x2=message.raw_x2, raw_y2=message.raw_y2, angle=message.angle,
                angle_well_1=message.angle_well_1, angle_well_2=message.angle_well_2,
                pos=message.pos, vel=message.vel, pos_dec_rank=message.rank,
                networkclient=self.networkclient)
            #print('posterior sum message supervisor: ',message.timestamp,time*1000)
            # return posterior_sum

        else:
            return None


class PosteriorSumRecvInterface(realtime_base.RealtimeMPIClass):
    def __init__(self, comm: MPI.Comm, rank, config, stim_decider, networkclient):
        super(PosteriorSumRecvInterface, self).__init__(
            comm=comm, rank=rank, config=config)

        self.stim = stim_decider
        self.networkclient = networkclient
        # NOTE: if you dont know how large the buffer should be, set it to a large number
        # then you will get an error saying what it should be set to
        # bytearray was 80 before adding spike_count, 92 before adding rank
        self.msg_buffer = bytearray(132)
        self.req = self.comm.Irecv(
            buf=self.msg_buffer, tag=realtime_base.MPIMessageTag.POSTERIOR.value)

    def __next__(self):
        rdy = self.req.Test()
        mpi_time = MPI.Wtime()
        if rdy:

            message = decoder_process.PosteriorSum.unpack(self.msg_buffer)
            self.req = self.comm.Irecv(
                buf=self.msg_buffer, tag=realtime_base.MPIMessageTag.POSTERIOR.value)

            # need to activate record_timing in this class if we want to use this here
            # self.record_timing(timestamp=timestamp, elec_grp_id=elec_grp_id,
            #                   datatype=datatypes.Datatypes.SPIKES, label='post_sum_recv')

            # okay so we are receiving the message! but now it needs to get into the stim decider
            #print('posterior from decoder',message)
            self.stim.posterior_sum(bin_timestamp=message.bin_timestamp, spike_timestamp=message.spike_timestamp,
                                    target=message.target, offtarget=message.offtarget, box=message.box, arm1=message.arm1,
                                    arm2=message.arm2, arm3=message.arm3, arm4=message.arm4, arm5=message.arm5,
                                    arm6=message.arm6, arm7=message.arm7, arm8=message.arm8,
                                    spike_count=message.spike_count, crit_ind=message.crit_ind,
                                    posterior_max=message.posterior_max, dec_rank=message.rank, 
                                    tet1=message.tet1,tet2=message.tet2,tet3=message.tet3,
                                    tet4=message.tet4,tet5=message.tet5,
                                    networkclient=self.networkclient)
            #print('posterior sum message supervisor: ',message.spike_timestamp,time*1000)
            # return posterior_sum

        else:
            return None


class MainGuiRecvInterface(realtime_base.RealtimeMPIClass):
    def __init__(self, comm: MPI.Comm, rank, config, stim_decider):
        super().__init__(comm=comm, rank=rank, config=config)
        self.stim_decider = stim_decider
        self.req = self.comm.irecv(source=self.config["rank"]["gui"])

    def __next__(self):
        rdy, msg = self.req.test()
        if rdy:
            self.stim_decider.process_gui_request_message(msg)
            self.req = self.comm.irecv(source=self.config["rank"]["gui"])


############################################################################
# Remove altogether when system has stabilized to new network API
############################################################################
# class MainProcessClient(tnp.AbstractModuleClient):
#     def __init__(self, name, addr, port, config):
#         super().__init__(name, addr, port)
#         # self.main_manager = main_manager
#         self.config = config
#         self.started = False
#         self.ntrode_list_sent = False
#         self.terminated = False

#     def registerTerminationCallback(self, callback):
#         self.terminate = callback

#     def registerStartupCallback(self, callback):
#         self.startup = callback

#     # MEC added: to get ripple tetrode list
#     def registerStartupCallbackRippleTetrodes(self, callback):
#         self.startupRipple = callback

#     def recv_acquisition(self, command, timestamp):
#         if command == tnp.acq_PLAY:
#             if not self.ntrode_list_sent:
#                 self.startup(
#                     self.config['trodes_network']['decoding_tetrodes'])
#                 # added MEC
#                 self.startupRipple(
#                     self.config['trodes_network']['ripple_tetrodes'])
#                 self.started = True
#                 self.ntrode_list_sent = True

#         if command == tnp.acq_STOP:
#             if not self.terminated:
#                 # self.main_manager.trigger_termination()
#                 self.terminate()
#                 self.terminated = True
#                 self.started = False

#     def recv_quit(self):
#         self.terminate()

class MainProcessClient(object):
    def __init__(self, config, manager):
        self.config = config
        self.manager = manager
        self.started = False

        server_address = utils.get_network_address(config)
        self.acq_sub = TrodesAcquisitionSubscriber(server_address=server_address)
        self.trodes_hardware = TrodesHardware(server_address=server_address)

    def send_statescript_shortcut_message(self, val):
        self.trodes_hardware.ecu_shortcut_message(val)

    def __next__(self):
        try:
            data = self.acq_sub.receive(noblock=True)

            # only start up once
            if ('play' in data['command'] or 'record' in data['command']) and not self.started:
                self.manager.start()
                self.started = True
            if 'stop' in data['command']: # 'stop' for playback, 'stoprecord' for recording
                self.manager.trigger_termination()
        except ZMQError:
            pass

##########################################################################
# Data handlers/managers
##########################################################################
class StimDecider(realtime_base.BinaryRecordBaseWithTiming):
    def __init__(self, rank, config,
                 send_interface: StimDeciderMPISendInterface, ripple_n_above_thresh=sys.maxsize,
                 lockout_time=0):

        super().__init__(rank=rank,
                         local_rec_manager=binary_record.RemoteBinaryRecordsManager(manager_label='state',
                                                                                    local_rank=rank,
                                                                                    manager_rank=config['rank']['supervisor']),
                         send_interface=send_interface,
                         rec_ids=[realtime_base.RecordIDs.STIM_STATE,
                                  realtime_base.RecordIDs.STIM_LOCKOUT,
                                  realtime_base.RecordIDs.STIM_MESSAGE,
                                  realtime_base.RecordIDs.STIM_HEAD_DIRECTION],
                         rec_labels=[['timestamp', 'elec_grp_id', 'threshold_state'],
                                     ['timestamp', 'velocity', 'lockout_num', 'lockout_state', 'tets_above_thresh',
                                      'big_rip_message_sent','spike_count'],
                                     ['bin_timestamp', 'spike_timestamp', 'lfp_timestamp', 'time',
                                      'shortcut_message_sent', 'ripple_number', 'posterior_time_bin', 'delay', 'velocity',
                                      'real_pos','spike_count', 'spike_count_base','taskState',
                                      'posterior_max_arm', 'content_threshold','ripple_end','credible_int',
                                      'max_arm_repeats', 'replay_bin_length',
                                      'target_1','target_2','offtarget_1','offtarget_2',
                                      'spike_count_1','spike_count_2','ripple_tets',
                                      'box_1', 'arm1_1', 'arm2_1', 'arm3_1', 'arm4_1', 
                                      'box_2', 'arm1_2', 'arm2_2', 'arm3_2','arm4_2',
                                      'unique_tets','center_well_dist'],
                                      ['timestamp', 'well', 'raw_x', 'raw_y', 'raw_x2', 'raw_y2',
                                       'angle', 'angle_well_1', 'angle_well_2',
                                       'to_well_angle_range', 'within_angle_range',
                                       'distance_from_center_well', 'max_distance_from_center_well',
                                       'duration', 'lockout_time']],
                         rec_formats=['Iii',
                                      'Idiiddi',
                                      'IIidiiidddidiididiidddddddddddddddddid',
                                      'IHhhhhddddddddd'])
        # NOTE: for binary files: I,i means integer, d means decimal

        self.rank = rank
        self._send_interface = send_interface
        self._ripple_n_above_thresh = ripple_n_above_thresh
        self._lockout_time = lockout_time
        self._ripple_thresh_states = {}
        self._conditioning_ripple_thresh_states = {}
        self._enabled = False
        self.config = config

        self._last_lockout_timestamp = 0
        self._lockout_count = 0
        self._in_lockout = False

        # lockout for conditioning big ripples
        self._conditioning_last_lockout_timestamp = 0
        self._conditioning_in_lockout = False

        # lockout for posterior sum
        self._posterior_in_lockout = False
        self._posterior_last_lockout_timestamp = 0

        # lockout for ripple end to send posterior sum
        self._ripple_end_in_lockout = False
        self._ripple_end_last_lockout_timestamp = 0
        self._ripple_end_lockout_time = 300

        #self.ripple_time_bin = 0
        self.post_sum_sliding_window = self.config['ripple_conditioning']['post_sum_sliding_window']
        self.no_ripple_time_bin = 0
        self.replay_target_arm = self.config['ripple_conditioning']['replay_target_arm']
        self.replay_non_target_arm = self.config['ripple_conditioning']['replay_non_target_arm']
        #self.posterior_arm_sum = np.zeros((1,9))
        self.posterior_arm_sum = np.zeros((9,))
        self.target_sum_array_1 = np.zeros((self.post_sum_sliding_window,))
        self.target_sum_array_2 = np.zeros((self.post_sum_sliding_window,))
        self.offtarget_sum_array_1 = np.zeros((self.post_sum_sliding_window,))
        self.offtarget_sum_array_2 = np.zeros((self.post_sum_sliding_window,))        
        self.target_sum_avg_1 = 0
        self.target_sum_avg_2 = 0
        self.offtarget_sum_avg_1 = 0
        self.offtarget_sum_avg_2 = 0        
        # initialize with single 1 so that first pass throught posterior_sum works
        self.norm_posterior_arm_sum_1 = np.asarray([0, 1, 0, 0, 0, 0, 0, 0, 0])
        self.norm_posterior_arm_sum_2 = np.asarray([0, 1, 0, 0, 0, 0, 0, 0, 0])
        self.box_post = 0
        self.arm1_post = 0
        self.arm2_post = 0
        self.arm3_post = 0
        self.arm4_post = 0
        self.arm5_post = 0
        self.arm6_post = 0
        self.arm7_post = 0
        self.arm8_post = 0
        self.num_above = 0
        self.ripple_number = 0
        self.shortcut_message_sent = False
        self.shortcut_message_arm = 99
        self.lfp_timestamp = 0
        self.bin_timestamp = 0
        self.bin_timestamp_1 = 0
        self.bin_timestamp_2 = 0
        self.spike_timestamp = 0
        self.crit_ind = 0
        self.post_max = 0
        self.dec_rank = 0
        self.target_post = 0
        self.decoder_1_count = 0
        self.decoder_2_count = 0
        self.ripple_num_tet_above = 0
        self.ripple_tet_num_array = np.zeros((self.post_sum_sliding_window,))
        self.ripple_tet_num_avg = 0

        self.velocity = 0
        self.linearized_position = 0
        self.raw_x = 0
        self.raw_y = 0
        self.center_well_pos = self.config['ripple_conditioning']['center_well_position']
        self.center_well_proximity = False
        self.pos_dec_rank = 0
        self.vel_pos_counter = 0
        self.thresh_counter = 0
        self.postsum_timing_counter = 0
        #self.stim_message_sent = 0
        self.big_rip_message_sent = 0
        self.arm_replay_counter = np.zeros((8,))
        self.posterior_time_bin = 0
        self.posterior_spike_count = 0
        self.posterior_arm_threshold = self.config['ripple_conditioning']['posterior_sum_threshold']
        self.ripple_detect_velocity = self.config['ripple_conditioning']['ripple_detect_velocity']
        self._ripple_lockout_count = 0
        self._in_ripple_lockout = False
        self._ripple_last_lockout_timestamp = 0
        self._ripple_lockout_time = self.config['ripple_conditioning']['ripple_lockout']

        # set max repeats allowed at each arm during content trials
        self.max_arm_repeats = 1
        # marker for stim_message to note end of ripple (message sent or end of lockout)
        self.ripple_end = False
        self.shortcut_msg_on = False
        
        # # to send a shortcut message with every ripple
        # self.rip_cond_only = self.config['ripple_conditioning']['posterior_sum_rip_only']
        # renamed to self.reward_mode
        self.reward_mode = self.config['reward_mode']

        # for behavior: 1 sec lockout between shortcut messages
        self._trodes_message_lockout_timestamp = 0
        self._trodes_message_lockout = self.config['ripple_conditioning']['shortcut_msg_lockout']

        # for continous running sum of posterior
        #self.running_post_sum_counter = 0
        self.posterior_sum_array_1 = np.zeros((self.post_sum_sliding_window, 9))
        self.sum_array_sum_1 = np.zeros((9,))
        self.posterior_sum_timestamps = np.zeros((self.post_sum_sliding_window, 1))
        self.posterior_sum_array_2 = np.zeros((self.post_sum_sliding_window, 9))
        self.sum_array_sum_2 = np.zeros((9,))      
        self.post_sum_sliding_window_actual = 0
        self.spike_count_array_2 = np.zeros((self.post_sum_sliding_window,))
        self.spike_count_2 = 0
        self.spike_count_array_1 = np.zeros((self.post_sum_sliding_window,))
        self.spike_count_1 = 0
        self.enc_cred_int_array = np.zeros((5,self.post_sum_sliding_window))
        self.min_unique_tets = self.config['ripple_conditioning']['min_unique_tets']
        self.max_center_well_dist =  self.config['ripple_conditioning']['max_center_well_dist']
        self.center_well_dist_cm = 0

        # for sum of posterior during whole ripple
        self.posterior_sum_ripple = np.zeros((9,))
        self.ripple_bin_count = 0
        self.other_arm_thresh = self.config['ripple_conditioning']['other_arm_threshold']
        self.second_post_sum_thresh = self.config['ripple_conditioning']['second_post_sum_thresh']
        self.other_arms = self.config['ripple_conditioning']['replay_non_target_arm']

        # for spike count average
        if self.config['ripple_conditioning']['session_type'] == 'run':
            self.spike_count_base_avg = self.config['ripple_conditioning']['previous_spike_count_avg']
            self.spike_count_base_std = self.config['ripple_conditioning']['previous_spike_count_std']
        else:
            self.spike_count_base_avg = 0
            self.spike_count_base_std = 0
        self.spk_count_window_len = 3
        self.spk_count_avg_history = 5
        self.spk_count_window = np.zeros((1,self.spk_count_window_len))
        self.spk_count_avg = np.zeros((1,self.spk_count_avg_history))
        self.spike_count = 0

        # credible interval
        self.credible_window = np.zeros((1,6))
        self.credible_avg = 0

        # used to pring out number of replays during session
        self.arm1_replay_counter = 0
        self.arm2_replay_counter = 0
        self.arm3_replay_counter = 0
        self.arm4_replay_counter = 0

        # to make rips longer if replay in box
        self.long_ripple = False
        self.long_rip_sum_counter = 0
        self.long_rip_sum_array = np.zeros((self.post_sum_sliding_window, 9))

        # taskState
        self.taskState = 1

        # instructive task
        self.instructive = self.config['ripple_conditioning']['instructive']
        self.position_limit = 1

        # make arm_coords conditional on number of arms
        self.number_arms = self.config['pp_decoder']['number_arms']
        # take arm coords directely from config
        self.arm_coords = np.array(self.config['encoder']['arm_coords'])

        self.max_pos = self.arm_coords[-1][-1] + 1
        self.pos_bins = np.arange(0, self.max_pos, 1)
        self.pos_trajectory = np.zeros((1,self.max_pos))

        # head direction params
        # only an approximation since camera module timestamps do not come in at
        # regularly spaced time intervals (although we assume the acquisition of
        # frames is more or less at a constant frame rate)
        div, rem = divmod(
            self.config['camera']['frame_rate'] * self.config['head_direction']['min_duration'],
            1)
        if rem:
            div += 1
        self.angle_buffer = [None] * div
        self.min_duration_head_angle = self.config['head_direction']['min_duration']
        self.to_well_angle_range = self.config['head_direction']['well_angle_range']
        self.within_angle_range = self.config['head_direction']['within_angle_range']
        self.max_center_well_dist_head = self.config['head_direction']['max_center_well_dist']
        self.head_direction_lockout_time = self.config['head_direction']['lockout_time']
        self.head_direction_stim_time = 0

        # if self.config['datasource'] == 'trodes':
        #    self.networkclient = MainProcessClient("SpykshrkMainProc", config['trodes_network']['address'],config['trodes_network']['port'], self.config)
        # self.networkclient.initializeHardwareConnection()
        mpi_time = MPI.Wtime()

        # Setup bin rec file
        # main_manager.rec_manager.register_rec_type_message(rec_type_message=self.get_record_register_message())

    def reset(self):
        self._ripple_thresh_states = {}

    def enable(self):
        self.class_log.info('Enabled stim decider.')
        self._enabled = True
        self.reset()

    def disable(self):
        self.class_log.info('Disable stim decider.')
        self._enabled = False
        self.reset()

    def update_n_threshold(self, ripple_n_above_thresh):
        self._ripple_n_above_thresh = ripple_n_above_thresh

    def update_lockout_time(self, lockout_time):
        self._lockout_time = lockout_time
        print('content ripple lockout time:', self._lockout_time)

    def update_conditioning_lockout_time(self, conditioning_lockout_time):
        self._conditioning_lockout_time = conditioning_lockout_time
        print('big ripple lockout time:', self._conditioning_lockout_time)

    def update_posterior_lockout_time(self, posterior_lockout_time):
        self._posterior_lockout_time = posterior_lockout_time
        print('posterior sum lockout time:', self._posterior_lockout_time)

    def update_ripple_threshold_state(self, timestamp, elec_grp_id, threshold_state, conditioning_thresh_state, networkclient):
        # Log timing
        # if self.thresh_counter % 1000 == 0  and self.config['ripple_conditioning']['session_type'] == 'run':
        #     self.record_timing(timestamp=timestamp, elec_grp_id=elec_grp_id,
        #                        datatype=datatypes.Datatypes.LFP, label='stim_rip_state')
        mpi_time = MPI.Wtime()

        # record timestamp from ripple node
        if elec_grp_id == self.config['trodes_network']['ripple_tetrodes'][0]:
            self.lfp_timestamp = timestamp

        #print('received thresh states: ',threshold_state,conditioning_thresh_state)

        # this is for ripple detection
        if self._enabled:
            self.thresh_counter += 1

            self._ripple_thresh_states.setdefault(elec_grp_id, 0)

        #     # only write state if state changed
        #     if self._ripple_thresh_states[elec_grp_id] != threshold_state:
        #         self.write_record(realtime_base.RecordIDs.STIM_STATE,
        #                           timestamp, elec_grp_id, threshold_state)

            # count number of tets above threshold for content ripple
            self._ripple_thresh_states[elec_grp_id] = threshold_state
            num_above = 0
            for state in self._ripple_thresh_states.values():
                num_above += state
            self.ripple_num_tet_above = num_above

            # end ripple lockout
            if self._in_ripple_lockout and (self.bin_timestamp_1 > self._ripple_last_lockout_timestamp + 
                                            self._ripple_lockout_time):
                self._in_ripple_lockout = False
                self.write_record(realtime_base.RecordIDs.STIM_LOCKOUT,
                                  self.bin_timestamp_1, mpi_time, self._ripple_lockout_count, 
                                  self._in_ripple_lockout, num_above, 0, self.spike_count)
                #print('ripple end. ripple num:',self._ripple_lockout_count,'timestamp',self.bin_timestamp_1)
                self._ripple_lockout_count += 1

            # detect ripples 
            if ((num_above >= self._ripple_n_above_thresh) and not self._in_ripple_lockout):
                if self.velocity < self.ripple_detect_velocity:
                    print('detection of ripple. timestamp',self.bin_timestamp_1, 
                    'ripple num:',self._ripple_lockout_count)

                # this starts the lockout for the content ripple threshold and tells us there is a ripple
                self._in_ripple_lockout = True
                self._ripple_last_lockout_timestamp = self.bin_timestamp_1
                #print('last lockout timestamp',self._last_lockout_timestamp)

                self.write_record(realtime_base.RecordIDs.STIM_LOCKOUT,self.bin_timestamp_1, 
                    self.velocity, self._ripple_lockout_count, self._in_ripple_lockout,
                                  num_above, 0, self.spike_count)

        return num_above

    def process_gui_request_message(self, message):
        if isinstance(message, GuiMainParameterMessage):
            self.replay_target_arm = message.target_arm
            self.posterior_arm_threshold = message.posterior_threshold
            self.update_n_threshold(message.num_above_threshold)
            self.max_center_well_dist = message.max_center_well_distance
            self.ripple_detect_velocity = message.ripple_velocity_threshold
            self.shortcut_msg_on = message.shortcut_message_on
            self.instructive = message.instructive_task
            self.reward_mode = message.reward_mode
            #if self.decoder_1_count % 200 == 0:
            #    print('posterior threshold:', self.posterior_arm_threshold,
            #        'rip num tets',self._ripple_n_above_thresh,'ripple vel', self.ripple_detect_velocity, 
            #        'reward mode', self.reward_mode,'shortcut:',self.shortcut_msg_on,'arm:',self.replay_target_arm,
            #        'position limit:',self.position_limit,'well dist max (cm)',self.max_center_well_dist)
        else:
            self.class_log.info(f"Received message of unknown type {type(message)}, ignoring")

    # sends statescript message at end of replay
    def posterior_sum_statescript_message(self, arm, networkclient):
        arm = arm
        networkclient = networkclient
        mpi_time = MPI.Wtime()

        #remove posterior_time_bin from printing b/c we arent using it now

        if self.taskState == 2 and self.linearized_position<8:
            print('reward count. arm1:',self.arm1_replay_counter,'arm2:',self.arm2_replay_counter)
            print('max posterior in arm:', arm)
            print('position:', np.around(self.linearized_position, decimals=2),
                  'bin timestamp:', self.bin_timestamp, '1st spike timestamp:', self.spike_timestamp,
                  'lfp timestamp:', self.lfp_timestamp, 
                  'delay bin:', np.around((self.lfp_timestamp - self.bin_timestamp_1) / 30, decimals=1),
                  'delay spike:', np.around((self.lfp_timestamp - self.spike_timestamp) / 30, decimals=1),
                  'spike count 1:', self.spike_count_1, 'spike count 2:', self.spike_count_2, 
                  'sliding window:', self.post_sum_sliding_window,
                  'well dist',np.around(self.center_well_dist_cm,decimals=2))
            print('target',np.around(self.target_sum_avg_1,decimals=3),
                    np.around(self.target_sum_avg_2,decimals=3),
                    'offtarget',np.around(self.offtarget_sum_avg_1,decimals=3),
                   np.around(self.offtarget_sum_avg_2,decimals=3))
            print('ripple tetrodes',self.ripple_tet_num_array,self.ripple_tet_num_avg)
            print('unique tets with good spike',np.nonzero(np.unique(self.enc_cred_int_array))[0].shape[0])
        # for task1 and 3, try just printing when he is in box
        #elif self.config['ripple_conditioning']['session_type'] == 'run':
        elif self.linearized_position<8:    
            print('max posterior in arm:', arm, np.around(self.norm_posterior_arm_sum[arm], decimals=2),
                  'position:', np.around(self.linearized_position, decimals=2))
            #print('unique tets with good spike',np.nonzero(np.unique(self.enc_cred_int_array))[0].shape[0])

        #self.shortcut_message_arm = np.argwhere(self.norm_posterior_arm_sum>self.posterior_arm_threshold)[0][0]
        self.shortcut_message_arm = arm

        if self.taskState == 2 and self.shortcut_message_arm == 1 and self.linearized_position<8:
            self.arm1_replay_counter += 1
        elif self.taskState == 2 and self.shortcut_message_arm == 2 and self.linearized_position<8:
            self.arm2_replay_counter += 1
        elif self.taskState == 2 and self.shortcut_message_arm == 3 and self.linearized_position<8:
            self.arm3_replay_counter += 1
        elif self.taskState == 2 and self.shortcut_message_arm == 4 and self.linearized_position<8:
            self.arm4_replay_counter += 1

        # only send message for arm 1 replay if it was not last rewarded arm
        # also: check statescript and make sure there is no 3 second delay for light/reward! - looks good

        if self.arm_replay_counter[arm - 1] < self.max_arm_repeats:

            # only send message during taskState 2 and for arm 1
            # now includes on/off switch from new_rippple file - self.shortcut_msg_on
            # includes shortcut message lockout

            # 12-23: added filter for unqiue good tets
            # this should be for conditioning - remove crit_ind filter
            # was lfp_timestamp, now bin_timestamp_1
            if (self.taskState == 2 and self.shortcut_message_arm == self.replay_target_arm and
                (self.bin_timestamp_1 > self._trodes_message_lockout_timestamp + self._trodes_message_lockout)
                and self.reward_mode == "replay" and self.shortcut_msg_on and not self.instructive and 
                self.center_well_proximity and 
                np.nonzero(np.unique(self.enc_cred_int_array))[0].shape[0]>=self.min_unique_tets):
                # NOTE: we can now replace this with the actual shortcut message!
                networkclient.send_statescript_shortcut_message(14)
                print('replay conditoning: statescript trigger 14')

                # old statescript message
                # note: statescript can only execute one function at a time, so trigger function 15 and set replay_arm variable
                #statescript_command = f'replay_arm = {self.shortcut_message_arm};\ntrigger(15);\n'
                #print('string for statescript:',statescript_command)
                #networkclient.sendMsgToModule('StateScript', 'StatescriptCommand', 's', [statescript_command])
                #networkclient.sendMsgToModule('StateScript', 'StatescriptCommand', 's', ['replay_arm = 1;\ntrigger(15);\n'])
                #print('sent StateScript message for arm', self.shortcut_message_arm,
                #      'replay in ripple', self._lockout_count)

                # arm replay counters, only active at wait well and adds to current counter and sets other arms to 0
                # not using this counter currently
                #print('arm replay count:', self.arm_replay_counter)
                self.shortcut_message_sent = True
                # original
                #self._trodes_message_lockout_timestamp = self.lfp_timestamp
                # 2 decoders
                self._trodes_message_lockout_timestamp = self.bin_timestamp_1

            # # to make whitenoise for incorrect arm
            # elif (self.taskState == 2 and self.shortcut_message_arm == self.replay_non_target_arm
            #     and not self.rip_cond_only and self.shortcut_msg_on):
            #     # NOTE: we can now replace this with the actual shortcut message!
            #     # for shortcut, each arm is assigned a different message
            #     networkclient.sendStateScriptShortcutMessage(15)
            #     print('statescript trigger 15')

            #     # arm replay counters, only active at wait well and adds to current counter and sets other arms to 0
            #     # not using this counter currentrly
            #     #print('arm replay count:', self.arm_replay_counter)
            #     self.shortcut_message_sent = True

            self.ripple_end = True
            self.write_record(realtime_base.RecordIDs.STIM_MESSAGE,
                              self.bin_timestamp, self.spike_timestamp, self.lfp_timestamp, mpi_time, self.shortcut_message_sent,
                              self._lockout_count, self.posterior_time_bin,
                              (self.lfp_timestamp - self.bin_timestamp) / 30, self.velocity,self.linearized_position,
                              self.posterior_spike_count, self.spike_count_base_avg, self.taskState,
                              self.shortcut_message_arm, self.posterior_arm_threshold, self.ripple_end, 
                              self.credible_avg, self.max_arm_repeats, 
                              self.post_sum_sliding_window,
                              self.target_sum_avg_1, self.target_sum_avg_2,
                              self.offtarget_sum_avg_1, self.offtarget_sum_avg_2,
                              self.spike_count_1,self.spike_count_2,self.ripple_tet_num_avg,
                              self.norm_posterior_arm_sum_1[0], self.norm_posterior_arm_sum_1[1], self.norm_posterior_arm_sum_1[2],
                              self.norm_posterior_arm_sum_1[3], self.norm_posterior_arm_sum_1[4], self.norm_posterior_arm_sum_2[0],
                              self.norm_posterior_arm_sum_2[1], self.norm_posterior_arm_sum_2[2], self.norm_posterior_arm_sum_2[3],
                              self.norm_posterior_arm_sum_2[4],
                              np.nonzero(np.unique(self.enc_cred_int_array))[0].shape[0],self.center_well_dist_cm)
        else:
            print('more than ', self.max_arm_repeats,' replays of arm', arm, 'in a row!')

    # MEC: this function brings in velocity and linearized position from decoder process

    def velocity_position(
        self, bin_timestamp, raw_x, raw_y, raw_x2, raw_y2, angle, angle_well_1, angle_well_2,
        pos, vel, pos_dec_rank, networkclient):
        
        self.velocity = vel
        self.linearized_position = pos
        self.raw_x = raw_x
        self.raw_y = raw_y
        self.vel_pos_counter += 1
        self.pos_dec_rank = pos_dec_rank

        if self.velocity < self.ripple_detect_velocity:
            #print('immobile, vel = ',self.velocity)
            pass
        if self.linearized_position >= 3 and self.linearized_position <= 5:
            #print('position at rip/wait!')
            pass
        #print('main pos/vel data',self.linearized_position,self.velocity,self.raw_x,self.raw_y)

        if time.time() - self.head_direction_stim_time < self.head_direction_lockout_time:
            return
        
        self.angle_buffer[1:] = self.angle_buffer[:-1]
        self.angle_buffer[0] = angle
        if None in self.angle_buffer: # not enough data yet, don't reward
            return
        is_within_angle_range = (
            abs(max(self.angle_buffer) - min(self.angle_buffer)) <= self.within_angle_range)
        
        x = (raw_x + raw_x2) / 2
        y = (raw_y + raw_y2) / 2
        dist = np.sqrt( (x - self.center_well_pos[0])**2 + (y - self.center_well_pos[1])**2 )
        is_in_center_well_proximity = dist <= self.max_center_well_dist_head

        record_head_direction_stim = False
        
        # this will only work if the angles to the wells +/- the acceptable angle range
        # do not overlap! otherwise we could end up with a situation where the animal's
        # head direction is detected to be pointing to multiple wells
        if (is_within_angle_range and is_in_center_well_proximity and
            abs(angle - angle_well_1) <= self.to_well_angle_range):
            
            networkclient.send_statescript_shortcut_message(14)
            self.class_log.info("Statescript trigger for well 1")
            well = 1
            record_head_direction_stim = True
        if (is_within_angle_range and is_in_center_well_proximity and
            abs(angle - angle_well_2) <= self.to_well_angle_range):

            networkclient.send_statescript_shortcut_message(14)
            self.class_log.info("Statescript trigger for well 2")
            well = 2
            record_head_direction_stim = True

        if record_head_direction_stim:
            self.write_record(
                realtime_base.RecordIDs.STIM_HEAD_DIRECTION, bin_timestamp, well, raw_x, raw_y,
                raw_x2, raw_y2, angle, angle_well_1, angle_well_2, self.to_well_angle_range,
                self.within_angle_range, dist, self.max_center_well_dist_head, 
                self.min_duration_head_angle, self.head_direction_lockout_time)
        

    # MEC: this function sums the posterior during each ripple, then sends shortcut message
    # need to add location filter so it only sends message when rat is at rip/wait well - no, that is in statescript
    def posterior_sum(self, bin_timestamp, spike_timestamp, target, offtarget, box, arm1, arm2, arm3, arm4, arm5, arm6, arm7, arm8, spike_count, crit_ind, posterior_max, dec_rank, tet1,tet2,tet3,tet4,tet5,networkclient):
        mpi_time = MPI.Wtime()
        self.bin_timestamp = bin_timestamp
        self.spike_timestamp = spike_timestamp
        self.box_post = box
        self.arm1_post = arm1
        self.arm2_post = arm2
        self.arm3_post = arm3
        self.arm4_post = arm4
        self.arm5_post = arm5
        self.arm6_post = arm6
        self.arm7_post = arm7
        self.arm8_post = arm8
        self.spike_count = spike_count
        self.crit_ind = crit_ind
        self.post_max = posterior_max
        self.dec_rank = dec_rank
        self.target_post = target
        self.offtarget_post = offtarget
        self.tet1 = tet1
        self.tet2 = tet2
        self.tet3 = tet3
        self.tet4 = tet4
        self.tet5 = tet5

        # bin timestamp for each decoder
        if self.dec_rank == self.config['rank']['decoder'][0]:
            self.bin_timestamp_1 = bin_timestamp
            self.decoder_1_count += 1
        elif self.dec_rank == self.config['rank']['decoder'][1]:
            self.bin_timestamp_2 = bin_timestamp
            self.decoder_2_count += 1
        #print('tet list in main',self.tet1,self.tet2,self.tet3,self.tet4,self.tet5)

        # would like to reset taskstate to 1 automatically - but this didnt work
        #if self.thresh_counter < 15000:
        #    #with open("config/taskstate.txt","a") as reward_arm_file:
        #    #    fd = reward_arm_file.fileno()
        #    #    fcntl.fcntl(fd, fcntl.F_SETFL, os.O_NONBLOCK)                
        #    #    reward_arm_file.write(str(1)+'\n')
        #    print('set tasktate to 1 at beginning')

        # calculate distance to center well
        if self.dec_rank == self.config['rank']['decoder'][0]:
            center_well_dist = np.sqrt(np.square(self.raw_x-self.center_well_pos[0]) + np.square(self.raw_y-self.center_well_pos[1]))
            self.center_well_dist_cm = center_well_dist*self.config['encoder']['cmperpx']
            if center_well_dist < self.max_center_well_dist/self.config['encoder']['cmperpx']:
                self.center_well_proximity = True
                #if self.thresh_counter % 500 == 0:
                #    print('*** at center well ***')
            else:
                self.center_well_proximity = False

        #print('decoder rank',self.dec_rank)
        # lets try 500 instead of 1500
        #if self.thresh_counter % 1500 == 0  and self.config['ripple_conditioning']['session_type'] == 'run':
        if self.decoder_1_count % 400 == 0  and self.config['ripple_conditioning']['session_type'] == 'run':
            # if self.vel_pos_counter % 1000 == 0:
            #print('thresh_counter: ',self.thresh_counter)

            ##############################################################
            # Replaced with GUI
            # with open('config/new_ripple_threshold.txt') as posterior_threshold_file:
            #     fd = posterior_threshold_file.fileno()
            #     fcntl.fcntl(fd, fcntl.F_SETFL, os.O_NONBLOCK)
            #     # read file
            #     for post_thresh_file_line in posterior_threshold_file:
            #         pass
            #     new_posterior_threshold = post_thresh_file_line
            # # this allows half SD increase in ripple threshold (looks for three digits, eg 065 = 6.5 SD)
            # print('line length',len(new_posterior_threshold))
            # if len(new_posterior_threshold) == 33:
            #     self.posterior_arm_threshold = np.int(new_posterior_threshold[8:11]) / 100
            #     #self.ripple_detect_velocity = np.int(new_posterior_threshold[14:17]) / 10
            #     #self.second_post_sum_thresh = np.int(new_posterior_threshold[14:17]) / 100
            #     self.rip_cond_only = np.int(new_posterior_threshold[18:19])
            #     self.shortcut_msg_on = np.int(new_posterior_threshold[20:21])
            #     self._ripple_n_above_thresh = np.int(new_posterior_threshold[22:23])
            #     # insert conditional so replay target arm only used if not instructive
            #     #self.replay_target_arm = np.int(new_posterior_threshold[24:25])
            #     self.position_limit = np.int(new_posterior_threshold[26:28])
            #     self.max_center_well_dist = np.int(new_posterior_threshold[29:32])
            #     print('posterior threshold:', self.posterior_arm_threshold,
            #         'rip num tets',self._ripple_n_above_thresh,'ripple vel', self.ripple_detect_velocity, 
            #         'rip cond', self.rip_cond_only,'shortcut:',self.shortcut_msg_on,'arm:',self.replay_target_arm,
            #         'position limit:',self.position_limit,'well dist max (cm)',self.max_center_well_dist)
            ##############################################################

            with open('config/taskstate.txt', 'rb') as f:
                fd = f.fileno()
                fcntl.fcntl(fd, fcntl.F_SETFL, os.O_NONBLOCK)
                f.seek(-2, os.SEEK_END)
                while f.read(1) != b'\n':
                    f.seek(-2, os.SEEK_CUR)
                self.taskState = int(f.readline().decode()[0:1])
            print('main taskState:',self.taskState)

            if self.decoder_1_count % 1200 == 0:
                print('posterior threshold:', self.posterior_arm_threshold,
                    'rip num tets',self._ripple_n_above_thresh,'ripple vel', self.ripple_detect_velocity, 
                    'reward mode', self.reward_mode,'shortcut msg:',self.shortcut_msg_on,'target arm:',self.replay_target_arm,
                    'position limit:',self.position_limit,'well dist max (cm)',self.max_center_well_dist)            

        # to test shortcut message delay
        #if self.decoder_1_count % 800 == 0 and self.taskState == 1:
        #    print('TESTING! bin timestamp:', self.bin_timestamp, '1st spike timestamp:', self.spike_timestamp,
        #          'lfp timestamp:', self.lfp_timestamp, 
        #          'delay bin:', np.around((self.lfp_timestamp - self.bin_timestamp_1) / 30, decimals=1),
        #          'delay spike:', np.around((self.lfp_timestamp - self.spike_timestamp) / 30, decimals=1))            
        #    #networkclient.sendStateScriptShortcutMessage(14)
        #    networkclient.send_statescript_shortcut_message(14)
        #    print('TESTING: statescript trigger 14', flush=True)            


        # not currently using this - might use in future
        # read arm_reward text file written by trodes to find last rewarded arm
        # use this to prevent repeated rewards to a specific arm (set arm1_replay_counter)
        if self.thresh_counter % 30000 == 0  and self.config['ripple_conditioning']['session_type'] == 'run':
            # reset counters each time you read the file - b/c file might not change
            self.arm_replay_counter = np.zeros((8,))

            # turn this off for now - we are not using this and trodes write to this file
            # with open('config/rewarded_arm_trodes.txt') as rewarded_arm_file:
            #     fd = rewarded_arm_file.fileno()
            #     fcntl.fcntl(fd, fcntl.F_SETFL, os.O_NONBLOCK)
            #     # read file
            #     for rewarded_arm_file_line in rewarded_arm_file:
            #         pass
            #     rewarded_arm = rewarded_arm_file_line
            # rewarded_arm = np.int(rewarded_arm[0:2])
            # #print('last rewarded arm = ', rewarded_arm)
            # if rewarded_arm > 0:
            #     self.arm_replay_counter[rewarded_arm - 1] = 1

        # check posterior lockout and normal lockout with print statement - seems to work
        if not self._posterior_in_lockout and self._in_lockout:
            #print('inside posterior sum delay time, bin timestamp:',bin_timestamp/30)
            pass

        # ***ACTIVE***
        # sliding window sum of posterior at all times

        #self.running_post_sum_counter += 1

        if self.decoder_1_count % 10000 == 0:
            print('running sum of posterior', self.decoder_1_count)

        if self.thresh_counter % 3000 == 0:
            print('decoder1 delay',(self.lfp_timestamp - self.bin_timestamp_1) / 30,'count',self.decoder_1_count)
            print('decoder2 delay',(self.lfp_timestamp - self.bin_timestamp_2) / 30,'count',self.decoder_2_count)
            #print('lockout time',self._lockout_time)
            #print(self.norm_posterior_arm_sum_1[self.other_arms])

        # timer if needed
        # if self.running_post_sum_counter % 1 == 0:
        #    self.record_timing(timestamp=spike_timestamp, elec_grp_id=0,
        #                       datatype=datatypes.Datatypes.LFP, label='postsum_1')

        # sum of taget segment - want a sliding window average for this over 30 msec (5 bins)
        if self.dec_rank == self.config['rank']['decoder'][0]:
            self.target_sum_array_1[np.mod(self.decoder_1_count,
                                        self.post_sum_sliding_window)] = self.target_post
            self.target_sum_avg_1 = self.target_sum_array_1.sum()/len(self.target_sum_array_1)
            self.offtarget_sum_array_1[np.mod(self.decoder_1_count,
                                        self.post_sum_sliding_window)] = self.offtarget_post
            self.offtarget_sum_avg_1 = self.offtarget_sum_array_1.sum()/len(self.offtarget_sum_array_1)
            self.spike_count_array_1[np.mod(self.decoder_1_count,
                                    self.post_sum_sliding_window)] = self.spike_count
            self.spike_count_1 = self.spike_count_array_1.sum()
            self.enc_cred_int_array[0,np.mod(self.decoder_1_count,
                            self.post_sum_sliding_window)] = self.tet1
            self.enc_cred_int_array[1,np.mod(self.decoder_1_count,
                            self.post_sum_sliding_window)] = self.tet2
            self.enc_cred_int_array[2,np.mod(self.decoder_1_count,
                            self.post_sum_sliding_window)] = self.tet3
            self.enc_cred_int_array[3,np.mod(self.decoder_1_count,
                            self.post_sum_sliding_window)] = self.tet4
            self.enc_cred_int_array[4,np.mod(self.decoder_1_count,
                            self.post_sum_sliding_window)] = self.tet5
            #print(self.enc_cred_int_array)
            #print(np.nonzero(np.unique(self.enc_cred_int_array))[0].shape[0])

        elif self.dec_rank == self.config['rank']['decoder'][1]:
            self.target_sum_array_2[np.mod(self.decoder_2_count,
                                        self.post_sum_sliding_window)] = self.target_post
            self.target_sum_avg_2 = self.target_sum_array_2.sum()/len(self.target_sum_array_2)
            self.offtarget_sum_array_2[np.mod(self.decoder_2_count,
                                        self.post_sum_sliding_window)] = self.offtarget_post
            self.offtarget_sum_avg_2 = self.offtarget_sum_array_2.sum()/len(self.offtarget_sum_array_2)
            self.spike_count_array_2[np.mod(self.decoder_2_count,
                                    self.post_sum_sliding_window)] = self.spike_count
            self.spike_count_2 = self.spike_count_array_2.sum()

        # NOTE: need to add an array for the tet lists and then check this for 2 or 3 unique entries 
        # below once a arm end event has been found

        # check arm 2 end
        #if self.offtarget_post > 0.5:
        #    print('arm 2 end',self.offtarget_sum_avg_2,self.offtarget_sum_avg_1)

        #if self.running_post_sum_counter % 1000 == 0:
        #    print('taget sum decoder 1', self.target_sum_avg_1, self.config['rank']['decoder'][0])
        #    print('taget sum decoder 2', self.target_sum_avg_2, self.config['rank']['decoder'][1])  
        #    print('normed posterior sum',self.norm_posterior_arm_sum_1,self.norm_posterior_arm_sum_1[2])     

        # need to do in separately for each decoder
        # new variables for each arm posterior - for function to send statescript message at end of ripple lockout
        if self.dec_rank == self.config['rank']['decoder'][0]:
            new_posterior_sum = np.asarray([self.box_post, self.arm1_post, self.arm2_post, self.arm3_post,
                                        self.arm4_post, self.arm5_post, self.arm6_post, self.arm7_post,
                                        self.arm8_post])

            # for whole ripple sum - add to this array but dont sum or normalize
            # NEW 10-17-20: calculate average here
            self.posterior_sum_array_1[np.mod(self.decoder_1_count,
                                        self.post_sum_sliding_window), :] = new_posterior_sum
            self.sum_array_sum_1 = np.sum(self.posterior_sum_array_1, axis=0)
            self.norm_posterior_arm_sum_1 = self.sum_array_sum_1 / self.post_sum_sliding_window

            # make array of last 9 time bins for number tets above ripple threshold
            self.ripple_tet_num_array[np.mod(self.decoder_1_count,
                                        self.post_sum_sliding_window)] = self.ripple_num_tet_above
            self.ripple_tet_num_avg = np.sum(self.ripple_tet_num_array, axis=0) / self.config['ripple_conditioning']['post_sum_sliding_window']            
        
        elif self.dec_rank == self.config['rank']['decoder'][1]:
            new_posterior_sum = np.asarray([self.box_post, self.arm1_post, self.arm2_post, self.arm3_post,
                                        self.arm4_post, self.arm5_post, self.arm6_post, self.arm7_post,
                                        self.arm8_post])

            # for whole ripple sum - add to this array but dont sum or normalize
            # NEW 10-17-20: calculate average here
            self.posterior_sum_array_2[np.mod(self.decoder_2_count,
                                        self.post_sum_sliding_window), :] = new_posterior_sum
            self.sum_array_sum_2 = np.sum(self.posterior_sum_array_2, axis=0)
            self.norm_posterior_arm_sum_2 = self.sum_array_sum_2 / self.post_sum_sliding_window

        # not using right now
        # keep track of decoder bin timestamp for posteriors in the sliding sum - check how many msec is total diff
        self.posterior_sum_timestamps[np.mod(self.decoder_1_count,
                                             self.post_sum_sliding_window), :] = self.bin_timestamp
        self.post_sum_sliding_window_actual = np.ptp(self.posterior_sum_timestamps) / 30

        # spike count baseline average
        # comment out to prevent updating
        if self.decoder_1_count == 1:
            print('initial spike count baseline:', np.around(self.spike_count_base_avg,decimals=3),
                                                    np.around(self.spike_count_base_std,decimals=3))
        # update spike count baseline if sleep session
        # try updating with roqui to see how much it changes comapred to sleep
        if self.config['ripple_conditioning']['session_type'] == 'sleep':
            self.spike_count_base_avg += ((self.spike_count - self.spike_count_base_avg) 
                                            / ((1000/(self.config['pp_decoder']['bin_size']/30))
                                                *self.config['ripple_conditioning']['spike_count_window_sec']))
            self.spike_count_base_std += ((abs(self.spike_count - self.spike_count_base_avg)  - self.spike_count_base_std)
                                            / ((1000/(self.config['pp_decoder']['bin_size']/30))
                                                *self.config['ripple_conditioning']['spike_count_window_sec'])) 

        if self.decoder_1_count % 2000 == 0:
            print('spike count baseline. mean:', np.around(self.spike_count_base_avg,decimals=3),
                  'std:',np.around(self.spike_count_base_std,decimals=3),
                  'diff:',np.around(self.spike_count_base_avg-(0.5*self.spike_count_base_std),decimals=3))

        if self.decoder_1_count % 30 == 0:
            self.write_record(realtime_base.RecordIDs.STIM_LOCKOUT, self.lfp_timestamp, 0, 0, 2,
                                  self.spike_count_base_avg, self.spike_count_base_std, self.spike_count)

        # sliding window sum of spike count
        self.spk_count_window[0,np.mod(self.decoder_1_count,self.spk_count_window_len)] = self.spike_count
        self.spk_count_avg[0,np.mod(self.decoder_1_count,
                                    self.spk_count_avg_history)] = np.average(self.spk_count_window)

        # sliding window average of credible interval
        self.credible_window[0,np.mod(self.decoder_1_count,6)] = self.crit_ind
        self.credible_avg = np.average(self.credible_window)


        # NEW 10-17-20: check target sum for non-local event, and check other arms, then send message, start lockout
        # NOTE currently 2nd set of tetrodes cutoff is hard coded here
        # NOTE currently no velocity filter
        # for lockout try lfp_timestamp - how often are getting lfp messages???
        if self.config['ripple_conditioning']['number_of_decoders'] == 2:
            if ((self.target_sum_avg_1 > self.posterior_arm_threshold and not self._in_lockout) or 
                (self.target_sum_avg_2 > self.posterior_arm_threshold and not self._in_lockout)):
                #self._in_lockout = True
                if self.target_sum_avg_1 > self.target_sum_avg_2:
                    if self.target_sum_avg_2 > self.second_post_sum_thresh:
                        if (np.all(self.norm_posterior_arm_sum_1[self.other_arms]<self.other_arm_thresh) and 
                            np.all(self.norm_posterior_arm_sum_2[self.other_arms]<self.other_arm_thresh) ):
                            #print('arm1 end detected decode 1',self.target_sum_avg_1 ,self.target_sum_avg_2)
                            self.norm_posterior_arm_sum = self.norm_posterior_arm_sum_1
                            #print(self._in_lockout)
                            self._in_lockout = True
                            self._last_lockout_timestamp = self.bin_timestamp_1
                            #self._lockout_count += 1
                            self.posterior_sum_statescript_message(1, networkclient)

                elif self.target_sum_avg_2 > self.target_sum_avg_1:
                    if self.target_sum_avg_1 > self.second_post_sum_thresh:
                        if (np.all(self.norm_posterior_arm_sum_1[self.other_arms]<self.other_arm_thresh) and 
                            np.all(self.norm_posterior_arm_sum_2[self.other_arms]<self.other_arm_thresh) ):
                            #print('arm1 end detected decode 2',self.target_sum_avg_1 ,self.target_sum_avg_2)
                            self.norm_posterior_arm_sum = self.norm_posterior_arm_sum_2
                            #print(self._in_lockout)
                            self._in_lockout = True
                            self._last_lockout_timestamp = self.bin_timestamp_1
                            #self._lockout_count += 1
                            self.posterior_sum_statescript_message(1, networkclient)              

            # check non-target arm end posterior
            elif ((self.offtarget_sum_avg_1 > self.posterior_arm_threshold and not self._in_lockout) or 
                (self.offtarget_sum_avg_2 > self.posterior_arm_threshold and not self._in_lockout)):
                #self._in_lockout = True
                #print('arm2 end')
                if self.offtarget_sum_avg_1 > self.offtarget_sum_avg_2:
                    if self.offtarget_sum_avg_2 > self.second_post_sum_thresh:
                        if (np.all(self.norm_posterior_arm_sum_1[self.replay_target_arm]<self.other_arm_thresh) and 
                            np.all(self.norm_posterior_arm_sum_2[self.replay_target_arm]<self.other_arm_thresh) ):
                            #print('arm2 end detected decode 1',self.offtarget_sum_avg_1 ,self.offtarget_sum_avg_2)
                            self.norm_posterior_arm_sum = self.norm_posterior_arm_sum_1
                            #print(self._in_lockout)
                            self._in_lockout = True
                            self._last_lockout_timestamp = self.bin_timestamp_1
                            #self._lockout_count += 1
                            self.posterior_sum_statescript_message(2, networkclient)

                elif self.offtarget_sum_avg_2 > self.offtarget_sum_avg_1:
                    if self.offtarget_sum_avg_1 > self.second_post_sum_thresh:
                        if (np.all(self.norm_posterior_arm_sum_1[self.replay_target_arm]<self.other_arm_thresh) and 
                            np.all(self.norm_posterior_arm_sum_2[self.replay_target_arm]<self.other_arm_thresh) ):
                            #print('arm2 end detected decode 2',self.offtarget_sum_avg_1 ,self.offtarget_sum_avg_2)
                            self.norm_posterior_arm_sum = self.norm_posterior_arm_sum_2
                            #print(self._in_lockout)
                            self._in_lockout = True
                            self._last_lockout_timestamp = self.bin_timestamp_1
                            #self._lockout_count += 1
                            self.posterior_sum_statescript_message(2, networkclient) 

        elif self.config['ripple_conditioning']['number_of_decoders'] == 1:
            # send message for end of arm only in target_sum_avg_1
            # now requires both other arm and box to be < 0.2
            if (self.target_sum_avg_1 > self.posterior_arm_threshold and not self._in_lockout):
                #self._in_lockout = True
                if (np.all(self.norm_posterior_arm_sum_1[self.other_arms]<self.other_arm_thresh)
                    and self.norm_posterior_arm_sum_1[0]<self.other_arm_thresh):
                    #print('arm1 end detected decode 1',self.target_sum_avg_1 ,self.target_sum_avg_2)
                    self.norm_posterior_arm_sum = self.norm_posterior_arm_sum_1
                    #print(self._in_lockout)
                    self._in_lockout = True
                    self._last_lockout_timestamp = self.bin_timestamp_1
                    #self._lockout_count += 1
                    # unique tets that are non-zero
                    #np.nonzero(np.unique(self.enc_cred_int_array))[0].shape[0]

                    # this was 1 before
                    self.posterior_sum_statescript_message(self.replay_target_arm, networkclient)    

            elif (self.offtarget_sum_avg_1 > self.posterior_arm_threshold and not self._in_lockout):
                #self._in_lockout = True
                #print('arm2 end')
                if (np.all(self.norm_posterior_arm_sum_1[self.replay_target_arm]<self.other_arm_thresh)
                    and self.norm_posterior_arm_sum_1[0]<self.other_arm_thresh):
                    #print('arm2 end detected decode 1',self.offtarget_sum_avg_1 ,self.offtarget_sum_avg_2)
                    self.norm_posterior_arm_sum = self.norm_posterior_arm_sum_1
                    #print(self._in_lockout)
                    self._in_lockout = True
                    self._last_lockout_timestamp = self.bin_timestamp_1
                    #self._lockout_count += 1
                    # this was 2 before
                    self.posterior_sum_statescript_message(self.other_arms[0], networkclient)


        # end lockout for non-local event
        # also using a timer - 200 msec
        # was bin_timestamp, now try lfp_timestamp
        if (self._in_lockout and self.bin_timestamp_1 > (self._last_lockout_timestamp + self._lockout_time)):
            self._in_lockout = False
            self.write_record(realtime_base.RecordIDs.STIM_LOCKOUT,
                              self.bin_timestamp, mpi_time, self._lockout_count, self._in_lockout,
                              0, self.big_rip_message_sent, self.spike_count)
            #print('non local event end. num:',self._lockout_count,'current',self.bin_timestamp_1,
            #    'last',self._last_lockout_timestamp,'lock time',self._lockout_time)
            self._lockout_count += 1
            #print('ripple lockout ended. time:',np.around(timestamp/30,decimals=2))
        elif not self._in_lockout:
            self.shortcut_message_sent = False

class MainSimulatorManager(rt_logging.LoggingClass):

    def __init__(self, rank, config, parent, send_interface, stim_decider):

        self.rank = rank
        self.config = config
        self.parent = parent
        self.send_interface = send_interface
        self.stim_decider = stim_decider

        self.time_sync_on = False

        self.rec_manager = binary_record.BinaryRecordsManager(manager_label='state',
                                                              save_dir=self.config['files']['output_dir'],
                                                              file_prefix=self.config['files']['prefix'],
                                                              file_postfix=self.config['files']['rec_postfix'])

        self.local_timing_file = \
            timing_system.TimingFileWriter(save_dir=self.config['files']['output_dir'],
                                           file_prefix=self.config['files']['prefix'],
                                           mpi_rank=self.rank,
                                           file_postfix=self.config['files']['timing_postfix'])

        self.ranks_sending_recs = list(range(self.parent.comm.Get_size()))
        self.ranks_sending_recs.remove(self.rank)
        self.ranks_sending_recs.remove(self.config["rank"]["gui"])
        self.set_up_ranks = []
        self.all_ranks_set_up = False

        # stim decider bypass the normal record registration message sending
        for message in stim_decider.get_record_register_messages():
            self.rec_manager.register_rec_type_message(message)

        self.master_time = MPI.Wtime()

        super().__init__()

    def synchronize_time(self):
        self.class_log.debug("Sending time sync messages to simulator node.")
        self.send_interface.send_time_sync_simulator()
        self.send_interface.all_barrier()
        self.master_time = MPI.Wtime()
        self.class_log.debug("Post barrier time set as master.")

    def send_calc_offset_time(self, rank, remote_time):
        offset_time = self.master_time - remote_time
        self.send_interface.send_time_sync_offset(rank, offset_time)

    # MEC edited this function to take in list of ripple tetrodes only
    def _ripple_ranks_startup(self, ripple_trode_list):
        for rip_rank in self.config['rank']['ripples']:
            self.send_interface.send_num_ntrode(
                rank=rip_rank, num_ntrodes=len(ripple_trode_list))

        # Round robin allocation of channels to ripple
        enable_count = 0
        all_ripple_process_enable = [[]
                                     for _ in self.config['rank']['ripples']]
        # MEC changed trode_liist to ripple_trode_list
        for chan_ind, chan_id in enumerate(ripple_trode_list):
            all_ripple_process_enable[enable_count % len(
                self.config['rank']['ripples'])].append(chan_id)
            enable_count += 1

        # Set channel assignments for all ripple ranks
        # MEC changed send_channel_selection to sned_ripple_channel_selection
        for rank_ind, rank in enumerate(self.config['rank']['ripples']):
            self.send_interface.send_ripple_channel_selection(
                rank, all_ripple_process_enable[rank_ind])

        for rip_rank in self.config['rank']['ripples']:

            # Map json RippleParameterMessage onto python object and then send
            rip_param_message = ripple_process.RippleParameterMessage(
                **self.config['ripple']['RippleParameterMessage'])
            self.send_interface.send_ripple_parameter(
                rank=rip_rank, param_message=rip_param_message)

            # Convert json string keys into int (ntrode_id) and send
            rip_mean_base_dict = dict(map(lambda x: (int(x[0]), x[1]),
                                          self.config['ripple']['CustomRippleBaselineMeanMessage'].items()))
            #print('ripple mean: ',rip_mean_base_dict)
            self.send_interface.send_ripple_baseline_mean(
                rank=rip_rank, mean_dict=rip_mean_base_dict)

            # Convert json string keys into int (ntrode_id) and send
            rip_std_base_dict = dict(map(lambda x: (int(x[0]), x[1]),
                                         self.config['ripple']['CustomRippleBaselineStdMessage'].items()))
            #print('ripple std: ',rip_std_base_dict)
            self.send_interface.send_ripple_baseline_std(
                rank=rip_rank, std_dict=rip_std_base_dict)

    def _stim_decider_startup(self):
        print('startup stim decider')
        # Convert JSON Ripple Parameter config into message
        rip_param_message = ripple_process.RippleParameterMessage(
            **self.config['ripple']['RippleParameterMessage'])

        # Update stim decider's ripple parameters
        self.stim_decider.update_n_threshold(rip_param_message.n_above_thresh)
        self.stim_decider.update_lockout_time(rip_param_message.lockout_time)
        self.stim_decider.update_conditioning_lockout_time(
            rip_param_message.ripple_conditioning_lockout_time)
        self.stim_decider.update_posterior_lockout_time(
            rip_param_message.posterior_lockout_time)

        if rip_param_message.enabled:
            self.stim_decider.enable()
        else:
            self.stim_decider.disable()

    def _encoder_rank_startup(self, trode_list):

        for enc_rank in self.config['rank']['encoders']:
            self.send_interface.send_num_ntrode(
                rank=enc_rank, num_ntrodes=len(trode_list))

        # Round robin allocation of channels to encoders
        enable_count = 0
        all_encoder_process_enable = [[]
                                      for _ in self.config['rank']['encoders']]
        for chan_ind, chan_id in enumerate(trode_list):
            all_encoder_process_enable[enable_count % len(
                self.config['rank']['encoders'])].append(chan_id)
            enable_count += 1

        print('finished round robin')
        # Set channel assignments for all encoder ranks
        for rank_ind, rank in enumerate(self.config['rank']['encoders']):
            print('rank', rank, 'encoder tet',
                  all_encoder_process_enable[rank_ind])
            self.send_interface.send_channel_selection(
                rank, all_encoder_process_enable[rank_ind])

    def _decoder_rank_startup(self, trode_list):
        # original: single decoder
        #rank = self.config['rank']['decoder']
        #self.send_interface.send_channel_selection(rank, trode_list)
        # for more than 1 decoder
        for dec_rank in self.config['rank']['decoder']:
            self.send_interface.send_channel_selection(dec_rank, trode_list)

    def _writer_startup(self):
        # Update binary_record file writers before starting datastream
        for rec_rank in self.config['rank_settings']['enable_rec']:
            if rec_rank is not self.rank:
                self.send_interface.send_new_writer_message(rank=rec_rank,
                                                            new_writer_message=self.rec_manager.new_writer_message())

                self.send_interface.send_start_rec_message(rank=rec_rank)

        # Update and start bin rec for StimDecider.  Registration is done through MPI but setting and starting
        # the writer must be done locally because StimDecider does not have a MPI command message receiver
        self.stim_decider.set_record_writer_from_message(
            self.rec_manager.new_writer_message())
        self.stim_decider.start_record_writing()

    def _turn_on_datastreams(self):
        # Then turn on data streaming to ripple ranks
        for rank in self.config['rank']['ripples']:
            self.send_interface.send_turn_on_datastreams(rank)

        # Turn on data streaming to encoder
        for rank in self.config['rank']['encoders']:
            self.send_interface.send_turn_on_datastreams(rank)

        # Turn on data streaming to decoder
        # original 1 decoder
        #self.send_interface.send_turn_on_datastreams(
        #    self.config['rank']['decoder'])
        # more than 1 decoder
        for rank in self.config['rank']['decoder']:
            self.send_interface.send_turn_on_datastreams(rank)        

        self.time_sync_on = True

    # MEC edited
    def handle_ntrode_list(self, trode_list):

        self.class_log.debug(
            "Received decoding ntrode list {:}.".format(trode_list))
        print('start handel ntrode list')

        # self._ripple_ranks_startup(trode_list)
        self._encoder_rank_startup(trode_list)
        self._decoder_rank_startup(trode_list)
        self._stim_decider_startup()

        # self._writer_startup()
        # self._turn_on_datastreams()

    # MEC added
    def handle_ripple_ntrode_list(self, ripple_trode_list):

        self.class_log.debug(
            "Received ripple ntrode list {:}.".format(ripple_trode_list))

        self._ripple_ranks_startup(ripple_trode_list)

        self._writer_startup()
        self._turn_on_datastreams()

    def start(self):
        self.class_log.debug("Starting up")
        self.handle_ntrode_list(self.config["trodes_network"]["decoding_tetrodes"])
        self.handle_ripple_ntrode_list(self.config["trodes_network"]["ripple_tetrodes"])

    def register_rec_type_message(self, message):
        self.rec_manager.register_rec_type_message(message)

    def trigger_termination(self):
        self.send_interface.terminate_all()
        self.parent.trigger_termination()

    def update_all_rank_setup_status(self, rank):
        self.set_up_ranks.append(rank)
        if sorted(self.set_up_ranks) == self.ranks_sending_recs:
            self.all_ranks_set_up = True
            self.class_log.debug(f"Received from {self.set_up_ranks}, expected {self.ranks_sending_recs}")


##########################################################################
# Process
##########################################################################
class MainProcess(realtime_base.RealtimeProcess):

    def __init__(self, comm: MPI.Comm, rank, config):

        self.comm = comm    # type: MPI.Comm
        self.rank = rank
        self.config = config

        super().__init__(comm=comm, rank=rank, config=config)

        # MEC added
        self.stim_decider_send_interface = StimDeciderMPISendInterface(
            comm=comm, rank=rank, config=config)

        self.stim_decider = StimDecider(rank=rank, config=config,
                                        send_interface=self.stim_decider_send_interface)

        # self.posterior_recv_interface = PosteriorSumRecvInterface(comm=comm, rank=rank, config=config,
        #                                                          stim_decider=self.stim_decider)

        # self.stim_decider = StimDecider(rank=rank, config=config,
        #                                send_interface=StimDeciderMPISendInterface(comm=comm,
        #                                                                           rank=rank,
        #                                                                           config=config))

        # self.data_recv = StimDeciderMPIRecvInterface(comm=comm, rank=rank, config=config,
        #                                             stim_decider=self.stim_decider)

        self.send_interface = MainMPISendInterface(
            comm=comm, rank=rank, config=config)

        self.manager = MainSimulatorManager(rank=rank, config=config, parent=self, send_interface=self.send_interface,
                                            stim_decider=self.stim_decider)
        print('======================================')
        print('In MainProcess: datasource = ', config['datasource'])
        print('======================================')
        if config['datasource'] == 'trodes':
            # print('about to configure trdoes network for tetrode: ',
            #       self.manager.handle_ntrode_list, self.rank)
            # time.sleep(5+1*self.rank)

            # self.networkclient = MainProcessClient(
            #     "SpykshrkMainProc", config['trodes_network']['address'], config['trodes_network']['port'], self.config)
            # if self.networkclient.initialize() != 0:
            #     print("Network could not successfully initialize")
            #     del self.networkclient
            #     quit()
            # # added MEC
            # self.networkclient.initializeHardwareConnection()
            # self.networkclient.registerStartupCallback(
            #     self.manager.handle_ntrode_list)
            # # added MEC
            # self.networkclient.registerStartupCallbackRippleTetrodes(
            #     self.manager.handle_ripple_ntrode_list)
            # self.networkclient.registerTerminationCallback(
            #     self.manager.trigger_termination)
            # print('completed trodes setup')

            #############################################################
            self.networkclient = MainProcessClient(config, self.manager)
            #############################################################

        self.vel_pos_recv_interface = VelocityPositionRecvInterface(comm=comm, rank=rank, config=config,
                                                                    stim_decider=self.stim_decider,
                                                                    networkclient=self.networkclient)

        self.posterior_recv_interface = PosteriorSumRecvInterface(comm=comm, rank=rank, config=config,
                                                                  stim_decider=self.stim_decider,
                                                                  networkclient=self.networkclient)

        self.data_recv = StimDeciderMPIRecvInterface(comm=comm, rank=rank, config=config,
                                                     stim_decider=self.stim_decider, networkclient=self.networkclient)

        self.recv_interface = MainSimulatorMPIRecvInterface(comm=comm, rank=rank,
                                                            config=config, main_manager=self.manager)

        self.gui_recv = MainGuiRecvInterface(comm, rank, config, self.stim_decider)

        self.mpi_status = MPI.Status()

        self.started = False

        # First Barrier to finish setting up nodes, waiting for Simulator to send ntrode list.
        # The main loop must be active to receive binary record registration messages, so the
        # first Barrier is placed here.
        self.class_log.debug("First Barrier")
        self.send_interface.all_barrier()
        self.class_log.debug("Past First Barrier")

    def trigger_termination(self):
        raise StopIteration()

    def main_loop(self):
        # self.thread.start()

        check_user_input = True

        # Synchronize rank times immediately
        last_time_bin = int(time.time())

        try:

            while True:

                    # Synchronize rank times
                    if self.manager.time_sync_on:
                        current_time_bin = int(time.time())
                        if current_time_bin >= last_time_bin + 10:
                            self.manager.synchronize_time()
                            last_time_bin = current_time_bin

                    self.recv_interface.__next__()
                    self.data_recv.__next__()
                    self.vel_pos_recv_interface.__next__()
                    self.posterior_recv_interface.__next__()
                    self.networkclient.__next__()
                    self.gui_recv.__next__()

                    if check_user_input and self.manager.all_ranks_set_up:
                        print("***************************************", flush=True)
                        print("   All ranks are set up, ok to start   ", flush=True)
                        print("***************************************", flush=True)
                        self.send_interface.send_setup_complete()
                        self.class_log.debug("Notified GUI that setup was complete")
                        check_user_input = False

        except StopIteration as ex:
            self.class_log.info(
                'Terminating MainProcess (rank: {:})'.format(self.rank))

        self.class_log.info("Main Process Main reached end, exiting.")
