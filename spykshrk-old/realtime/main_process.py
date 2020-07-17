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
from mpi4py import MPI
from spikegadgets import trodesnetwork as tnp

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


class MainProcessClient(tnp.AbstractModuleClient):
    def __init__(self, name, addr, port, config):
        super().__init__(name, addr, port)
        # self.main_manager = main_manager
        self.config = config
        self.started = False
        self.ntrode_list_sent = False
        self.terminated = False

    def registerTerminationCallback(self, callback):
        self.terminate = callback

    def registerStartupCallback(self, callback):
        self.startup = callback

    # MEC added: to get ripple tetrode list
    def registerStartupCallbackRippleTetrodes(self, callback):
        self.startupRipple = callback

    def recv_acquisition(self, command, timestamp):
        if command == tnp.acq_PLAY:
            if not self.ntrode_list_sent:
                self.startup(
                    self.config['trodes_network']['decoding_tetrodes'])
                # added MEC
                self.startupRipple(
                    self.config['trodes_network']['ripple_tetrodes'])
                self.started = True
                self.ntrode_list_sent = True

        if command == tnp.acq_STOP:
            if not self.terminated:
                # self.main_manager.trigger_termination()
                self.terminate()
                self.terminated = True
                self.started = False

    def recv_quit(self):
        self.terminate()


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
        print('===============================')
        print('In MainProcess: datasource = ', config['datasource'])
        print('===============================')
        if config['datasource'] == 'trodes':
            print('about to configure trdoes network for tetrode: ',
                  self.manager.handle_ntrode_list, self.rank)
            time.sleep(0.5 * self.rank)

            self.networkclient = MainProcessClient(
                "SpykshrkMainProc", config['trodes_network']['address'], config['trodes_network']['port'], self.config)
            if self.networkclient.initialize() != 0:
                print("Network could not successfully initialize")
                del self.networkclient
                quit()
            # added MEC
            self.networkclient.initializeHardwareConnection()
            self.networkclient.registerStartupCallback(
                self.manager.handle_ntrode_list)
            # added MEC
            self.networkclient.registerStartupCallbackRippleTetrodes(
                self.manager.handle_ripple_ntrode_list)
            self.networkclient.registerTerminationCallback(
                self.manager.trigger_termination)
            print('completed trodes setup')

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

        self.terminate = False

        self.mpi_status = MPI.Status()

        # First Barrier to finish setting up nodes, waiting for Simulator to send ntrode list.
        # The main loop must be active to receive binary record registration messages, so the
        # first Barrier is placed here.
        self.class_log.debug("First Barrier")
        self.send_interface.all_barrier()
        self.class_log.debug("Past First Barrier")

    def trigger_termination(self):
        self.terminate = True

    def main_loop(self):
        # self.thread.start()

        # Synchronize rank times immediately
        last_time_bin = int(time.time())

        while not self.terminate:

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

        self.class_log.info("Main Process Main reached end, exiting.")


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
                                  realtime_base.RecordIDs.STIM_MESSAGE],
                         rec_labels=[['timestamp', 'elec_grp_id', 'threshold_state'],
                                     ['timestamp', 'time', 'lockout_num', 'lockout_state', 'tets_above_thresh',
                                      'big_rip_message_sent'],
                                     ['bin_timestamp', 'spike_timestamp', 'lfp_timestamp', 'time',
                                      'shortcut_message_sent', 'ripple_number', 'ripple_time_bin', 'delay',
                                      'spike_count', 'posterior_max_arm', 'content_threshold', 'ripple_end',
                                      'max_arm_repeats', 'box', 'arm1', 'arm2', 'arm3', 'arm4', 'arm5', 'arm6', 'arm7', 'arm8']],
                         rec_formats=['Iii',
                                      'Idiiqi',
                                      'IIidiiidiidiiddddddddd'])
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
        self.no_ripple_time_bin = 0
        self.replay_target_arm = self.config['pp_decoder']['replay_target_arm']
        #self.posterior_arm_sum = np.zeros((1,9))
        self.posterior_arm_sum = np.zeros((9,))
        # initialize with single 1 so that first pass throught posterior_sum works
        self.norm_posterior_arm_sum = np.asarray([0, 1, 0, 0, 0, 0, 0, 0, 0])
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
        self.spike_timestamp = 0

        self.velocity = 0
        self.linearized_position = 0
        self.vel_pos_counter = 0
        self.thresh_counter = 0
        self.postsum_timing_counter = 0
        #self.stim_message_sent = 0
        self.big_rip_message_sent = 0
        self.arm_replay_counter = np.zeros((8,))
        self.posterior_time_bin = 0
        self.posterior_spike_count = 0
        self.posterior_arm_threshold = self.config['ripple_conditioning']['posterior_sum_threshold']
        # set max repeats allowed at each arm during content trials
        self.max_arm_repeats = 1
        # marker for stim_message to note end of ripple (message sent or end of lockout)
        self.ripple_end = 0

        # for continous running sum of posterior
        self.running_post_sum_counter = 0
        self.posterior_sum_array = np.zeros(
            (self.config['ripple_conditioning']['post_sum_sliding_window'], 9))
        self.sum_array_sum = np.zeros((9,))
        self.posterior_sum_timestamps = np.zeros(
            (self.config['ripple_conditioning']['post_sum_sliding_window'], 1))
        self.post_sum_sliding_window_actual = 0

        # used to pring out number of replays during session
        self.arm1_replay_counter = 0
        self.arm2_replay_counter = 0

        # if self.config['datasource'] == 'trodes':
        #    self.networkclient = MainProcessClient("SpykshrkMainProc", config['trodes_network']['address'],config['trodes_network']['port'], self.config)
        # self.networkclient.initializeHardwareConnection()
        time = MPI.Wtime()

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
        if self.thresh_counter % 100 == 0:
            self.record_timing(timestamp=timestamp, elec_grp_id=elec_grp_id,
                               datatype=datatypes.Datatypes.LFP, label='stim_rip_state')
        time = MPI.Wtime()

        # record timestamp from ripple node
        self.lfp_timestamp = timestamp

        #print('received thresh states: ',threshold_state,conditioning_thresh_state)

        if self._enabled:
            self.thresh_counter += 1

            # if self.thresh_counter % 1500 == 0 and self._in_lockout:
            #    #print('in normal lockout for ripple detection - one line per tetrode')
            #    pass

            # if self.thresh_counter % 1500 == 0 and self._conditioning_in_lockout:
            #    #print('in conditoning lockout for ripple detection - one line per tetrode')
            #    pass

            self._ripple_thresh_states.setdefault(elec_grp_id, 0)
            self._conditioning_ripple_thresh_states.setdefault(elec_grp_id, 0)

            # only write state if state changed
            if self._ripple_thresh_states[elec_grp_id] != threshold_state:
                self.write_record(realtime_base.RecordIDs.STIM_STATE,
                                  timestamp, elec_grp_id, threshold_state)

            # count number of tets above threshold for content ripple
            self._ripple_thresh_states[elec_grp_id] = threshold_state
            num_above = 0
            for state in self._ripple_thresh_states.values():
                num_above += state

            # count number of tets above threshold for large ripple
            self._conditioning_ripple_thresh_states[elec_grp_id] = conditioning_thresh_state
            conditioning_num_above = 0
            for conditioning_state in self._conditioning_ripple_thresh_states.values():
                conditioning_num_above += conditioning_state

            # end lockout for content ripples
            if self._in_lockout and (timestamp > self._last_lockout_timestamp + self._lockout_time):
                # End lockout
                self._in_lockout = False
                self.write_record(realtime_base.RecordIDs.STIM_LOCKOUT,
                                  timestamp, time, self._lockout_count, self._in_lockout,
                                  num_above, self.big_rip_message_sent)
                self._lockout_count += 1
                #print('ripple lockout ended. time:',np.around(timestamp/30,decimals=2))

            # end lockout for posterior sum
            if self._posterior_in_lockout and (timestamp > self._posterior_last_lockout_timestamp +
                                               self._posterior_lockout_time):
                # End lockout
                self._posterior_in_lockout = False
                self.write_record(realtime_base.RecordIDs.STIM_LOCKOUT,
                                  timestamp, time, self._lockout_count, self._posterior_in_lockout,
                                  num_above, self.big_rip_message_sent)
                #self._lockout_count += 1
                #print('posterior sum lockout ended. time:',np.around(timestamp/30,decimals=2))

            # end lockout for large ripples
            # note: currently only one variable for counting both lockouts
            if self._conditioning_in_lockout and (timestamp > self._conditioning_last_lockout_timestamp +
                                                  self._conditioning_lockout_time):
                # End lockout
                self._conditioning_in_lockout = False
                self.write_record(realtime_base.RecordIDs.STIM_LOCKOUT,
                                  timestamp, time, self._lockout_count, self._conditioning_in_lockout,
                                  num_above, self.big_rip_message_sent)
                self._lockout_count += 1
                print('end of lockout big ripple')

            # end lockout for running posterior sum at end of ripple
            # lockout hard coded for 10 msec (300 timestamps)
            if self._ripple_end_in_lockout and (timestamp > self._ripple_end_last_lockout_timestamp +
                                                self._ripple_end_lockout_time):
                # End lockout
                self._ripple_end_in_lockout = False
                # self.write_record(realtime_base.RecordIDs.STIM_LOCKOUT,
                #                  timestamp, time, self._lockout_count, self._in_lockout,
                #                  num_above, self.big_rip_message_sent)
                #self._lockout_count += 1
                print('end ripple lockout ended. time:',
                      np.around(timestamp / 30, decimals=2))

            # detection of large ripples: 2 tets above rip thresh, velocity below vel thresh, not in lockout (125 msec after previous rip)
            # ideally this would also take in some output from statescript that says whether this time is ripple or content conditioning
            # because this statement is before posterior sum, posterior sum will not run if the two ripple thresholds are the same
            if (conditioning_num_above >= self._ripple_n_above_thresh) and self.velocity < self.config['ripple_conditioning']['ripple_detect_velocity'] and not self._conditioning_in_lockout:
                self.big_rip_message_sent = 0
                print('tets above cond ripple thresh: ', conditioning_num_above, timestamp,
                      self._conditioning_ripple_thresh_states, np.around(self.velocity, decimals=2))
                #print('lockout time: ',self._lockout_time)
                # this will flash light every time a ripple is detected
                #networkclient.sendMsgToModule('StateScript', 'StatescriptCommand', 's', ['trigger(16);\n'])

                # this is the easiest place to send a statescript message for ripple conditioning
                # this will send a message for 1st threshold crossing - no time delay
                # need to re-introduce lockout so that only one message is sent per ripple (7500 = 5 sec)
                # lockout is in timestamps - so 1 seconds = 30000
                # for a dynamic filter, we need to get ripple size from the threshold message and send to statescript
                #print('sent behavior message based on ripple thresh',time,timestamp)
                #networkclient.sendMsgToModule('StateScript', 'StatescriptCommand', 's', ['trigger(7);\n'])

                # for trodes okay to send variable and function in one command, cant send two functions in one
                # command and cant send two function in two back-to-back commands - gives compile error
                #networkclient.sendMsgToModule('StateScript', 'StatescriptCommand', 's', ['replay_arm = 1;\ntrigger(15);\n'])

                # for trodes, this is the syntax for a shortcut message - but does not work currently
                # number is function number
                # networkclient.sendStateScriptShortcutMessage(16)
                #print('sent shortcut message')

                # this starts the lockout for large ripple threshold
                self._conditioning_in_lockout = True
                self._conditioning_last_lockout_timestamp = timestamp
                self.big_rip_message_sent = 1

                # MEC this will get rid of lockout
                #self._in_lockout = True
                #self._last_lockout_timestamp = timestamp

                #self.class_log.debug("Ripple threshold detected {}.".format(self._ripple_thresh_states))

                self.write_record(realtime_base.RecordIDs.STIM_LOCKOUT,
                                  timestamp, self.velocity, self._lockout_count, self._conditioning_in_lockout,
                                  conditioning_num_above, self.big_rip_message_sent)

                # dan's way of sending stim to trodes - not using this.
                # self._send_interface.start_stimulation()

            # detection of content ripples: 2 tets above rip thresh, velocity below vel thresh, not in lockout (500 msec after previous rip)
            elif (num_above >= self._ripple_n_above_thresh) and self.velocity < self.config['ripple_conditioning']['ripple_detect_velocity'] and not self._in_lockout:
                # add to the ripple count- no should use lockout_count
                #self.ripple_number += 1
                # this needs to just turn on light
                # i think this needs to use lockout too, otherwise too many messages
                # but this could interfere with detection of larger ripples
                # i think we need no lockout here because it will mask large ripples
                # so need to come up with a better way to trigger light - turn off for now
                print('detection of ripple for content, lfp timestamp',
                      timestamp, np.around(timestamp / 30, decimals=2))
                print('arm1 replays:', self.arm1_replay_counter,
                      'arm2 replays:', self.arm2_replay_counter)
                #print('sent light message based on ripple thresh',time,timestamp)
                #networkclient.sendMsgToModule('StateScript', 'StatescriptCommand', 's', ['trigger(15);\n'])

                # this starts the lockout for the content ripple threshold and tells us there is a ripple
                # start posterior lockout here too, it is shorter, so it should have 25 msec delay between ripples
                self._in_lockout = True
                self._posterior_in_lockout = True
                self._last_lockout_timestamp = timestamp
                #print('last lockout timestamp',self._last_lockout_timestamp)
                self._posterior_last_lockout_timestamp = timestamp
                # this should allow us to tell difference between ripples that trigger conditioning
                self.big_rip_message_sent = 0

                self.write_record(realtime_base.RecordIDs.STIM_LOCKOUT,
                                  timestamp, self.velocity, self._lockout_count, self._in_lockout,
                                  num_above, self.big_rip_message_sent)

            # this is where sending the statescript message should be triggered if lockout is over
            # this should force message to be sent on time, without a delay from empty decoder bins
            # we may also need to incorperate a lower limit for spike_count
            # set trigger for when lfp timestamp is close to end of posterior lockout time
            # may need another lockout period of 10 msec here to prevent this line for triggering many times
            # LFP_timestamp is updated every 20 timestamps
            # start a 10 msec lockout period so that this function only runs once
            # we might need to make this lockout much longer and loop it in to posterior_sum so that posterior sum
            # doesnt send a second message once the empty bins finally arrive
            # with abs < 60, this will trigger 2msec before the end of posterior_lockout
            # so this will always happen before the next posterior message (i think)
            # this might no send some messages because it happens before posterior lockout is over so it wont
            # trigger sending message for ripples where time runs about before one arm passes threshold...
            # NOTE: correct, this currnetly doesnt do anything!
            # if not self._ripple_end_in_lockout and abs((self._posterior_last_lockout_timestamp+self._posterior_lockout_time) - timestamp) < 60:
            #     # force sending of posterior sum statescript message
            #     # run posterior_sum with current values
            #     #print('each time you see this posterior sum will run. time diff:',
            #     #      abs((self._posterior_last_lockout_timestamp+self._posterior_lockout_time) - timestamp))
            #     print('at ripple end, forced posterior_sum to send message.')

            #     self.posterior_sum(bin_timestamp=self.bin_timestamp,spike_timestamp=self.spike_timestamp,
            #                   box=self.box_post,arm1=self.arm1_post,
            #                   arm2=self.arm2_post,arm3=self.arm3_post,arm4=self.arm4_post,
            #                   arm5=self.arm5_post,arm6=self.arm6_post,arm7=self.arm7_post,arm8=self.arm8_post,
            #                   spike_count=self.spike_count,networkclient=networkclient)

            #     self._ripple_end_in_lockout = True
            #     self._ripple_end_last_lockout_timestamp = timestamp

            return num_above

    # this function handles sending statescript message at end of replay
    def posterior_sum_statescript_message(self, arm, networkclient):
        arm = arm
        networkclient = networkclient
        time = MPI.Wtime()

        print('max posterior in arm:', arm, np.around(self.norm_posterior_arm_sum[arm], decimals=2),
              'posterior sum:', np.around(
                  self.norm_posterior_arm_sum.sum(), decimals=2),
              'position:', np.around(self.linearized_position, decimals=2),
              'posterior bins in ripple:', self.posterior_time_bin, 'ending bin timestamp:', self.bin_timestamp,
              'lfp timestamp:', self.lfp_timestamp, 'delay:', np.around(
                  (self.lfp_timestamp - self.bin_timestamp) / 30, decimals=1),
              'spike count:', self.posterior_spike_count, 'sliding window:', self.post_sum_sliding_window_actual)
        #self.shortcut_message_arm = np.argwhere(self.norm_posterior_arm_sum>self.posterior_arm_threshold)[0][0]
        self.shortcut_message_arm = arm

        # only send message for arm 1 replay if it was not last rewarded arm
        if self.arm_replay_counter[arm - 1] < self.max_arm_repeats:
            # NOTE: we can now replace this with the actual shortcut message!
            # for shortcut, each arm is assigned a different message

            # note: statescript can only execute one function at a time, so trigger function 15 and set replay_arm variable
            statescript_command = f'replay_arm = {arm};\ntrigger(15);\n'
            #print('string for statescript:',statescript_command)
            networkclient.sendMsgToModule(
                'StateScript', 'StatescriptCommand', 's', [statescript_command])
            #networkclient.sendMsgToModule('StateScript', 'StatescriptCommand', 's', ['replay_arm = 1;\ntrigger(15);\n'])
            print('sent StateScript message for arm', arm,
                  'replay in ripple', self._lockout_count)

            # arm replay counters, only active at wait well and adds to current counter and sets other arms to 0
            print('arm replay count:', self.arm_replay_counter)
            self.shortcut_message_sent = True

            self.ripple_end = 1
            self.write_record(realtime_base.RecordIDs.STIM_MESSAGE,
                              self.bin_timestamp, self.spike_timestamp, self.lfp_timestamp, time, self.shortcut_message_sent,
                              self._lockout_count, self.posterior_time_bin,
                              (self.lfp_timestamp - self.bin_timestamp) / 30,
                              self.posterior_spike_count,
                              self.shortcut_message_arm, self.posterior_arm_threshold, self.ripple_end, self.max_arm_repeats,
                              self.norm_posterior_arm_sum[0], self.norm_posterior_arm_sum[1], self.norm_posterior_arm_sum[2],
                              self.norm_posterior_arm_sum[3], self.norm_posterior_arm_sum[4], self.norm_posterior_arm_sum[5],
                              self.norm_posterior_arm_sum[6], self.norm_posterior_arm_sum[7], self.norm_posterior_arm_sum[8])
        else:
            print('more than ', self.max_arm_repeats,
                  ' replays of arm', arm, 'in a row!')

    # MEC: this function brings in velocity and linearized position from decoder process

    def velocity_position(self, bin_timestamp, vel, pos):
        self.velocity = vel
        self.linearized_position = pos
        self.vel_pos_counter += 1

        if self.velocity < self.config['ripple_conditioning']['ripple_detect_velocity']:
            #print('immobile, vel = ',self.velocity)
            pass
        if self.linearized_position >= 3 and self.linearized_position <= 5:
            #print('position at rip/wait!')
            pass

    # MEC: this function sums the posterior during each ripple, then sends shortcut message
    # need to add location filter so it only sends message when rat is at rip/wait well - no, that is in statescript
    def posterior_sum(self, bin_timestamp, spike_timestamp, box, arm1, arm2, arm3, arm4, arm5, arm6, arm7, arm8, spike_count, networkclient):
        time = MPI.Wtime()
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

        # reset posterior arm threshold (e.g. 0.5) based on the new_ripple_threshold text file
        # this should run every 10 sec, using thresh_counter which refers to each message from ripple node
        # 5 nodes so 5*1500 = 7500 per second
        # pos_vel counter seems to work better - this is still way faster, not sure how often it gets sent...
        # ah okay, the problem is that this is within posterior sum which only happens sporatically
        # so matching will happen less often - 5x , 10x - not sure???
        # this will depend on the number of spikes, so that is not ideal
        # should probably use a lockout instead...
        if self.thresh_counter % 10000 == 0:
            # if self.vel_pos_counter % 1000 == 0:
            #print('thresh_counter: ',self.thresh_counter)
            with open('config/new_ripple_threshold.txt') as posterior_threshold_file:
                fd = posterior_threshold_file.fileno()
                fcntl.fcntl(fd, fcntl.F_SETFL, os.O_NONBLOCK)
                # read file
                for post_thresh_file_line in posterior_threshold_file:
                    pass
                new_posterior_threshold = post_thresh_file_line
            # this allows half SD increase in ripple threshold (looks for three digits, eg 065 = 6.5 SD)
            # final 2 characters in line are new arm posterior threshold (eg 08 > 0.8)
            self.posterior_arm_threshold = np.int(
                new_posterior_threshold[8:10]) / 10
            print('posterior arm threshold = ', self.posterior_arm_threshold)

        # read arm_reward text file written by trodes to find last rewarded arm
        # use this to prevent repeated rewards to a specific arm (set arm1_replay_counter)
        if self.thresh_counter % 10000 == 0:
            # reset counters each time you read the file - b/c file might not change
            self.arm_replay_counter = np.zeros((8,))

            with open('config/rewarded_arm_trodes.txt') as rewarded_arm_file:
                fd = rewarded_arm_file.fileno()
                fcntl.fcntl(fd, fcntl.F_SETFL, os.O_NONBLOCK)
                # read file
                for rewarded_arm_file_line in rewarded_arm_file:
                    pass
                rewarded_arm = rewarded_arm_file_line
            rewarded_arm = np.int(rewarded_arm[0:2])
            print('last rewarded arm = ', rewarded_arm)
            if rewarded_arm > 0:
                self.arm_replay_counter[rewarded_arm - 1] = 1

        # check posterior lockout and normal lockout with print statement - seems to work
        if not self._posterior_in_lockout and self._in_lockout:
            #print('inside posterior sum delay time, bin timestamp:',bin_timestamp/30)
            pass

        # ***ACTIVE***
        # sliding window sum of posterior at all times

        self.running_post_sum_counter += 1

        if self.running_post_sum_counter % 10000 == 0:
            print('running sum of posterior', self.running_post_sum_counter)

        # timer if needed
        # if self.running_post_sum_counter % 1 == 0:
        #    self.record_timing(timestamp=spike_timestamp, elec_grp_id=0,
        #                       datatype=datatypes.Datatypes.LFP, label='postsum_1')

        # new variables for each arm posterior - for function to send statescript message at end of ripple lockout
        new_posterior_sum = np.asarray([self.box_post, self.arm1_post, self.arm2_post, self.arm3_post,
                                        self.arm4_post, self.arm5_post, self.arm6_post, self.arm7_post,
                                        self.arm8_post])

        # this is running sum of whole epoch - dont want this
        #self.posterior_arm_sum = self.posterior_arm_sum + new_posterior_sum
        #self.norm_posterior_arm_sum = self.posterior_arm_sum/self.running_post_sum_counter

        # this generates posterior sum with a sliding window - # of decoder bins defined in config
        self.posterior_sum_array[np.mod(self.running_post_sum_counter,
                                        self.config['ripple_conditioning']['post_sum_sliding_window']), :] = new_posterior_sum
        self.sum_array_sum = np.sum(self.posterior_sum_array, axis=0)
        self.norm_posterior_arm_sum = self.sum_array_sum / \
            self.config['ripple_conditioning']['post_sum_sliding_window']

        # keep track of decoder bin timestamp for posteriors in the sliding sum - check out many msec is total diff
        self.posterior_sum_timestamps[np.mod(self.running_post_sum_counter,
                                             self.config['ripple_conditioning']['post_sum_sliding_window']), :] = self.bin_timestamp
        self.post_sum_sliding_window_actual = np.ptp(
            self.posterior_sum_timestamps) / 30

        # start of a ripple: check posterior sum and then send message
        if (self._posterior_in_lockout
            and self.velocity < self.config['ripple_conditioning']['ripple_detect_velocity']
                and not self.shortcut_message_sent):
            self.posterior_time_bin += 1
            self.shortcut_message_arm = 99
            self.ripple_end = 0
            self.posterior_spike_count = self.posterior_spike_count + spike_count

            if self.posterior_time_bin == 1:
                # print('start of ripple. sum is ',np.around(self.norm_posterior_arm_sum,decimals=1),
                #      np.around(self.sum_array_sum,decimals=1),np.around(new_posterior_sum,decimals=1))
                print('start of ripple. bins:', self.config['ripple_conditioning']['post_sum_sliding_window'],
                      'sum', np.around(self.norm_posterior_arm_sum, decimals=1))

            # check if current posterior sum is above 0.5 in any segment
            detected_region = np.argwhere(
                self.norm_posterior_arm_sum > self.posterior_arm_threshold)
            if len(detected_region == 1):
                detected_region = detected_region[0][0]
                # replay detection of box - only send message for arm replays
                # save post sum every time bin - turned off 2-22
                if detected_region == 0:
                    if self.posterior_time_bin == 1:
                        print('replay in box - no StateScript message. ripple', self._lockout_count,
                              'sliding window', self.post_sum_sliding_window_actual)
                        self.shortcut_message_arm = 0
                        self.write_record(realtime_base.RecordIDs.STIM_MESSAGE,
                                          bin_timestamp, spike_timestamp, self.lfp_timestamp, time, self.shortcut_message_sent,
                                          self._lockout_count, self.posterior_time_bin,
                                          (self.lfp_timestamp -
                                           self.bin_timestamp) / 30,
                                          self.posterior_spike_count,
                                          self.shortcut_message_arm, self.posterior_arm_threshold, self.ripple_end, self.max_arm_repeats,
                                          self.norm_posterior_arm_sum[0], self.norm_posterior_arm_sum[
                                              1], self.norm_posterior_arm_sum[2],
                                          self.norm_posterior_arm_sum[3], self.norm_posterior_arm_sum[
                                              4], self.norm_posterior_arm_sum[5],
                                          self.norm_posterior_arm_sum[6], self.norm_posterior_arm_sum[7], self.norm_posterior_arm_sum[8])
                else:
                    print('arm', detected_region)
                    self.posterior_sum_statescript_message(
                        detected_region, networkclient)

            # no maximum location for whole ripple
            # save post sum every time bin
            else:
                if self.posterior_time_bin == 1:
                    print(
                        'no segment above 0.5 - no StateScript message. ripple', self._lockout_count)
                    self.write_record(realtime_base.RecordIDs.STIM_MESSAGE,
                                      bin_timestamp, spike_timestamp, self.lfp_timestamp, time, self.shortcut_message_sent,
                                      self._lockout_count, self.posterior_time_bin,
                                      (self.lfp_timestamp -
                                       self.bin_timestamp) / 30,
                                      self.posterior_spike_count,
                                      self.shortcut_message_arm, self.posterior_arm_threshold, self.ripple_end, self.max_arm_repeats,
                                      self.norm_posterior_arm_sum[0], self.norm_posterior_arm_sum[
                                          1], self.norm_posterior_arm_sum[2],
                                      self.norm_posterior_arm_sum[3], self.norm_posterior_arm_sum[
                                          4], self.norm_posterior_arm_sum[5],
                                      self.norm_posterior_arm_sum[6], self.norm_posterior_arm_sum[7], self.norm_posterior_arm_sum[8])

        elif not self._posterior_in_lockout:
            self.shortcut_message_sent = False
            self.posterior_time_bin = 0
            self.posterior_spike_count = 0

        # timer if needed
        # if self.running_post_sum_counter % 1 == 0:
        #    self.record_timing(timestamp=spike_timestamp, elec_grp_id=0,
        #                       datatype=datatypes.Datatypes.LFP, label='postsum_2')

        # *** NOT ACTIVE - set in config ***
        # running sum of posterior during a ripple
        # marker for ripple detection is: self._in_lockout NO it's self._posterior_in_lockout
        # marker for already reached sum above threshold: self.shortcut_message_sent
        if (self.config['ripple_conditioning']['posterior_sum_rip_only']
            and self._posterior_in_lockout
            and self.velocity < self.config['ripple_conditioning']['ripple_detect_velocity']
                and not self.shortcut_message_sent):

            # if self.ripple_time_bin == 0:
            #    self.ripple_number += 1
            #    #print('ripple number: ',self.ripple_number)
            self.posterior_time_bin += 1
            self.no_ripple_time_bin = 0
            #self.ripple_time_bin += 1
            self.postsum_timing_counter += 1

            # comfirm that posterior threshold has been updated and print that we are in posterio sum function
            if self.posterior_time_bin == 1:
                print('in posterior sum function, threshold = ', self.posterior_arm_threshold,
                      'starting bin timestamp', bin_timestamp, 'starting LFP timestamp', self.lfp_timestamp)

            if self.postsum_timing_counter % 1 == 0:
                self.record_timing(timestamp=spike_timestamp, elec_grp_id=0,
                                   datatype=datatypes.Datatypes.LFP, label='postsum_in')

            # while the ripple is progressing we need to add the current posterior sum to the sum of all earlier ones
            #new_posterior_sum = np.asarray([box,arm1,arm2,arm3,arm4,arm5,arm6,arm7,arm8])

            # new variables for each arm posterior - for function to send statescript message at end of ripple lockout
            new_posterior_sum = np.asarray([self.box_post, self.arm1_post, self.arm2_post, self.arm3_post,
                                            self.arm4_post, self.arm5_post, self.arm6_post, self.arm7_post,
                                            self.arm8_post])

            # print statement to check sum is working
            # print('new posterior, bin number:',self.posterior_time_bin,np.around(new_posterior_sum,decimals=2),
            #      'decoder timestamp',bin_timestamp,'lfp timestamp',self.lfp_timestamp)
            self.posterior_arm_sum = self.posterior_arm_sum + new_posterior_sum

            # also add up spikes in the replay
            self.posterior_spike_count = self.posterior_spike_count + spike_count

            # MEC: 10-27-19: try turning off stim_message record to see if that helps record saving problem
            self.ripple_end = 0
            # this save the posterior arm sum every time bin
            self.write_record(realtime_base.RecordIDs.STIM_MESSAGE,
                              bin_timestamp, spike_timestamp, self.lfp_timestamp, time, self.shortcut_message_sent,
                              self.ripple_number, self.posterior_time_bin, self.posterior_spike_count,
                              self.shortcut_message_arm, self.posterior_arm_threshold, self.ripple_end, self.max_arm_repeats,
                              new_posterior_sum[0], new_posterior_sum[1], new_posterior_sum[2],
                              new_posterior_sum[3], new_posterior_sum[4], new_posterior_sum[5],
                              new_posterior_sum[6], new_posterior_sum[7], new_posterior_sum[8])

            # normalize posterior_arm_sum - make new variable so its doesnt keep getting divided
            # 12-15-19 normalized posterior often doesnt add to 1 - fixed typo in message from decoder
            # self.posterior_time_bin should increase each time a new posterior message comes in during a ripple
            self.norm_posterior_arm_sum = self.posterior_arm_sum / self.posterior_time_bin

            # send a statescript message if posterior is above threshold in one arm and it has been about 50 msec
            # for delay use 10 posterior_time_bins, about 50 msec
            # could also use simliar calcuation for end of lockout
            # includes velocity filter now

            # return arm with max posterior: first check if only 1 arm above self.posterior_arm_threshold,
            # make that arm a variable, then check if each arm is the max, and if so, send statescript message
            # shortcut message fn_num is an interger for function number in statescript
            # alternative: only require that any outer arm is above 0.3 - then put reward in that arm

            # should this be len(argwhere) > 0? does this mean this already?
            # NOTE: trigger to send statescript msg if lfp_timestamp > posterior_lockout is above
            # NOTE: the trigger is NOT working
            # 1-16-20: change so that only outer arm replay sends message, not box replay
            detected_region = np.argwhere(
                self.norm_posterior_arm_sum > self.posterior_arm_threshold)
            if len(detected_region == 1) and self.posterior_time_bin >= 10:
                detected_region = detected_region[0][0]
                # replay detection of box - dont use this now, only send message for arm replays
                if detected_region == 0:
                    pass
                    #print('replay in box - no StateScript message.')
                    # test functionalized posterior sum
                    # self.posterior_sum_statescript_message(2,networkclient)
                    # print('max posterior in box',np.around(self.norm_posterior_arm_sum[0],decimals=2),
                    #       'posterior sum: ',np.around(self.norm_posterior_arm_sum.sum(),decimals=2),
                    #       'position ',np.around(self.linearized_position,decimals=2),'ripple ',self.ripple_number,
                    #       'posterior bins in ripple ',self.posterior_time_bin,'ending bin timestamp',bin_timestamp,
                    #       'lfp timestamp',self.lfp_timestamp,'delay',(self.lfp_timestamp-bin_timestamp)/30,
                    #       'spike count',self.posterior_spike_count)
                    # self.shortcut_message_arm = detected_region[0][0]
                    # # For testing: while bill is in sleep box, this seems to be triggered most frequently
                    # #networkclient.sendMsgToModule('StateScript', 'StatescriptCommand', 's', ['replay_arm = 2;\ntrigger(15);\n'])
                    # #print('arm counters: ',self.arm1_replay_counter,self.arm2_replay_counter,
                    # #      self.arm3_replay_counter,self.arm4_replay_counter)
                    # self.shortcut_message_sent = True

                    # self.ripple_end = 1
                    # self.write_record(realtime_base.RecordIDs.STIM_MESSAGE,
                    #                   bin_timestamp, spike_timestamp, self.lfp_timestamp, time, self.shortcut_message_sent,
                    #                   self.ripple_number, self.posterior_time_bin, self.shortcut_message_arm,
                    #                   self.posterior_arm_threshold,self.ripple_end,self.max_arm_repeats,
                    #                   self.norm_posterior_arm_sum[0],self.norm_posterior_arm_sum[1],self.norm_posterior_arm_sum[2],
                    #                   self.norm_posterior_arm_sum[3],self.norm_posterior_arm_sum[4],self.norm_posterior_arm_sum[5],
                    #                   self.norm_posterior_arm_sum[6],self.norm_posterior_arm_sum[7],self.norm_posterior_arm_sum[8])

                else:
                    self.posterior_sum_statescript_message(
                        detected_region, networkclient)

        # if end of ripple (time bin) and no arm posterior crossed threshold (message sent) or box replay
        # these records are indicated by: ripple_end = 1 and shortcut_message_sent = 0
        # say in printout whether ripple ended because of not enough bins or posterior sum below threhsold
        # variable shortcue_message_arm: 99 if <10 time bins or no arm above threshold, otherwise repeated arm
        elif (self.config['ripple_conditioning']['posterior_sum_rip_only']
              and self.no_ripple_time_bin == 1
              and not self.shortcut_message_sent):
            if self.posterior_time_bin < 10:
                print('ripple ended before 10 time bins', ' ', np.around(self.norm_posterior_arm_sum, decimals=2),
                      'ripple: ', self.ripple_number, 'posterior sum: ', np.around(
                          self.norm_posterior_arm_sum.sum(), decimals=2),
                      'position ', np.around(
                          self.linearized_position, decimals=2),
                      'posterior bins in ripple ', self.posterior_time_bin, 'ending bin timestamp', bin_timestamp,
                      'lfp timestamp', self.lfp_timestamp, 'delay', (
                          self.lfp_timestamp - bin_timestamp) / 30,
                      'spike count', self.posterior_spike_count)
                self.shortcut_message_arm = 99

            else:
                print('repeated reward replay or no arm posterior above ', self.posterior_arm_threshold, ' ', np.around(self.norm_posterior_arm_sum, decimals=2),
                      'ripple: ', self.ripple_number, 'posterior sum: ', np.around(
                          self.norm_posterior_arm_sum.sum(), decimals=2),
                      'position ', np.around(
                          self.linearized_position, decimals=2),
                      'posterior bins in ripple ', self.posterior_time_bin, 'ending bin timestamp', bin_timestamp,
                      'lfp timestamp', self.lfp_timestamp, 'delay', (
                          self.lfp_timestamp - bin_timestamp) / 30,
                      'spike count', self.posterior_spike_count)
                detected_region = np.argwhere(
                    self.norm_posterior_arm_sum > self.posterior_arm_threshold)
                if len(detected_region == 1) == 0:
                    self.shortcut_message_arm = 99

            # can use this statescript message for testing
            # networkclient.sendMsgToModule('StateScript', 'StatescriptCommand', 's', ['replay_arm = 3;\ntrigger(15);\n'])
            self.shortcut_message_sent = False
            self.no_ripple_time_bin += 1

            self.ripple_end = 1
            self.write_record(realtime_base.RecordIDs.STIM_MESSAGE,
                              bin_timestamp, spike_timestamp, self.lfp_timestamp, time, self.shortcut_message_sent,
                              self.ripple_number, self.posterior_time_bin,  self.posterior_spike_count,
                              self.shortcut_message_arm, self.posterior_arm_threshold, self.ripple_end, self.max_arm_repeats,
                              self.norm_posterior_arm_sum[0], self.norm_posterior_arm_sum[1], self.norm_posterior_arm_sum[2],
                              self.norm_posterior_arm_sum[3], self.norm_posterior_arm_sum[4], self.norm_posterior_arm_sum[5],
                              self.norm_posterior_arm_sum[6], self.norm_posterior_arm_sum[7], self.norm_posterior_arm_sum[8])

            # end of posterior lockout signals end of ripple
        elif self.config['ripple_conditioning']['posterior_sum_rip_only'] and not self._posterior_in_lockout:
            self.no_ripple_time_bin += 1

            if self.no_ripple_time_bin > 2:
                self.posterior_time_bin = 0
                self.posterior_spike_count = 0
                self.posterior_arm_sum = np.zeros((9,))
                self.norm_posterior_arm_sum = np.zeros((9,))
                self.shortcut_message_arm = 99
                self.shortcut_message_sent = False


class StimDeciderMPIRecvInterface(realtime_base.RealtimeMPIClass):
    def __init__(self, comm: MPI.Comm, rank, config, stim_decider: StimDecider, networkclient):
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


class PosteriorSumRecvInterface(realtime_base.RealtimeMPIClass):
    def __init__(self, comm: MPI.Comm, rank, config, stim_decider: StimDecider, networkclient):
        super(PosteriorSumRecvInterface, self).__init__(
            comm=comm, rank=rank, config=config)

        self.stim = stim_decider
        self.networkclient = networkclient
        # NOTE: if you dont know how large the buffer should be, set it to a large number
        # then you will get an error saying what it should be set to
        # bytearray was 80 before adding spike_count
        self.msg_buffer = bytearray(84)
        self.req = self.comm.Irecv(
            buf=self.msg_buffer, tag=realtime_base.MPIMessageTag.POSTERIOR.value)

    def __next__(self):
        rdy = self.req.Test()
        time = MPI.Wtime()
        if rdy:

            message = decoder_process.PosteriorSum.unpack(self.msg_buffer)
            self.req = self.comm.Irecv(
                buf=self.msg_buffer, tag=realtime_base.MPIMessageTag.POSTERIOR.value)

            # need to activate record_timing in this class if we want to use this here
            # self.record_timing(timestamp=timestamp, elec_grp_id=elec_grp_id,
            #                   datatype=datatypes.Datatypes.SPIKES, label='post_sum_recv')

            # okay so we are receiving the message! but now it needs to get into the stim decider
            self.stim.posterior_sum(bin_timestamp=message.bin_timestamp, spike_timestamp=message.spike_timestamp,
                                    box=message.box, arm1=message.arm1,
                                    arm2=message.arm2, arm3=message.arm3, arm4=message.arm4, arm5=message.arm5,
                                    arm6=message.arm6, arm7=message.arm7, arm8=message.arm8,
                                    spike_count=message.spike_count, networkclient=self.networkclient)
            #print('posterior sum message supervisor: ',message.spike_timestamp,time*1000)
            # return posterior_sum

        # can we put a condtional statement here to run posterior_sum every 8 LFP measurements (~5 msec)
        # we bring in stim_decider to this class as stim so stim.thresh_counter should work
        # will this work if there is no message??
        # this doesnt seem to work because it gets triggered 10-20 times for each matching timestamp
        # elif not rdy and self.stim.thresh_counter % 16 == 0 and self.stim.postsum_timing_counter > 0 and self.stim._posterior_in_lockout:
            # self.stim.posterior_sum(bin_timestamp=message.bin_timestamp,spike_timestamp=message.spike_timestamp,
            #                        box=message.box,arm1=message.arm1,
            #                        arm2=message.arm2,arm3=message.arm3,arm4=message.arm4,arm5=message.arm5,
            #                        arm6=message.arm6,arm7=message.arm7,arm8=message.arm8,
            #                        spike_count=0,networkclient=self.networkclient)
            #print('no spike posterior',self.stim.thresh_counter)
            # with this version of running posterior_sum we want spike_count = 0, so we can count spikes accurately

        else:
            return None


class VelocityPositionRecvInterface(realtime_base.RealtimeMPIClass):
    def __init__(self, comm: MPI.Comm, rank, config, stim_decider: StimDecider, networkclient):
        super(VelocityPositionRecvInterface, self).__init__(
            comm=comm, rank=rank, config=config)

        self.stim = stim_decider
        self.networkclient = networkclient
        # NOTE: if you dont know how large the buffer should be, set it to a large number
        # then you will get an error saying what it should be set to
        self.msg_buffer = bytearray(16)
        self.req = self.comm.Irecv(
            buf=self.msg_buffer, tag=realtime_base.MPIMessageTag.VEL_POS.value)

    def __next__(self):
        rdy = self.req.Test()
        time = MPI.Wtime()
        if rdy:

            message = decoder_process.VelocityPosition.unpack(self.msg_buffer)
            self.req = self.comm.Irecv(
                buf=self.msg_buffer, tag=realtime_base.MPIMessageTag.VEL_POS.value)

            # okay so we are receiving the message! but now it needs to get into the stim decider
            self.stim.velocity_position(
                bin_timestamp=message.bin_timestamp, pos=message.pos, vel=message.vel)
            #print('posterior sum message supervisor: ',message.timestamp,time*1000)
            # return posterior_sum

        else:
            return None


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


class MainSimulatorManager(rt_logging.LoggingClass):

    def __init__(self, rank, config, parent: MainProcess, send_interface: MainMPISendInterface,
                 stim_decider: StimDecider):

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
        rank = self.config['rank']['decoder']
        self.send_interface.send_channel_selection(rank, trode_list)

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
        self.send_interface.send_turn_on_datastreams(
            self.config['rank']['decoder'])

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

    def register_rec_type_message(self, message):
        self.rec_manager.register_rec_type_message(message)

    def trigger_termination(self):
        self.send_interface.terminate_all()

        self.parent.trigger_termination()


class MainSimulatorMPIRecvInterface(realtime_base.RealtimeMPIClass):

    def __init__(self, comm: MPI.Comm, rank, config, main_manager: MainSimulatorManager):
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
