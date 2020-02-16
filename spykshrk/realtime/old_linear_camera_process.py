import os
import struct
import numpy as np
import math
from mpi4py import MPI
from spykshrk.realtime import realtime_base, realtime_logging, binary_record, datatypes
from spykshrk.realtime.simulator import simulator_process

from spykshrk.realtime.datatypes import SpikePoint, LinearPosPoint, CameraModulePoint
from spykshrk.realtime.realtime_base import ChannelSelection, TurnOnDataStream
from spykshrk.realtime.tetrode_models import kernel_encoder
import spykshrk.realtime.rst.RSTPython as RST

# for now this script just holds the two classes: LinearPositionAssignment and VelocityCalculator
# the functions in these classed are called in encoder_process

# class LinearPosandVelResultsMessage(realtime_logging.PrintableMessage):

#     _header_byte_fmt = '=qidi'
#     _header_byte_len = struct.calcsize(_header_byte_fmt)

#     def __init__(self, timestamp, lin_pos, lin_vel):
#         self.timestamp = timestamp
#         self.lin_pos = lin_pos
#         self.lin_vel = lin_vel

#     def pack(self):
#         #pos_hist_len = len(self.pos_hist)
#         #pos_hist_byte_len = pos_hist_len * struct.calcsize('=d')

#         message_bytes = struct.pack(self._header_byte_fmt,
#                                     self.timestamp,
#                                     self.lin_pos,
#                                     self.lin_vel)

#         #message_bytes = message_bytes + self.pos_hist.tobytes()
#         message_bytes = message_bytes

#         return message_bytes

#     @classmethod
#     def unpack(cls, message_bytes):
#         timestamp, lin_pos, lin_vel = struct.unpack(cls._header_byte_fmt,message_bytes[0:cls._header_byte_len])

#         #pos_hist = np.frombuffer(message_bytes[cls._header_byte_len:cls._header_byte_len+pos_hist_len])

#         return cls(timestamp=timestamp, lin_pos=lin_pos, lin_vel=lin_vel)

# class ArmCoordinatessResultsMessage(realtime_logging.PrintableMessage):

#     _header_byte_fmt = '=qidi'
#     _header_byte_len = struct.calcsize(_header_byte_fmt)

#     def __init__(self, timestamp, arm_coords):
#         self.timestamp = timestamp
#         self.arm_coords = arm_coords

#     def pack(self):
#         #pos_hist_len = len(self.pos_hist)
#         #pos_hist_byte_len = pos_hist_len * struct.calcsize('=d')

#         message_bytes = struct.pack(self._header_byte_fmt,
#                                     self.timestamp,
#                                     self.arm_coords)

#         #message_bytes = message_bytes + self.pos_hist.tobytes()

#         return message_bytes

#     @classmethod
#     def unpack(cls, message_bytes):
#         timestamp, arm_coords = struct.unpack(cls._header_byte_fmt,message_bytes[0:cls._header_byte_len])

#         #pos_hist = np.frombuffer(message_bytes[cls._header_byte_len:cls._header_byte_len+pos_hist_len])

#         return cls(timestamp=timestamp, arm_coords=arm_coords)

# class CameraMPISendInterface(realtime_base.RealtimeMPIClass):
#     def __init__(self, comm: MPI.Comm, rank, config):
#         super(EncoderMPISendInterface, self).__init__(comm=comm, rank=rank, config=config)

#     def send_record_register_messages(self, record_register_messages):
#         self.class_log.debug("Sending binary record registration messages.")
#         for message in record_register_messages:
#             self.comm.send(obj=message, dest=self.config['rank']['supervisor'],
#                            tag=realtime_base.MPIMessageTag.COMMAND_MESSAGE)
#         self.class_log.debug("Done sending binary record registration messages.")

#         # define a send function, then have a for loop for each enocder node, and decoder, and ripple
#     def send_linear_pos_and_vel(self, query_result_message: LinearPosandVelResultsMessage):
#         # need loop for encoder nodes
#         self.comm.Send(buf=query_result_message.pack(), dest=self.config['rank']['enocder'],
#                        tag=realtime_base.MPIMessageTag.LINEAR_POS_AND_VEL)
#         self.comm.Send(buf=query_result_message.pack(), dest=self.config['rank']['decoder'],
#                        tag=realtime_base.MPIMessageTag.LINEAR_POS_AND_VEL)
#         self.comm.Send(buf=query_result_message.pack(), dest=self.config['rank']['ripple'],
#                        tag=realtime_base.MPIMessageTag.LINEAR_POS_AND_VEL)

#     def send_arm_coordinates(self, query_result_message: ArmCoordsResultsMessage):
#         self.comm.Send(buf=query_result_message.pack(), dest=self.config['rank']['encoder'],
#                        tag=realtime_base.MPIMessageTag.ARM_COORDINATES)
#         self.comm.Send(buf=query_result_message.pack(), dest=self.config['rank']['decoder'],
#                        tag=realtime_base.MPIMessageTag.ARM_COORDINATES)

#     def send_time_sync_report(self, time):
#         self.comm.send(obj=realtime_base.TimeSyncReport(time),
#                        dest=self.config['rank']['supervisor'],
#                        tag=realtime_base.MPIMessageTag.COMMAND_MESSAGE)

#     def all_barrier(self):
#         self.comm.Barrier()

class LinearPositionAssignment:
    def __init__(self):
        self.segment = 0
        self.segment_pos = 0
        self.shift_linear_distance_by_arm_dictionary = dict()
        #this line runs the arm_shift_dict during __init__
        self.arm_shift_dictionary()

    def arm_shift_dictionary(self):
        # 0-6 = box, 7 = arm1 ... 14 = arm8
        # 6-9-19: seems to work as expected! matches offline linearization!
        # 8-15-19: updated for new track geometry
        # 0 = home->rip/wait, 1-8 = rip/wait->arms, 9-16 = outer arms

        #self.shift_linear_distance_by_arm_dictionary

        # this bins position into 5cm bins by dividing all the positions by 5
        # note: current max position = 140

        hardcode_armorder = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14] #add progressive stagger in this order
        hardcode_shiftamount = 4 # add this stagger to sum of previous shifts (was 20 for 1cm)
        # for now set all arm lengths to 60 for 1cm (12 for 5cm)
        linearization_arm_length = 12

        # Define dictionary for shifts for each arm segment
        #shift_linear_distance_by_arm_dictionary = dict() # initialize empty dictionary
        # with this setup max position is 129
        for arm in hardcode_armorder: # for each outer arm
            if arm < 7: # if all box segments, do nothing
                temporary_variable_shift = 0

            #for first arm replace linearization_arm_length with 0 for the box
            elif arm == 7:
                temporary_variable_shift = hardcode_shiftamount + 1 + self.shift_linear_distance_by_arm_dictionary[hardcode_armorder[arm - 1]]

            else: # if arms 2-8, shift with gap
                temporary_variable_shift = hardcode_shiftamount + 12 + self.shift_linear_distance_by_arm_dictionary[hardcode_armorder[arm - 1]]

            self.shift_linear_distance_by_arm_dictionary[arm] = temporary_variable_shift

        return self.shift_linear_distance_by_arm_dictionary

    def assign_position(self, segment, segment_pos):
        self.assigned_pos = 0

        # TO DO: split box into two segments, inner segment (0): 0.3-0.7, outer segment (1): 0-0.3 and 0.7-1
        # also need to bin position here - use math.ceil to round UP to nearest integer

        # make all assigned positions in the box between 0-1, so now the box is effectively 1 bin
        if segment < 7:
            #self.assigned_pos = segment_pos + self.shift_linear_distance_by_arm_dictionary[segment]
            if segment_pos < 0.7 and segment_pos > 0.3:
                self.assigned_pos = 0
            else:
                self.assigned_pos = 1
        else:
            self.assigned_pos = math.ceil(segment_pos*12 + self.shift_linear_distance_by_arm_dictionary[segment])

        return self.assigned_pos

class VelocityCalculator:
    def __init__(self):
        self.x = 0
        self.y = 0
        self.lastx = 0
        self.lasty = 0

        #this is the number of speed measurements used to smooth
        self.NSPEED_FILT_POINTS = 30
        self.speed = [0] * self.NSPEED_FILT_POINTS
        self.speedFilt = [0] * self.NSPEED_FILT_POINTS

        #this is a half-gaussian kernel used to smooth the instanteous speed
        self.speedFilterValues = [0.0393,0.0392,0.0391,0.0389,0.0387,0.0385,0.0382,0.0379,
        0.0375,0.0371,0.0367,0.0362,0.0357,0.0352,0.0347,0.0341,0.0334,0.0328,0.0321,
        0.0315,0.0307,0.0300,0.0293,0.0285,0.0278,0.0270,0.0262,0.0254,0.0246,0.0238]

        for i in np.arange(0,self.NSPEED_FILT_POINTS,1):
            self.speedFilt[i] = self.speedFilterValues[i]
        self.ind = self.NSPEED_FILT_POINTS - 1;

    def calculator(self, x, y):
        self.smoothSpeed = 0
        i = 0
        tmpind = 0
        #need to bring in network.pxpercm from trodes
        #cmperpx = 1/network.pxpercm
        cmperpx = 0.2

        # note: for remy cmperpx should be <0.2
        # it seems like the speed is still pretty high with jittering of headstage...
        # maybe this is because positon isnt smoothed??

        self.speed[self.ind] = ((x * cmperpx - self.lastx) * (x * cmperpx - self.lastx) +
                      (y * cmperpx - self.lasty) * (y * cmperpx - self.lasty))
        #print(x,y,self.lastx,self.lasty,network.pxpercm,self.speed[0])

        # this is distance / time - because 1/0.03 = 30
        if self.speed[self.ind] != 0:
            self.speed[self.ind] = np.sqrt(self.speed[self.ind])*30

        self.lastx = x * cmperpx
        self.lasty = y * cmperpx

        for i in np.arange(0,self.NSPEED_FILT_POINTS,1):
            tmpind = (self.ind + i) % self.NSPEED_FILT_POINTS
            self.smoothSpeed = self.smoothSpeed + self.speed[tmpind]*self.speedFilt[i]

        self.ind = self.ind - 1

        if self.ind < 0:
            self.ind = self.NSPEED_FILT_POINTS - 1

        return self.smoothSpeed

#initialize VelocityCalculator
velCalc = VelocityCalculator()

#initialize LinearPositionAssignment
linPosAssign = LinearPositionAssignment()

# class CameraManager(realtime_base.BinaryRecordBase):

#     def __init__(self, rank, config, local_rec_manager, send_interface: EncoderMPISendInterface,
#                  pos_interface: realtime_base.DataSourceReceiver):

#         super(RStarEncoderManager, self).__init__(rank=rank,
#                                                   local_rec_manager=local_rec_manager,
#                                                   send_interface=send_interface,
#                                                   rec_ids=[realtime_base.RecordIDs.ENCODER_QUERY,
#                                                            realtime_base.RecordIDs.ENCODER_OUTPUT],
#                                                   rec_labels=[['timestamp',
#                                                                'elec_grp_id',
#                                                                'weight',
#                                                                'position'],
#                                                               ['timestamp',
#                                                                'elec_grp_id',
#                                                                'position'] +
#                                                               ['x{:0{dig}d}'.
#                                                                format(x, dig=len(str(config['encoder']
#                                                                                      ['position']['bins'])))
#                                                                for x in range(config['encoder']['position']['bins'])]],
#                                                   rec_formats=['qidd',
#                                                                'qid'+'d'*config['encoder']['position']['bins']])

#         self.rank = rank
#         self.config = config
#         self.mpi_send = send_interface
#         self.pos_interface = pos_interface

#         self.encoders = {}

#         self.spk_counter = 0
#         self.pos_counter = 0

#         self.current_pos = 0
#         self.current_vel = 0

#         self.speed = 0
#         self.linposassign = 0


#     def register_pos_datatype(self):
#         # Register position, right now only one position channel is supported
#         self.pos_interface.register_datatype_channel(-1)

#     def turn_on_datastreams(self):
#         self.class_log.info("Turn on datastreams.")
#         self.pos_interface.start_all_streams()

#     def trigger_termination(self):
#         self.spike_interface.stop_iterator()

#     def process_next_data(self):

#         msgs = self.pos_interface.__next__()
#         if msgs is None:
#             # No data avaliable but datastreams are still running, continue polling
#             pass
#         else:
#             datapoint = msgs[0]
#             timing_msg = msgs[1]
#             if isinstance(datapoint, CameraModulePoint):
#                 self.pos_counter += 1

#                 #pasrsing the message into my variables
#                 self.current_pos = datapoint.x
#                 self.current_vel = datapoint.vel

#                 #send my variables to calculators
#                 speed = velCalc.calculator(npbuff[0][3], npbuff[0][4])
#                 linposassign = linPosAssign.assign_position(npbuff[0][1], npbuff[0][2])

#                 # want to report timing and latency of the velocity caclulator
#                 self.record_timing(timestamp=datapoint.timestamp, elec_grp_id=datapoint.elec_grp_id,
#                                    datatype=datatypes.Datatypes.POSITION, label='camera')

#                 self.write_record(realtime_base.RecordIDs.CAMERA_OUTPUT,
#                                       query_result.query_time,
#                                       query_result.elec_grp_id,
#                                       self.current_pos,
#                                       *query_result.query_hist)

#                 self.mpi_send.send_linear_pos_and_vel(LinearPosandVelResultsMessage(timestamp=query_result.query_time,
#                                                                                position=self.linposassign,
#                                                                                velocity=self.speed))
#                 self.mpi_send.send_arm_coordinates(ArmCoordinatessResultsMessage(timestamp=query_result.query_time,
#                                                                                arm_coords=self.arm_coords))

#                 if self.pos_counter % 1000 == 0:
#                     self.class_log.info('Received {} pos datapoints.'.format(self.pos_counter))
#                 pass


# class CameraMPIRecvInterface(realtime_base.RealtimeMPIClass):
#     def __init__(self, comm: MPI.Comm, rank, config, camera_manager: CameraManager):
#         super(EncoderMPIRecvInterface, self).__init__(comm=comm, rank=rank, config=config)
#         self.camera_manager = camera_manager

#         self.req = self.comm.irecv(tag=realtime_base.MPIMessageTag.COMMAND_MESSAGE.value)

#     def __next__(self):
#         rdy, msg = self.req.test()
#         if rdy:
#             self.process_request_message(msg)

#             self.req = self.comm.irecv(tag=realtime_base.MPIMessageTag.COMMAND_MESSAGE.value)

#     def process_request_message(self, message):

#         if isinstance(message, realtime_base.TerminateMessage):
#             self.class_log.debug("Received TerminateMessage")
#             raise StopIteration()

#         elif isinstance(message, realtime_base.NumTrodesMessage):
#             self.class_log.debug("Received number of NTrodes Message.")
#             self.camera_manager.set_num_trodes(message)

#         elif isinstance(message, ChannelSelection):
#             self.class_log.debug("Received NTrode channel selection {:}.".format(message.ntrode_list))
#             self.camera_manager.select_ntrodes(message.ntrode_list)

#         elif isinstance(message, TurnOnDataStream):
#             self.class_log.debug("Turn on data stream")
#             self.camera_manager.turn_on_datastreams()

#         elif isinstance(message, binary_record.BinaryRecordCreateMessage):
#             self.camera_manager.set_record_writer_from_message(message)

#         elif isinstance(message, realtime_base.TimeSyncInit):
#             self.camera_manager.sync_time()

#         elif isinstance(message, realtime_base.TimeSyncSetOffset):
#             self.camera_manager.update_offset(message.offset_time)

#         elif isinstance(message, realtime_base.StartRecordMessage):
#             self.camera_manager.start_record_writing()

#         elif isinstance(message, realtime_base.StopRecordMessage):
#             self.camera_manager.stop_record_writing()


# class CameraProcess(realtime_base.RealtimeProcess):
#     def __init__(self, comm: MPI.Comm, rank, config):

#         super().__init__(comm, rank, config)

#         self.local_rec_manager = binary_record.RemoteBinaryRecordsManager(manager_label='state', local_rank=rank,
#                                                                           manager_rank=config['rank']['supervisor'])

#         self.mpi_send = CameraMPISendInterface(comm=comm, rank=rank, config=config)

#         if self.config['datasource'] == 'simulator':
#             pos_interface = simulator_process.SimulatorRemoteReceiver(comm=self.comm,
#                                                                       rank=self.rank,
#                                                                       config=self.config,
#                                                                       datatype=datatypes.Datatypes.LINEAR_POSITION)
#         elif self.config['datasource'] == 'trodes':
#             pos_interface = simulator_process.TrodesDataReceiver(comm=self.comm,
#                                                                       rank=self.rank,
#                                                                       config=self.config,
#                                                                       datatype=datatypes.Datatypes.LINEAR_POSITION)

#         self.camera_manager = CameraManager(rank=rank,
#                                            config=config,
#                                            local_rec_manager=self.local_rec_manager,
#                                            send_interface=self.mpi_send,
#                                            pos_interface=pos_interface)

#         self.mpi_recv = CameraMPIRecvInterface(comm=comm, rank=rank, config=config, encoder_manager=self.camera_manager)

#         self.terminate = False

#         # config['trodes_network']['networkobject'].registerTerminateCallback(self.trigger_termination)

#         # First Barrier to finish setting up nodes
#         self.class_log.debug("First Barrier")
#         self.comm.Barrier()

#     def trigger_termination(self):
#         self.terminate = True

#     def main_loop(self):

#         self.camera_manager.setup_mpi()

#         # First thing register pos datatype
#         self.camera_manager.register_pos_datatype()

#         try:
#             while not self.terminate:
#                 self.mpi_recv.__next__()
#                 self.camera_manager.process_next_data()

#         except StopIteration as ex:
#             self.class_log.info('Terminating CameraProcess (rank: {:})'.format(self.rank))

#         self.class_log.info("Camera Process reached end, exiting.")
