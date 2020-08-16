import sys

from mpi4py import MPI
import threading
import struct
import time
import numpy as np

import spykshrk.realtime.realtime_logging as rt_logging
import spykshrk.realtime.realtime_base as realtime_base
import spykshrk.realtime.datatypes as datatypes
import spykshrk.realtime.simulator.nspike_data as nspike_data
import spykshrk.realtime.simulator.sim_databuffer as sim_databuffer
from spykshrk.realtime import binary_record
import spykshrk.realtime.timing_system as timing_system

from spikegadgets import trodesnetwork as tnp

class SimulatorError(RuntimeError):
    pass


class ReqDatatypeChannelDataMessage(rt_logging.PrintableMessage):
    def __init__(self, datatype, channel):
        self. datatype = datatype
        self.channel = channel


class StartAllStreamMessage(rt_logging.PrintableMessage):
    def __init__(self):
        pass


class StopAllStreamMessage(rt_logging.PrintableMessage):
    def __init__(self):
        pass


class PauseAllStreamMessages(rt_logging.PrintableMessage):
    def __init__(self):
        pass


class SimTrodeListMessage(rt_logging.PrintableMessage):
    def __init__(self, trode_list):
        self.trode_list = trode_list

#added MEC
class RippleTrodeListMessage(rt_logging.PrintableMessage):
    def __init__(self, ripple_trode_list):
        self.ripple_trode_list = ripple_trode_list

class TrodesDataReceiver(realtime_base.DataSourceReceiver):
# class SimulatorRemoteReceiver(realtime_base.DataSourceReceiver):
    """Class that receives data from trodes using its network api
    """
    def __init__(self, comm: MPI.Comm, rank, config, datatype):
        super().__init__(comm=comm, rank=rank, config=config, datatype=datatype)
        self.start = False
        self.stop = False
        self.timestamp = 0
        self.network = config['trodes_network']['networkobject'] 
        self.channels = []
        if self.datatype is datatypes.Datatypes.LFP:
            self.DataPointCls = datatypes.LFPPoint

        elif self.datatype is datatypes.Datatypes.SPIKES:
            self.DataPointCls = datatypes.SpikePoint

        elif self.datatype is datatypes.Datatypes.LINEAR_POSITION:
            #self.DataPointCls = datatypes.LinearPosPoint
            self.DataPointCls = datatypes.CameraModulePoint

        else:
            raise SimulatorError('{} is not a valid datatype.'.format(self.datatype))

    def register_datatype_channel(self, channel):
        self.channels.append(channel)
        
    def start_all_streams(self):
        if self.datatype is datatypes.Datatypes.LFP:
            chnls = [str(i) for i in self.channels]
            self.datastream = self.network.subscribeLFPData(300, chnls)
            self.datastream.initialize()
            self.buf = self.datastream.create_numpy_array()
            self.curntrode = -1
            self.subbedntrodes = len(chnls)
            print('ripple channels',self.channels)
            
        elif self.datatype is datatypes.Datatypes.SPIKES:
            chnls = [str(i)+',0' for i in self.channels]
            self.datastream = self.network.subscribeSpikesData(300, chnls)
            self.datastream.initialize()
            self.buf = self.datastream.create_numpy_array()
            print('encoder channels',self.channels)
            
        elif self.datatype is datatypes.Datatypes.LINEAR_POSITION:
            self.datastream = self.network.subscribeHighFreqData('PositionData', 'CameraModule', 20)
            self.datastream.initialize()
            ndtype = self.datastream.getDataType().dataFormat
            nbytesize = self.datastream.getDataType().byteSize
            bytesbuf = memoryview(bytes(nbytesize))
            self.buf = np.frombuffer(bytesbuf, dtype=np.dtype(ndtype))
            
        else:
            raise SimulatorError('{} is not a valid datatype.'.format(self.datatype))
        self.start = True

    def stop_all_streams(self):
        self.start = False

    def stop_iterator(self):
        self.stop = True

    def __iter__(self):
        return self

    def __next__(self):
        if self.stop:
            raise StopIteration()

        if not self.start:
            return None

        #if is lfp and curntrode is less than total subbed ntrodes
        #print('1')
        if self.datatype is datatypes.Datatypes.LFP:
            #print(self.curntrode, self.subbedntrodes)
            # if haven't gotten anything yet, curntrode = -1. if already sent all in one packet, curntrode = subbedntrodes
            if self.curntrode > 0 and self.curntrode < self.subbedntrodes:
                #print('lfp data 1st loop',self.buf)
                #print(curntrode)
                #print(subbedntrodes)
                # multiply raw LFP by voltage scaling factor for trodes amplifier (0.195) - this is specified in config file
                #print('raw LFP 1: ',self.buf[self.curntrode])
                self.buf[self.curntrode] = self.buf[self.curntrode]*self.config['trodes']['voltage_scaling_factor']
                #print('scaled LFP 1: ',self.timestamp.trodes_timestamp,self.buf[self.curntrode])
                pt = datatypes.LFPPoint(self.timestamp.trodes_timestamp, self.channels[self.curntrode], self.channels[self.curntrode], self.buf[self.curntrode])
                #print('pt = ',pt)
                self.curntrode = self.curntrode + 1
                return pt, None
        #print('4')
        
        n = self.datastream.available(0)
        if n:
            #print('5')
            byteswritten = 0
            systime = tnp.systemTimeMSecs()
            if self.datatype is datatypes.Datatypes.LFP:
                #print('lfp data 2nd loop',self.buf)
                self.timestamp = self.datastream.getData()
                # Reset curntrode value. If lfp buffer is more than 1, then above code will read from buffer before reading from Trodes stream
                self.curntrode = 0
                # multiply raw LFP by voltage scaling factor for trodes amplifier (0.195) - this is specified in config file
                #print('raw LFP 2: ',self.buf[self.curntrode])
                self.buf[self.curntrode] = self.buf[self.curntrode]*self.config['trodes']['voltage_scaling_factor']
                #print('scaled LFP 2: ',self.timestamp.trodes_timestamp,self.buf[self.curntrode])
                pt = datatypes.LFPPoint(self.timestamp.trodes_timestamp, self.channels[self.curntrode], self.channels[self.curntrode], self.buf[self.curntrode])
                #NOTE NOTE need to apply trodes amplifier correction factor here too!!!
                #print(pt)
                #print('lfp: ',self.buf)
                #print(self.channels)
                #print(self.curntrode)
                self.curntrode = 1
                return pt, None
                
            elif self.datatype is datatypes.Datatypes.SPIKES:
                self.timestamp = self.datastream.getData() #Data is [(ntrode, cluster, timestamp, [0-159 data - (i,data)] ) ]
                # Reshape data to look like what spykshrk expects
                #print('spikes: ',self.buf,self.timestamp)
                # multiply spike amplitude by voltage scaling factor for trodes amplifier (0.195) - this is specified in config file
                d = self.buf[0][3][:,1]
                #print('raw: ',d)
                d = d*self.config['trodes']['voltage_scaling_factor']
                #print('scaled: ',d)
                d = d.astype(int)
                newshape = (int(len(d)/40), 40)
                #print(datatypes.SpikePoint(self.timestamp.trodes_timestamp, self.buf[0][0], np.reshape(d, newshape)))
                return datatypes.SpikePoint(self.timestamp.trodes_timestamp, self.buf[0][0], np.reshape(d, newshape)), None
                
            elif self.datatype is datatypes.Datatypes.LINEAR_POSITION:
                byteswritten = self.datastream.readData(self.buf) #Data is [(timestamp, linear segment, position, x location, y location)]
                #print(self.buf[0][0], self.buf[0][1], self.buf[0][2])
                return datatypes.CameraModulePoint(self.buf[0][0], self.buf[0][1], self.buf[0][2], self.buf[0][3], self.buf[0][4]), None

            # # Option to return timing message but disabled
            # timing_message = None
            # data_message = self.DataPointCls
            # return self.DataPointCls, timing_message

        else:
            return None



class SimulatorRemoteReceiver(realtime_base.DataSourceReceiver):
    """ A Class to be created and used by ranks that need to communicate with the Simulator Process/Rank.
    
    Goal is to provide an abstraction layer for interacting with other sources.
    """
    def __init__(self, comm: MPI.Comm, rank, config, datatype):
        super().__init__(comm=comm, rank=rank, config=config, datatype=datatype)
        self.start = False
        self.stop = False

        self.time_bytes = bytearray(100)
        self.mpi_reqs = []
        self.mpi_statuses = []

        if self.datatype is datatypes.Datatypes.LFP:

            self.data_bytes = bytearray(datatypes.LFPPoint.packed_message_size())
            self.mpi_sim_data_tag = realtime_base.MPIMessageTag.SIMULATOR_LFP_DATA
            self.config_enable_timing = 'enable_lfp'
            self.DataPointCls = datatypes.LFPPoint

            pass
        elif self.datatype is datatypes.Datatypes.SPIKES:
            self.data_bytes = bytearray(datatypes.SpikePoint.packed_message_size())
            self.mpi_sim_data_tag = realtime_base.MPIMessageTag.SIMULATOR_SPK_DATA
            self.config_enable_timing = 'enable_spk'
            self.DataPointCls = datatypes.SpikePoint
            pass
        elif self.datatype is datatypes.Datatypes.LINEAR_POSITION:
            self.data_bytes = bytearray(datatypes.LinearPosPoint.packed_message_size())
            self.mpi_sim_data_tag = realtime_base.MPIMessageTag.SIMULATOR_LINPOS_DATA
            self.config_enable_timing = 'enable_pos'
            self.DataPointCls = datatypes.LinearPosPoint
            pass
        else:
            raise SimulatorError('{} is not a valid datatype.'.format(self.datatype))

        self.mpi_reqs.append(self.comm.Irecv(buf=self.data_bytes,
                                             tag=self.mpi_sim_data_tag))
        self.mpi_statuses.append(MPI.Status)

    def register_datatype_channel(self, channel):
        self.comm.send(ReqDatatypeChannelDataMessage(datatype=self.datatype, channel=channel),
                       dest=self.config['rank']['simulator'],
                       tag=realtime_base.MPIMessageTag.COMMAND_MESSAGE.value)

    # This should be called after all initialization has been done and the first barrier has passed
    def start_all_streams(self):
        self.comm.send(StartAllStreamMessage(), dest=self.config['rank']['simulator'],
                       tag=realtime_base.MPIMessageTag.COMMAND_MESSAGE.value)
        self.start = True

    def stop_all_streams(self):
        self.comm.send(StopAllStreamMessage(), dest=self.config['rank']['simulator'],
                       tag=realtime_base.MPIMessageTag.COMMAND_MESSAGE.value)
        self.start = False

    def stop_iterator(self):
        self.stop = True

    def __iter__(self):
        return self

    def __next__(self):
        if self.stop:
            raise StopIteration()

        if not self.start:
            return None

        rdy = MPI.Request.Testall(requests=self.mpi_reqs)

        if rdy:
            data_message = self.DataPointCls.unpack(self.data_bytes)
            self.mpi_reqs[0] = self.comm.Irecv(buf=self.data_bytes,
                                               tag=self.mpi_sim_data_tag)

            # Option to return timing message but disabled
            timing_message = None
            return data_message, timing_message

        else:
            return None


class SimulatorSendInterface(realtime_base.RealtimeMPIClass):

    def __init__(self, comm: MPI.Comm, rank, config):
        super().__init__(comm=comm, rank=rank, config=config)

    def send_record_register_messages(self, record_register_messages):
        self.class_log.debug("Sending binary record registration messages.")
        for message in record_register_messages:
            self.comm.send(obj=message, dest=self.config['rank']['supervisor'],
                           tag=realtime_base.MPIMessageTag.COMMAND_MESSAGE)

    def send_terminate_error(self, msg):
        self.comm.send(realtime_base.TerminateErrorMessage(msg),
                       dest=self.config['rank']['supervisor'],
                       tag=realtime_base.MPIMessageTag.COMMAND_MESSAGE)

    def send_ntrode_list(self, ntrode_list):
        self.comm.send(obj=SimTrodeListMessage(ntrode_list),
                       dest=self.config['rank']['supervisor'],
                       tag=realtime_base.MPIMessageTag.COMMAND_MESSAGE)

    #added MEC
    def send_ripple_ntrode_list(self, ripple_ntrode_list):
        self.comm.send(obj=RippleTrodeListMessage(ripple_ntrode_list),
                       dest=self.config['rank']['supervisor'],
                       tag=realtime_base.MPIMessageTag.COMMAND_MESSAGE)

    def send_time_sync_other(self):
        rank_list = list(range(self.comm.size))
        rank_list.remove(self.rank)
        rank_list.remove(self.config['rank']['supervisor'])
        for rank in rank_list:
            self.comm.send(obj=realtime_base.TimeSyncInit(), dest=rank,
                           tag=realtime_base.MPIMessageTag.COMMAND_MESSAGE)

    def send_time_sync_report(self, time):
        self.comm.send(obj=realtime_base.TimeSyncReport(time),
                       dest=self.config['rank']['supervisor'],
                       tag=realtime_base.MPIMessageTag.COMMAND_MESSAGE)

    def all_barrier(self):
        self.comm.Barrier()

    def send_terminate(self):
        self.class_log.debug("Terminate all other ranks.")
        self.comm.send(obj=realtime_base.TerminateMessage(), dest=self.config['rank']['supervisor'],
                       tag=realtime_base.MPIMessageTag.COMMAND_MESSAGE)


class Simulator(realtime_base.BinaryRecordBaseWithTiming, realtime_base.RealtimeMPIClass):
    def __init__(self, comm, rank, config, mpi_send: SimulatorSendInterface, local_rec_manager):
        super().__init__(comm=comm, rank=rank, config=config,
                         local_rec_manager=local_rec_manager, send_interface=mpi_send)
        self.mpi_send = mpi_send

        self._stop_next = False

        try:
            self.nspike_anim = nspike_data.AnimalInfo(**config['simulator']['nspike_animal_info'])
            lfp_stream = nspike_data.EEGDataStream(self.nspike_anim)
            pos_stream = nspike_data.PosMatDataStream(self.nspike_anim)
            spk_stream = nspike_data.SpkDataStream(self.nspike_anim)
            self.databuffer = sim_databuffer.SimDataBuffer([lfp_stream(), spk_stream(), pos_stream()])
            #self.databuffer = sim_databuffer.SimDataBuffer([lfp_stream() ])

            self.lfp_chan_req_dict = {}
            self.spk_chan_req_dict = {}
            self.pos_chan_req = []
            self.data_itr = self.databuffer()


        except TypeError as err:
            self.class_log.exception("TypeError: nspike_animal_info does not match nspike_data.AnimalInfo arguments.",
                                     exc_info=err)
            self.mpi_send.send_terminate_error("For SimulatorThread, nspike_animal_info config did "
                                               "not match nspike_data.AnimalInfo arguments.")

        self.start_time = time.time()
        self.ntrode_list_sent = False
        self.running = False

    def send_ntrode_list(self):
        # Send ntrode configuration.  This automatically triggers a cascade of messages to start the simulation
        # and receiving ranks
        # does this do anything when trodes is the datasource??? - NO, it seems like this section is NOT used with trodes
        self.mpi_send.send_ntrode_list(self.config['simulator']['nspike_animal_info']['tetrodes'])

    def update_cont_chan_req(self, dest_rank, lfp_chan):
        if lfp_chan not in self.nspike_anim.tetrodes:
            raise SimulatorError("Rank {:} tried to request channel ({:}) not available in animal info.".
                                 format(dest_rank, lfp_chan))
        if lfp_chan in self.lfp_chan_req_dict:
            self.class_log.error(("LFP channels cannot be requested by more than one rank. Channel ({:}) requested by "
                                  "rank ({:}) but is already owned by rank ({:}). "
                                  "Overwriting previous assignment.").format(lfp_chan, dest_rank,
                                                                             self.lfp_chan_req_dict[lfp_chan]))
        self.lfp_chan_req_dict[lfp_chan] = dest_rank
        self.class_log.debug("Continuous channel/ntrode {:} registered by rank {:}".format(lfp_chan, dest_rank))

    def update_spk_chan_req(self, dest_rank, spk_chan):

        if spk_chan not in self.nspike_anim.tetrodes:
            raise SimulatorError("Rank {:} tried to request channel ({:}) not available in animal info.".
                                 format(dest_rank, spk_chan))

        spk_chan_assign = self.spk_chan_req_dict.setdefault(spk_chan, set())
        spk_chan_assign.add(dest_rank)

        self.class_log.debug("Spike channel/ntrode {:} registered by rank {:}".format(spk_chan, dest_rank))

    def update_linpos_chan_req(self, dest_rank):

        self.class_log.debug("Linear position registered by rank {:}".format(dest_rank))
        self.pos_chan_req.append(dest_rank)

    def sync_time(self):
        """Override normal sync time so Simulator will trigger the sync barrier of its consumers."""
        self.class_log.debug("Sending sync message to remaining nodes ({}).".format(self.rank))
        self.mpi_send.send_time_sync_other()
        super().sync_time()

    def start_datastream(self):
        self.class_log.debug("Start datastream.")
        self.running = True

    def pause_datastream(self):
        self.running = False

    def send_next_data(self):

        # Distribute tetrode list to nodes in 3 sec
        current_time = time.time()
        if not self.ntrode_list_sent and (current_time - self.start_time < 3.0):
            # First Barrier to finish setting up nodes, right before simulator starts by sending ntrode list
            self.class_log.debug("First Barrier")
            self.comm.Barrier()

            # Pause to allow other MPI initialization messages to finish before sending tetrode list, which
            # triggers the start of recording
            time.sleep(0.5)
            self.send_ntrode_list()
            self.ntrode_list_sent = True


        if not self.running:
            return None

        try:
            data_to_send = self.data_itr.__next__()
            if isinstance(data_to_send, datatypes.LFPPoint):

                self.record_timing(timestamp=data_to_send.timestamp, elec_grp_id=data_to_send.elec_grp_id,
                                   datatype=datatypes.Datatypes.LFP, label='sim_send')

                try:
                    bytes_to_send = data_to_send.pack()

                    self.comm.Ssend(buf=bytes_to_send, dest=self.lfp_chan_req_dict[data_to_send.elec_grp_id],
                                    tag=realtime_base.MPIMessageTag.SIMULATOR_LFP_DATA)

                except KeyError as err:
                    self.class_log.exception(("KeyError: Tetrode id ({:}) not in lfp channel request dict {:}, "
                                              "was likely never requested by a receiving/computing ranks.").
                                             format(data_to_send.ntrode_index, self.lfp_chan_req_dict), exc_info=err)

            elif isinstance(data_to_send, datatypes.SpikePoint):

                self.record_timing(timestamp=data_to_send.timestamp, elec_grp_id=data_to_send.elec_grp_id,
                                   datatype=datatypes.Datatypes.SPIKES, label='sim_send')
                try:
                    bytes_to_send = data_to_send.pack()

                    for dest_rank in self.spk_chan_req_dict[data_to_send.elec_grp_id]:
                        self.comm.Ssend(buf=bytes_to_send, dest=dest_rank,
                                        tag=realtime_base.MPIMessageTag.SIMULATOR_SPK_DATA)

                except KeyError as err:
                    self.class_log.exception(("KeyError: Tetrode id ({:}) not in spike channel request dict {:}, "
                                              "was likely never requested by a receiving/computing ranks.").
                                             format(data_to_send.elec_grp_id, self.spk_chan_req_dict), exc_info=err)

            elif isinstance(data_to_send, datatypes.LinearPosPoint):
                self.record_timing(timestamp=data_to_send.timestamp, elec_grp_id=-1,
                                   datatype=datatypes.Datatypes.LINEAR_POSITION, label='sim_send')
                bytes_to_send = data_to_send.pack()

                for dest_rank in self.pos_chan_req:
                    self.comm.Ssend(buf=bytes_to_send, dest=dest_rank,
                                    tag=realtime_base.MPIMessageTag.SIMULATOR_LINPOS_DATA)

        except StopIteration as err:
            # Simulation is done, send terminate message
            self.mpi_send.send_terminate()
            raise


class SimulatorProcess(realtime_base.RealtimeProcess):
    def __init__(self, comm: MPI.Comm, rank, config):
        super().__init__(comm=comm, rank=rank, config=config)
        self.terminate = False

        self.local_rec_manager = binary_record.RemoteBinaryRecordsManager(manager_label='state', local_rank=rank,
                                                                          manager_rank=config['rank']['supervisor'])

        self.mpi_send = SimulatorSendInterface(comm=comm, rank=rank, config=config)

        self.sim = Simulator(comm=comm, rank=rank, config=config,
                             mpi_send=self.mpi_send,
                             local_rec_manager=self.local_rec_manager)

        self.mpi_recv = SimulatorRecvInterface(comm=comm, rank=rank, config=config, simulator=self.sim)


    def trigger_termination(self):
        self.terminate = True

    def main_loop(self):

        self.sim.setup_mpi()

        try:
            while not self.terminate:

                self.mpi_recv.__next__()
                self.sim.send_next_data()
        except StopIteration as err:
            self.class_log.info("Simulator Process Main reached end, exiting.")


class SimulatorRecvInterface(realtime_base.RealtimeMPIClass):

    def __init__(self, comm: MPI.Comm, rank, config, simulator: Simulator):
        super(SimulatorRecvInterface, self).__init__(comm=comm, rank=rank, config=config)
        self.sim = simulator

        self.req = self.comm.irecv(tag=realtime_base.MPIMessageTag.COMMAND_MESSAGE)
        self.mpi_status = MPI.Status()

    def __next__(self):
        rdy, msg = self.req.test(status=self.mpi_status)
        if rdy:
            self.process_request_message(msg)

            self.req = self.comm.irecv(tag=realtime_base.MPIMessageTag.COMMAND_MESSAGE)

    def process_request_message(self, message):

        if isinstance(message, ReqDatatypeChannelDataMessage):
            if message.datatype is datatypes.Datatypes.LFP:
                self.sim.update_cont_chan_req(self.mpi_status.source, message.channel)

            elif message.datatype is datatypes.Datatypes.SPIKES:
                self.sim.update_spk_chan_req(self.mpi_status.source, message.channel)

            elif message.datatype is datatypes.Datatypes.LINEAR_POSITION:
                self.sim.update_linpos_chan_req(self.mpi_status.source)

        elif isinstance(message, StartAllStreamMessage):
            self.sim.start_datastream()

        elif isinstance(message, PauseAllStreamMessages):
            self.sim.pause_datastream()

        elif isinstance(message, binary_record.BinaryRecordCreateMessage):
            self.sim.set_record_writer_from_message(message)

        elif isinstance(message, realtime_base.StartRecordMessage):
            self.sim.start_record_writing()

        elif isinstance(message, realtime_base.StopRecordMessage):
            self.sim.stop_record_writing()

        elif isinstance(message, realtime_base.CloseRecordMessage):
            self.sim.close_record()

        elif isinstance(message, realtime_base.TimeSyncInit):
            self.class_log.debug('Received TimeSyncInit.')
            self.sim.sync_time()

        elif isinstance(message, realtime_base.TimeSyncSetOffset):
            self.sim.update_offset(message.offset_time)

        elif isinstance(message, realtime_base.TerminateMessage):
            raise StopIteration()

