import os
import struct
import numpy as np
from mpi4py import MPI
from spykshrk.realtime import realtime_base, realtime_logging, binary_record, datatypes
from spykshrk.realtime.simulator import simulator_process

from spykshrk.realtime.datatypes import SpikePoint, LinearPosPoint
from spykshrk.realtime.realtime_base import ChannelSelection, TurnOnDataStream
from spykshrk.realtime.tetrode_models import kernel_encoder
import spykshrk.realtime.rst.RSTPython as RST


class SpikeDecodeResultsMessage(realtime_logging.PrintableMessage):

    _header_byte_fmt = '=qii'
    _header_byte_len = struct.calcsize(_header_byte_fmt)

    def __init__(self, timestamp, ntrode_id, pos_hist):
        self.timestamp = timestamp
        self.ntrode_id = ntrode_id
        self.pos_hist = pos_hist

    def pack(self):
        pos_hist_len = len(self.pos_hist)
        pos_hist_byte_len = pos_hist_len * struct.calcsize('=d')

        message_bytes = struct.pack(self._header_byte_fmt,
                                    self.timestamp,
                                    self.ntrode_id,
                                    pos_hist_byte_len)

        message_bytes = message_bytes + self.pos_hist.tobytes()

        return message_bytes

    @classmethod
    def unpack(cls, message_bytes):
        timestamp, ntrode_id, pos_hist_len = struct.unpack(cls._header_byte_fmt,
                                                           message_bytes[0:cls._header_byte_len])

        pos_hist = np.frombuffer(message_bytes[cls._header_byte_len:cls._header_byte_len+pos_hist_len])

        return cls(timestamp=timestamp, ntrode_id=ntrode_id, pos_hist=pos_hist)


class EncoderMPISendInterface(realtime_base.RealtimeMPIClass):
    def __init__(self, comm: MPI.Comm, rank, config):
        super(EncoderMPISendInterface, self).__init__(comm=comm, rank=rank, config=config)

    def send_record_register_messages(self, record_register_messages):
        self.class_log.debug("Sending binary record registration messages.")
        for message in record_register_messages:
            self.comm.send(obj=message, dest=self.config['rank']['supervisor'],
                           tag=realtime_base.MPIMessageTag.COMMAND_MESSAGE)

    def send_decoded_spike(self, query_result_message: SpikeDecodeResultsMessage):
        self.comm.Send(buf=query_result_message.pack(), dest=self.config['rank']['decoder'],
                       tag=realtime_base.MPIMessageTag.SPIKE_DECODE_DATA)

    def send_time_sync_report(self, time):
        self.comm.send(obj=realtime_base.TimeSyncReport(time),
                       dest=self.config['rank']['supervisor'],
                       tag=realtime_base.MPIMessageTag.COMMAND_MESSAGE)

    def all_barrier(self):
        self.comm.Barrier()


class RStarEncoderManager(realtime_base.BinaryRecordBaseWithTiming, realtime_logging.LoggingClass):

    def __init__(self, rank, config, local_rec_manager, send_interface: EncoderMPISendInterface,
                 spike_interface: simulator_process.SimulatorRemoteReceiver,
                 pos_interface: simulator_process.SimulatorRemoteReceiver):

        super(RStarEncoderManager, self).__init__(rank=rank,
                                                  local_rec_manager=local_rec_manager,
                                                  rec_ids=[realtime_base.RecordIDs.ENCODER_QUERY,
                                                           realtime_base.RecordIDs.ENCODER_OUTPUT],
                                                  rec_labels=[['timestamp',
                                                               'trodes_id',
                                                               'weight',
                                                               'position'],
                                                              ['timestamp',
                                                               'trode_id',
                                                               'position'] +
                                                              ['x'+str(x) for x in
                                                               range(config['encoder']['position']['bins'])]],
                                                  rec_formats=['qidd',
                                                               'qid'+'d'*config['encoder']['position']['bins']])
        self.rank = rank
        self.config = config
        self.mpi_send = send_interface
        self.spike_interface = spike_interface
        self.pos_interface = pos_interface

        self.mpi_send.send_record_register_messages(self.get_record_register_messages())

        kernel = RST.kernel_param(mean=config['encoder']['kernel']['mean'],
                                  stddev=config['encoder']['kernel']['std'],
                                  min_val=config['encoder']['kernel']['lower'],
                                  max_val=config['encoder']['kernel']['upper'],
                                  interval=config['encoder']['kernel']['interval'])

        pos_bin_struct = kernel_encoder.PosBinStruct([config['encoder']['position']['lower'],
                                                      config['encoder']['position']['upper']],
                                                     config['encoder']['position']['bins'])
        self.rst_param = kernel_encoder.RSTParameter(kernel, pos_bin_struct)
        self.encoders = {}

        # Register position, right now only one position channel is supported
        self.pos_interface.register_datatype_channel(-1)

        self.spk_counter = 0
        self.pos_counter = 0

        self.current_pos = 0
        self.current_vel = 0

    def set_num_trodes(self, message: realtime_base.NumTrodesMessage):
        self.num_ntrodes = message.num_ntrodes
        self.class_log.info('Set number of ntrodes: {:d}'.format(self.num_ntrodes))

    def select_ntrodes(self, ntrode_list):
        self.class_log.debug("Registering spiking channels: {:}.".format(ntrode_list))
        for ntrode in ntrode_list:
            self.spike_interface.register_datatype_channel(channel=ntrode)

            self.encoders.setdefault(ntrode, kernel_encoder.RSTKernelEncoder('/tmp/ntrode{:}'.
                                                                             format(ntrode),
                                                                             True, self.rst_param))

    def turn_on_datastreams(self):
        self.class_log.info("Turn on datastreams.")
        self.spike_interface.start_all_streams()
        self.pos_interface.start_all_streams()

    def begin_time_sync(self):
        self.class_log.debug("Begin time sync barrier ({}).".format(self.rank))
        self.mpi_send.all_barrier()
        self.mpi_send.send_time_sync_report(MPI.Wtime())
        self.class_log.debug("Report post barrier time ({}).".format(self.rank))

    def trigger_termination(self):
        self.spike_interface.stop_iterator()

    def process_next_data(self):

        msgs = self.spike_interface.__next__()

        if msgs is None:
            # No data avaliable but datastreams are still running, continue polling
            pass
        else:
            datapoint = msgs[0]
            timing_msg = msgs[1]
            if isinstance(datapoint, SpikePoint):
                self.record_timing(timestamp=datapoint.timestamp, ntrode_id=datapoint.ntrode_id,
                                   datatype=datatypes.Datatypes.SPIKES, label='enc_recv')

                self.spk_counter += 1
                amp_marks = [max(x) for x in datapoint.data]

                if max(amp_marks) > self.config['encoder']['spk_amp']:
                    query_result = self.encoders[datapoint.ntrode_id]. \
                        query_mark_hist(amp_marks,
                                        datapoint.timestamp,
                                        datapoint.ntrode_id)                # type: kernel_encoder.RSTKernelEncoderQuery

                    # for weight, position in zip(query_result.query_weights, query_result.query_positions):
                    #     self.write_record(realtime_base.RecordIDs.ENCODER_QUERY,
                    #                       query_result.query_time,
                    #                       query_result.ntrode_id,
                    #                       weight, position)

                    self.write_record(realtime_base.RecordIDs.ENCODER_OUTPUT,
                                      query_result.query_time,
                                      query_result.ntrode_id,
                                      self.current_pos,
                                      *query_result.query_hist)

                    self.record_timing(timestamp=datapoint.timestamp, ntrode_id=datapoint.ntrode_id,
                                       datatype=datatypes.Datatypes.SPIKES, label='spk_dec')

                    self.mpi_send.send_decoded_spike(SpikeDecodeResultsMessage(timestamp=query_result.query_time,
                                                                               ntrode_id=query_result.ntrode_id,
                                                                               pos_hist=query_result.query_hist))

                    if abs(self.current_vel) >= self.config['encoder']['vel']:

                        self.encoders[datapoint.ntrode_id].new_mark(amp_marks)

                        self.record_timing(timestamp=datapoint.timestamp, ntrode_id=datapoint.ntrode_id,
                                           datatype=datatypes.Datatypes.SPIKES, label='spk_enc')

                if self.spk_counter % 10000 == 0:
                    self.class_log.debug('Received {} spikes.'.format(self.spk_counter))
                pass

        msgs = self.pos_interface.__next__()
        if msgs is None:
            # No data avaliable but datastreams are still running, continue polling
            pass
        else:
            datapoint = msgs[0]
            timing_msg = msgs[1]
            if isinstance(datapoint, LinearPosPoint):
                self.pos_counter += 1

                self.current_pos = datapoint.x
                self.current_vel = datapoint.vel
                for encoder in self.encoders.values():
                    encoder.update_covariate(datapoint.x)

                if self.pos_counter % 1000 == 0:
                    self.class_log.info('Received {} pos datapoints.'.format(self.pos_counter))
                pass


class EncoderMPIRecvInterface(realtime_base.RealtimeMPIClass):
    def __init__(self, comm: MPI.Comm, rank, config, encoder_manager: RStarEncoderManager):
        super(EncoderMPIRecvInterface, self).__init__(comm=comm, rank=rank, config=config)
        self.enc_man = encoder_manager

        self.req = self.comm.irecv(tag=realtime_base.MPIMessageTag.COMMAND_MESSAGE.value)

    def __next__(self):
        rdy, msg = self.req.test()
        if rdy:
            self.process_request_message(msg)

            self.req = self.comm.irecv(tag=realtime_base.MPIMessageTag.COMMAND_MESSAGE.value)

    def process_request_message(self, message):

        if isinstance(message, realtime_base.TerminateMessage):
            self.class_log.debug("Received TerminateMessage")
            raise StopIteration()

        elif isinstance(message, realtime_base.NumTrodesMessage):
            self.class_log.debug("Received number of NTrodes Message.")
            self.enc_man.set_num_trodes(message)

        elif isinstance(message, ChannelSelection):
            self.class_log.debug("Received NTrode channel selection {:}.".format(message.ntrode_list))
            self.enc_man.select_ntrodes(message.ntrode_list)

        elif isinstance(message, TurnOnDataStream):
            self.class_log.debug("Turn on data stream")
            self.enc_man.turn_on_datastreams()

        elif isinstance(message, binary_record.BinaryRecordCreateMessage):
            self.enc_man.set_record_writer_from_message(message)

        elif isinstance(message, realtime_base.TimeSyncInit):
            self.enc_man.begin_time_sync()

        elif isinstance(message, realtime_base.TimeSyncSetOffset):
            self.enc_man.update_offset(message.offset_time)

        elif isinstance(message, realtime_base.StartRecordMessage):
            self.enc_man.start_record_writing()

        elif isinstance(message, realtime_base.StopRecordMessage):
            self.enc_man.stop_record_writing()


class EncoderProcess(realtime_base.RealtimeProcess):
    def __init__(self, comm: MPI.Comm, rank, config):

        super().__init__(comm, rank, config)

        self.local_rec_manager = binary_record.RemoteBinaryRecordsManager(manager_label='state', local_rank=rank,
                                                                          manager_rank=config['rank']['supervisor'])

        self.mpi_send = EncoderMPISendInterface(comm=comm, rank=rank, config=config)

        if self.config['datasource'] == 'simulator':
            spike_interface = simulator_process.SimulatorRemoteReceiver(comm=self.comm,
                                                                        rank=self.rank,
                                                                        config=self.config,
                                                                        datatype=datatypes.Datatypes.SPIKES)

            pos_interface = simulator_process.SimulatorRemoteReceiver(comm=self.comm,
                                                                      rank=self.rank,
                                                                      config=self.config,
                                                                      datatype=datatypes.Datatypes.LINEAR_POSITION)

        self.enc_man = RStarEncoderManager(rank=rank,
                                           config=config,
                                           local_rec_manager=self.local_rec_manager,
                                           send_interface=self.mpi_send,
                                           spike_interface=spike_interface,
                                           pos_interface=pos_interface)

        self.mpi_recv = EncoderMPIRecvInterface(comm=comm, rank=rank, config=config, encoder_manager=self.enc_man)

        self.terminate = False

        # First Barrier to finish setting up nodes
        self.comm.Barrier()

    def trigger_termination(self):
        self.terminate = True

    def main_loop(self):

        try:
            while not self.terminate:
                self.mpi_recv.__next__()
                self.enc_man.process_next_data()

        except StopIteration as ex:
            self.class_log.info('Terminating EncodingProcess (rank: {:})'.format(self.rank))

        self.class_log.info("Encoding Process reached end, exiting.")
