import cProfile
import logging
import os
import time
from abc import ABCMeta, abstractmethod
from enum import Enum, IntEnum

from mpi4py import MPI

import spykshrk.realtime.binary_record as binary_record
from spykshrk.realtime import realtime_logging as rt_logging
from spykshrk.realtime.realtime_logging import LoggingClass, PrintableMessage
from spykshrk.realtime.timing_system import TimingMessage


class MPIMessageTag(IntEnum):
    COMMAND_MESSAGE = 1
    FEEDBACK_DATA = 2
    TIMING_MESSAGE = 3

    SIMULATOR_LFP_DATA = 10
    SIMULATOR_SPK_DATA = 11
    SIMULATOR_POS_DATA = 12
    SIMULATOR_LINPOS_DATA = 13

    SPIKE_DECODE_DATA = 20
    STIM_DECISION = 21
    POSTERIOR = 22
    VEL_POS = 23


class RecordIDs(IntEnum):
    RIPPLE_STATE = 1
    ENCODER_QUERY = 2
    ENCODER_OUTPUT = 3
    DECODER_OUTPUT = 4
    DECODER_MISSED_SPIKES = 5
    LIKELIHOOD_OUTPUT = 6
    OCCUPANCY = 7

    STIM_STATE = 10
    STIM_LOCKOUT = 11
    STIM_MESSAGE = 12

    TIMING = 100


class RealtimeMPIClass(LoggingClass):
    def __init__(self, comm: MPI.Comm, rank, config, *args, **kwargs):
        self.comm = comm
        self.rank = rank
        self.config = config
        super(RealtimeMPIClass, self).__init__(*args, **kwargs)


class DataSourceError(RuntimeError):
    pass


class BinaryRecordBaseError(RuntimeError):
    pass


class DataSourceReceiver(RealtimeMPIClass, metaclass=ABCMeta):
    """An abstract class that ranks should use to communicate between neural data sources.

    This class should not be instantiated, only its subclasses.

    This provides an abstraction layer for sources of neural data (e.g., saved file simulator, acquisition system)
    to pipe data (e.g., spikes, lfp, position) to ranks that request data for processing.  This is only an abstraction
    for a streaming data (e.g. sockets, MPI) and makes a number of assumptions:

    1. The type of data and 'channel' (e.g., electrode channels 1, 2, 3) can be streamed to different client processes
    and registered by a client one channel at a time

    2. The streams can be started and stopped arbitrarily after the connection is established (no rule if data is lost
    during pause)

    3. The connection is destroyed when the iterator stops.
    """

    @abstractmethod
    def __init__(self, comm, rank, config, datatype, *args, **kwds):
        """

        Args:
            comm:
            rank:
            config:
            datatype: The type of data to request to be streamed, specified by spykshrk.realtime.datatypes.Datatypes
            *args:
            **kwds:
        """
        super(DataSourceReceiver, self).__init__(comm=comm, rank=rank, config=config, *args, **kwds)
        self.datatype = datatype

    @abstractmethod
    def register_datatype_channel(self, channel):
        """

        Args:
            channel: The channel of the data type to stream

        Returns:
            None

        """
        pass

    @abstractmethod
    def start_all_streams(self):
        pass

    @abstractmethod
    def stop_all_streams(self):
        pass

    @abstractmethod
    def stop_iterator(self):
        pass

    @abstractmethod
    def __next__(self):
        pass


class TimingSystemBase(LoggingClass):

    def __init__(self, *args, **kwds):
        self.time_writer = None
        self.rank = kwds['rank']
        super(TimingSystemBase, self).__init__(*args, **kwds)

    def set_timing_writer(self, time_writer):
        self.time_writer = time_writer

    def write_timing_message(self, timing_msg: TimingMessage):
        if self.time_writer is not None:
            timing_msg.record_time(self.rank)
            self.time_writer.write_timing_message(timing_msg=timing_msg)
        else:
            self.class_log.warning('Tried writing timing message before timing file created.')


class NoneRecordBase(LoggingClass):
    """
    A dummy class for record writing that does nothing.  Can be used to replace BinaryRecordBase
    when record writing needs to be disabled.
    """

    def __init__(self, *args, **kwds):
        super().__init__(*args, **kwds)

    def get_record_register_messages(self):
        return []

    def set_record_writer_from_message(self, create_message):
        pass

    def set_record_writer(self, rec_writer):
        pass

    def start_record_writing(self):
        pass

    def stop_record_writing(self):
        pass

    def close_record(self):
        pass

    def write_record(self, rec_id, *args):
        pass


class BinaryRecordBase(LoggingClass):
    def __init__(self, *args, **kwds):
        super().__init__(*args, **kwds)
        self.rank = kwds['rank']
        self.local_rec_manager = kwds['local_rec_manager']
        self.send_interface = kwds['send_interface']
        try:
            self.rec_ids = kwds['rec_ids']
        except KeyError:
            self.rec_ids = []
        try:
            self.rec_labels = kwds['rec_labels']
        except KeyError:
            self.rec_labels = []
        try:
            self.rec_formats = kwds['rec_formats']
        except KeyError:
            self.rec_formats = []

        kwds_none = [not self.rec_ids, not self.rec_labels, not self.rec_formats]
        if any(kwds_none) and not all(kwds_none):
            raise binary_record.BinaryRecordsError("BinaryRecordBase missing **kwd arguements, "
                                                   "rec_ids, rec_labels, rec_formats must be populated or all"
                                                   "empty lists.")

        self.rec_writer = None    # type: binary_record.BinaryRecordsFileWriter
        self.rec_writer_enabled = False

        # Send binary record register message
        #self.send_interface.send_record_register_messages(self.get_record_register_messages())

    def setup_mpi(self):
        self.send_record_register_message()

    def send_record_register_message(self):

        # Send binary record register message
        self.send_interface.send_record_register_messages(self.get_record_register_messages())

    def get_record_register_messages(self):

        messages = []
        for rec_id, rec_label, rec_fmt in zip(self.rec_ids, self.rec_labels, self.rec_formats):
            messages.append(self.local_rec_manager.create_register_rec_type_message(rec_id=rec_id,
                                                                                    rec_labels=rec_label,
                                                                                    rec_struct_fmt=rec_fmt))

        return messages

    def set_record_writer_from_message(self, create_message):
        #self.class_log.info('Creating record from message {}'.format(create_message))
        self.class_log.info('Creating record from message.')
        self.set_record_writer(self.local_rec_manager.create_writer_from_message(create_message))

    def set_record_writer(self, rec_writer):
        self.class_log.info('Setting record writer')
        if self.rec_writer_enabled:
            raise BinaryRecordBaseError('Can\'t change writer file while recording is on going!')
        self.rec_writer = rec_writer

    def start_record_writing(self):
        self.class_log.info('Starting record writer.')
        if self.rec_writer:
            if not self.rec_writer.closed:
                self.rec_writer_enabled = True
            else:
                raise BinaryRecordBaseError('Can\'t start recording, file not open!')
        else:
            raise BinaryRecordBaseError('Can\'t start recording, record file never set!')

    def stop_record_writing(self):
        self.rec_writer_enabled = False

    def close_record(self):
        self.stop_record_writing()
        if self.rec_writer:
            self.rec_writer.close()

    def write_record(self, rec_id, *args):
        if rec_id not in self.rec_ids:
            raise binary_record.BinaryRecordsError('{} attempted to write record whose id {} it did not register. '
                                                   'Rcord: {}'.format(self.__class__.__name__, rec_id, args))
        if self.rec_writer_enabled and not self.rec_writer.closed:
            self.rec_writer.write_rec(rec_id, *args)
            return True
        return False


class BinaryRecordBaseWithTiming(BinaryRecordBase):
    def __init__(self, *args, **kwds):

        self.rec_ids = kwds.setdefault('rec_ids', [])
        self.rec_labels = kwds.setdefault('rec_labels', [])
        self.rec_formats = kwds.setdefault('rec_formats', [])

        self.rec_ids.append(RecordIDs.TIMING)
        self.rec_labels.append(['timestamp', 'elec_grp_id',
                                'rank', 'label', 'datatype', 'wtime_raw', 'wtime_adj'])
        self.rec_formats.append('qhb20shdd')

        self.offset_time = 0

        super(BinaryRecordBaseWithTiming, self).__init__(*args, **kwds)

    def record_timing(self, timestamp, elec_grp_id, datatype, label):
        if len(label) > 20:
            raise binary_record.BinaryRecordsError("Timing label {} too long, must be "
                                                   "10 characters or less.".format(label))

        time = MPI.Wtime()
        self.write_record(RecordIDs.TIMING, timestamp, elec_grp_id, self.rank, label.encode('utf-8'), datatype,
                          time, time + self.offset_time)

    # MEC silence message to terminal
    def sync_time(self):
        #self.class_log.debug("Begin time sync barrier ({}).".format(self.rank))
        self.send_interface.all_barrier()
        self.send_interface.send_time_sync_report(MPI.Wtime())
        #self.class_log.debug("Report post barrier time ({}).".format(self.rank))

    # MEC turn this off so i can look at terminal more easily
    def update_offset(self, offset_time):
        #self.class_log.debug("Updating time offset to {}".format(offset_time))
        self.offset_time = offset_time


class ExceptionLoggerWrapperMeta(type):
    """
        A metaclass that wraps the run() or main_loop() method so exceptions are logged.

        This metaclass is built to solve a very specific bug in MPI + threads: a race condition that sometimes
        prevents a thread's uncaught exception from being displayed to stderr before program stalls.

        This class also avoids the known issue with logging exception in threads using sys.excepthook, the hook
        needs to be set by the thread after it is started.
    """
    @staticmethod
    def exception_wrap(func):
        def outer(self):
            try:
                func(self)
            except Exception as ex:
                logging.exception(ex.args)
                # traceback.print_exc(file=sys.stdout)

        return outer

    def __new__(mcs, name, bases, attrs):
        if 'run' in attrs:
            attrs['run'] = mcs.exception_wrap(attrs['run'])
        if 'main_loop' in attrs:
            attrs['main_loop'] = mcs.exception_wrap(attrs['main_loop'])

        return super(ExceptionLoggerWrapperMeta, mcs).__new__(mcs, name, bases, attrs)

    def __init__(cls, name, bases, attrs):
        super(ExceptionLoggerWrapperMeta, cls).__init__(name, bases, attrs)


class ProfilerWrapperMeta(type):
    @staticmethod
    def profile_wrap(func):
        def outer(self):
            if self.enable_profiler:
                prof = cProfile.Profile()
                prof.runcall(func, self)
                prof.dump_stats(file=self.profiler_out_path)
            else:
                func(self)

        return outer

    def __new__(mcs, name, bases, attrs):
        if 'run' in attrs:
            attrs['run'] = mcs.profile_wrap(attrs['run'])
        if 'main_loop' in attrs:
            attrs['main_loop'] = mcs.profile_wrap(attrs['main_loop'])

        return super(ProfilerWrapperMeta, mcs).__new__(mcs, name, bases, attrs)

    def __init__(cls, name, bases, attrs):
        super(ProfilerWrapperMeta, cls).__init__(name, bases, attrs)


class RealtimeMeta(ExceptionLoggerWrapperMeta, ProfilerWrapperMeta):
    """ A metaclass that combines all coorperative metaclass features needed (wrapping the run/main_loop functions
    with cProfilers and catching unhandled exceptions.

    Care needs to be taken that if the meta classes are modifying attributes, each of those modifications is unique.
    This is an issue when chaining the wrapping of functions, the name for the wrapping function needs to be unique,
    e.g. ProfileWrapperMeta and ExceptionLoggerWrapperMeta cannot both use a wrapping function with the same name
    (wrap), they must be unique (exception_wrap and profile_wrap).

    """
    def __new__(mcs, name, bases, attrs):
        return super(RealtimeMeta, mcs).__new__(mcs, name, bases, attrs)

    def __init__(cls, name, bases, attrs):
        super(RealtimeMeta, cls).__init__(name, bases, attrs)


class RealtimeProcess(RealtimeMPIClass, metaclass=RealtimeMeta):

    def __init__(self, comm: MPI.Comm, rank, config, **kwds):

        super().__init__(comm=comm, rank=rank, config=config)

        self.enable_profiler = rank in self.config['rank_settings']['enable_profiler']
        self.profiler_out_path = os.path.join(config['files']['output_dir'], '{}.{:02d}.{}'.
                                              format(config['files']['prefix'],
                                                     rank,
                                                     config['files']['profile_postfix']))

    def main_loop(self):
        pass


class DataStreamIterator(LoggingClass, metaclass=ABCMeta):

    @abstractmethod
    def __init__(self):
        super().__init__()
        self.source_infos = {}
        self.source_handlers = []
        self.source_enabled_list = []

    @abstractmethod
    def turn_on_stream(self):
        pass

    @abstractmethod
    def turn_off_stream(self):
        pass

    def set_sources(self, source_infos):
        self.source_infos = source_infos

    def set_enabled(self, source_enable_list):
        self.source_enabled_list = source_enable_list

    @abstractmethod
    def __next__(self):
        pass


class TimeSyncInit(PrintableMessage):
    def __init__(self):
        pass


class TimeSyncReport(PrintableMessage):
    def __init__(self, time):
        self.time = time
        pass


class TimeSyncSetOffset(PrintableMessage):
    def __init__(self, offset_time):
        self.offset_time = offset_time
        pass


class TerminateErrorMessage(PrintableMessage):
    def __init__(self, message):
        self.message = message
        pass


class TerminateMessage(PrintableMessage):
    def __init__(self):
        pass


class EnableStimulationMessage(PrintableMessage):
    def __init__(self):
        pass


class DisableStimulationMessage(PrintableMessage):
    def __init__(self):
        pass


class StartRecordMessage(PrintableMessage):
    def __init__(self):
        pass


class StopRecordMessage(PrintableMessage):
    def __init__(self):
        pass


class CloseRecordMessage(PrintableMessage):
    def __init__(self):
        pass


class RequestStatusMessage(PrintableMessage):
    def __init__(self):
        pass


class ResetFilterMessage(PrintableMessage):
    def __init__(self):
        pass


class NumTrodesMessage(PrintableMessage):
    def __init__(self, num_ntrodes):
        self.num_ntrodes = num_ntrodes


class TurnOnLFPMessage(PrintableMessage):
    def __init__(self, lfp_enable_list):
        self.lfp_enable_list = lfp_enable_list


class TurnOffLFPMessage(PrintableMessage):
    def __init__(self):
        pass


class ChannelSelection(rt_logging.PrintableMessage):
    def __init__(self, ntrode_list):
        self.ntrode_list = ntrode_list

#MEC added
class RippleChannelSelection(rt_logging.PrintableMessage):
    def __init__(self, ripple_ntrode_list):
        self.ripple_ntrode_list = ripple_ntrode_list

class TurnOnDataStream(rt_logging.PrintableMessage):
    def __init__(self):
        pass
