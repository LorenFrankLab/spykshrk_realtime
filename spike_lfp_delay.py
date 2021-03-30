import time
import json
import datetime
import logging
import logging.config
import sys
import os.path
import getopt
from mpi4py import MPI

logging.basicConfig(level=logging.INFO)
from zmq import ZMQError
import trodesnetwork as tn
import numpy as np
import matplotlib.pyplot as plt

from spykshrk.realtime import utils
from spykshrk.realtime.trodes_data import TrodesNetworkDataReceiver
from spykshrk.realtime.datatypes import Datatypes

class TriggerClient(object):
    def __init__(self, stub, config):
        server_address = utils.get_network_address(config)
        self.acq_sub = tn.TrodesAcquisitionSubscriber(server_address=server_address)
        self.stub = stub
        self.started = False
        
    def __next__(self):
        try:
            data = self.acq_sub.receive(noblock=True)

            # only start up once
            if ('play' in data['command'] or 'record' in data['command']) and not self.started:
                self.stub.start()
                self.started = True
            if 'stop' in data['command']: # 'stop' for playback, 'stoprecord' for recording
                self.stub.trigger_termination()
        except ZMQError:
            pass


class EncoderStub(object):

    def __init__(self, comm, rank, config, ntid):
        self.comm = comm
        self.rank = rank
        self.config = config
        self.trigger_client = TriggerClient(self, config)
        self.spike_recv = TrodesNetworkDataReceiver(comm, rank, config, Datatypes.SPIKES)
        self.ntid = ntid
        self.spike_recv_setup = False
        self.loop = True

    def start(self):
        self.spike_recv.register_datatype_channel(self.ntid)
        self.spike_recv.start_all_streams()
        self.spike_recv_setup = True

    def main_loop(self):
        while self.loop:
            self.trigger_client.__next__()

            if self.spike_recv_setup:
                rv = self.spike_recv.__next__()
                if rv is not None:
                    msg = rv[0]
                    self.comm.send(msg.timestamp, dest=self.config['rank']['decoder'][0], tag=0)

    def save_data(self):
        pass

    def trigger_termination(self):
        self.loop = False


class DecoderStub(object):
    def __init__(self, comm, rank, config):
        self.comm = comm
        self.rank = rank
        self.config = config

        self.trigger_client = TriggerClient(self, config)
        self.lfp_recv = TrodesNetworkDataReceiver(comm, rank, config, Datatypes.LFP)
        self.lfp_recv_setup = False
        self.loop = True

        self.req = self.comm.irecv(tag=0)

        self.timestamp_ct = 0
        self.timestamp = 0

        self.time_bin_size = self.config['pp_decoder']['bin_size']
        self.decoder_bin_delay = self.config['pp_decoder']['bin_delay']

        self.lb = 10000000000
        self.spike_ct = 0
        self.dropped_spike_ct = 0

        self.diffs = np.zeros(1000000)
        self.diff_ind = 0

    def start(self):
        self.lfp_recv.register_datatype_channel(1)
        self.lfp_recv.start_all_streams()
        self.lfp_recv_setup = True
    
    def recv_from_encoder(self):
        ready, data = self.req.test()

        if ready:
            self.req = self.comm.irecv(tag=0)
            return data

    def main_loop(self):
        while self.loop:
            self.trigger_client.__next__()

            if self.lfp_recv_setup:

                # make sure enough space to store timestamp diffs
                if self.diff_ind == self.diffs.shape[0]:
                    self.diffs = np.hstack((self.diffs, np.zeros(self.diffs.shape[0])))
                
                res = self.recv_from_encoder()
                if res is not None:
                    self.diffs[self.diff_ind] = (self.timestamp - res) / 30
                    self.diff_ind += 1
                    self.spike_ct += 1

                    if res < self.lb:
                        print(f"Dropped spike! Timestamp difference: {(self.timestamp - res) / 30:.1f} ms")
                        self.dropped_spike_ct += 1

                rv = self.lfp_recv.__next__()
                if rv is not None:
                    self.timestamp = rv[0].timestamp
                    self.timestamp_ct += 1
                
                # schedule update
                if self.timestamp_ct % (self.time_bin_size/20) == 0:
                    self.lb = self.timestamp - self.decoder_bin_delay * self.time_bin_size

    def save_data(self):
        logging.info("Saving data")
        x = self.diffs[:self.diff_ind]
        np.save('lfp_spike_ts_diffs', x)

    def trigger_termination(self):
        self.loop = False


def main(argv):
    
    # parse the command line arguments
    try:
        opts, args = getopt.getopt(argv, "", ["config="])
    except getopt.GetoptError:
        logging.error('Usage: ...')
        sys.exit(2)

    for opt, arg in opts:
        if opt == '--config':
            config_filename = arg

    config = json.load(open(config_filename, 'r'))

    # setup MPI
    comm = MPI.COMM_WORLD  # type: MPI.Comm
    size = comm.Get_size()
    rank = comm.Get_rank()
    name = MPI.Get_processor_name()


    if rank in config["rank"]["encoders"]:
        process = EncoderStub(comm, rank, config, config["trodes_network"]["decoding_tetrodes"][rank])
    else:
        process = DecoderStub(comm, rank, config)

    comm.Barrier()
    time.sleep(0.001*rank)
    try:
        logging.info(f"Rank {rank} starting main loop")
        process.main_loop()
        logging.info(f"Rank {rank} finished main loop, exiting main")
    except KeyboardInterrupt:
        process.save_data()

if __name__ == "__main__":
    main(sys.argv[1:])
