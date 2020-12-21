# import trodes.FSData.fsDataMain as fsDataMain

from spykshrk.realtime import main_process, ripple_process, encoder_process, decoder_process
from spykshrk.realtime.simulator import simulator_process
import datetime
import logging
import logging.config
import cProfile
import sys
import os.path
import getopt
from mpi4py import MPI
from time import sleep
import numpy as np

import time
import json

from spikegadgets import trodesnetwork as tnp

class PythonClient(tnp.AbstractModuleClient):
    def __init__(self, config, rank):
        super().__init__("PythonRank"+str(rank), config['trodes_network']['address'],config['trodes_network']['port'])
        self.rank = rank
        self.registered = False
    def registerTerminateCallback(self, callback):
        self.terminate = callback
        self.registered = True

    # def recv_acquisition(self, command, timestamp):
        # if command == tnp.acq_STOP and self.registered:

    def recv_quit(self):
        self.terminate()


def main(argv):
    # parse the command line arguments
    try:
        opts, args = getopt.getopt(argv, "", ["config="])
    except getopt.GetoptError:
        logging.error('Usage: ...')
        sys.exit(2)

    # print(argv)
    # print(opts)
    for opt, arg in opts:
        if opt == '--config':
            config_filename = arg

    config = json.load(open(config_filename, 'r'))

    # setup MPI
    comm = MPI.COMM_WORLD           # type: MPI.Comm
    #comm.Barrier()
    size = comm.Get_size()
    rank = comm.Get_rank()
    name = MPI.Get_processor_name()

    # this hold the processes here until all MPI ahve started and then the sleep statment
    # staggers the startup for each rank so the trodes network will intialize one node at a time in order
    #for proc_rank in np.arange(0,size):
    # comm.Barrier()
    # # for drought cluster
    # #time.sleep(10+rank*3)

    # # for greenflash local spykshrk
    # time.sleep(1+rank*0.5)
    # #if proc_rank == rank:
    # print('got past barrier, rank = ',rank)
      


    # setup logging
    logging.config.dictConfig({
        'version': 1,
        'disable_existing_loggers': False,
        'formatters': {
            'simple': {
                'format': ('%(asctime)s.%(msecs)03d [%(levelname)s] '
                           '(MPI-{:02d}) %(threadName)s %(name)s: %(message)s').format(rank),
                'datefmt': '%H:%M:%S',
            },
        },
        'handlers': {
            'console': {
                'level': 'DEBUG',
                'class': 'logging.StreamHandler',
                'formatter': 'simple',
            },
            # 'debug_file_handler': {
            #     'class': 'spykshrk.realtime.realtime_logging.MakeFileHandler',
            #     'level': 'DEBUG',
            #     'formatter': 'simple',
            #     'filename': ('log/{date_str}_debug.log/{date_str}_MPI-{rank:02d}_debug.log'.
            #                  format(date_str=datetime.datetime.now().strftime('%Y-%m-%dT%H%M'),
            #                         rank=rank)),
            #     'encoding': 'utf8',
            # }
        },
        'loggers': {
            '': {
                'handlers': ['console'],
                'level': 'NOTSET',
                'propagate': True,
            }
        }
    })

    # logging.info('my name {}, my rank {}'.format(name, rank))

    # set up. may need to lengthen the amount of time for each process to instantiate,
    # initialize, and subscribe to appropriate streams
    comm.Barrier()
    time.sleep(rank*0.5)
    #t1 = time.time() + (rank*3)
    #while time.time() < t1:
    #    pass
    logging.info(f'Pre network client, rank: {rank}')
    network = PythonClient(config, rank)
    logging.info(f'Past network client, rank: {rank}')
    if network.initialize() != 0:
        logging.info(f"Network could not successfully initialize, rank: {rank}")
        del network
        quit()
    config['trodes_network']['networkobject'] = network
    logging.info(f'Past network initialize, rank {rank}')
    datastream = network.subscribeSpikesData(300, ['1,0'])
    datastream.initialize()
    buf = datastream.create_numpy_array()
    logging.info(f"Streams set up and buffer created, rank {rank}")

    # test whether the clients remain connected
    comm.Barrier()
    time.sleep(rank*0.5)
    t = 3
    logging.info(f"Simulating main loop running for {t} seconds, rank {rank}")
    time.sleep(t)

    # clean up
    comm.Barrier()
    time.sleep(rank)
    logging.info(f"Shutting down connections, rank {rank}")
    network.closeConnections()
    del network
    exit(0)
    


logging.info('Starting up main')
main(sys.argv[1:])
logging.info('Done with main')