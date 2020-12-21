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
            'debug_file_handler': {
                'class': 'spykshrk.realtime.realtime_logging.MakeFileHandler',
                'level': 'DEBUG',
                'formatter': 'simple',
                'filename': ('log/{date_str}_debug.log/{date_str}_MPI-{rank:02d}_debug.log'.
                             format(date_str=datetime.datetime.now().strftime('%Y-%m-%dT%H%M'),
                                    rank=rank)),
                'encoding': 'utf8',
            }
        },
        'loggers': {
            '': {
                'handlers': ['console', 'debug_file_handler'],
                'level': 'NOTSET',
                'propagate': True,
            }
        }
    })

    logging.info('my name {}, my rank {}'.format(name, rank))

    if size == 1:
        # MPI is not running or is running on a single node.  Single processor mode
        pass

    #print('Rank {}: sleeping for {} sec'.format(rank, rank*2))
    #time.sleep(rank*0.5)
    # Make sure output directory exists
    os.makedirs(os.path.join(config['files']['output_dir']), exist_ok=True)
    # Save config to output
    logging.info(f'In __main__.py: pre json.dump, rank: {rank}')

    output_config = open(os.path.join(config['files']['output_dir'], config['files']['prefix'] + '.config.json'), 'w')
    json.dump(config, output_config, indent=4)

    logging.info(f'In __main__.py: past json.dump, rank: {rank}')

    comm.Barrier()
    # MEC and LMF: edited this to add a delay between startup of each trodes network
    #time.sleep(rank/10.0)
    #time.sleep(rank)
    if config['datasource'] == 'trodes':
        if rank != config['rank']['supervisor']:
            time.sleep(3+rank*1.5)
            # configure trodes network highfreqdatatypes (main supervisor process has own client)
            logging.info(f'In __main__.py: pre network client, rank: {rank}')
            network = PythonClient(config, rank)
            logging.info(f'In __main__.py: past network client, rank: {rank}')
            if network.initialize() != 0:
                logging.info(f"Network could not successfully initialize, rank: {rank}")
                del network
                quit()
            config['trodes_network']['networkobject'] = network
            logging.info(f'In __main__.py: past network initialize, rank {rank}')
        # supervisor still sets up network client, but does so in its constructor. 
        # already has a delay. need to fix
        else:
            main_proc = main_process.MainProcess(comm=comm, rank=rank, config=config)


    # set up MPI nodes
    if rank == config['rank']['supervisor']:
        # don't need to instantiate process--already done previously.
        # but as I noted earlier, this needs a more robust fix
        logging.info(f"main process start in main, rank: {rank}")
        if config['datasource'] == 'trodes':
            main_proc.main_loop()
        else:
            main_proc = main_process.MainProcess(comm=comm, rank=rank, config=config)
            main_proc.main_loop()

    elif rank in config['rank']['ripples']:
        #time.sleep(0.1)
        logging.info(f'ripple process start in main, rank: {rank}')
        ripple_proc = ripple_process.RippleProcess(comm, rank, config=config)
        if config['datasource'] == 'trodes':
            network.registerTerminateCallback(ripple_proc.trigger_termination)
        ripple_proc.main_loop()
        logging.info(f'ripple process main loop ended, rank: {rank}')

    elif rank in config['rank']['encoders']:
        #time.sleep(0.1)
        logging.info(f'encoder process start in main, rank: {rank}')
        encoding_proc = encoder_process.EncoderProcess(comm, rank, config=config)
        if config['datasource'] == 'trodes':
            network.registerTerminateCallback(encoding_proc.trigger_termination)
        encoding_proc.main_loop()
        logging.info(f'encoder process main loop ended, rank: {rank}')

    # original: 1 decoder
    #if rank == config['rank']['decoder']:
    #    #time.sleep(0.1)
    #    decoding_proc = decoder_process.DecoderProcess(comm=comm, rank=rank, config=config)
    #    if config['datasource'] == 'trodes':
    #        network.registerTerminateCallback(decoding_proc.trigger_termination)
    #    decoding_proc.main_loop()
    #    print('decoder process main loop ended:',rank)

    # for more than 1 decoder
    elif rank in config['rank']['decoder']:
        #time.sleep(0.1)
        logging.info(f'decoder process start in main, rank: {rank}')
        decoding_proc = decoder_process.DecoderProcess(comm=comm, rank=rank, config=config)
        if config['datasource'] == 'trodes':
            network.registerTerminateCallback(decoding_proc.trigger_termination)
        decoding_proc.main_loop()
        logging.info(f'decoder process main loop ended, rank: {rank}')

    elif rank == config['rank']['simulator']:
        logging.info(f"simulator process start in main, rank: {rank}")
        simulator_proc = simulator_process.SimulatorProcess(comm, rank, config=config)
        simulator_proc.main_loop()
        logging.info(f"simulator process main loop ended, rank: {rank}")

    else:
        raise Exception(f"Error assigning rank {rank} to appropriate process")

    # clean up
    if config['datasource'] == 'trodes':
        if rank == config['rank']['supervisor']:
            main_proc.networkclient.closeConnections()
            del main_proc.networkclient
            logging.info(f"Main process Trodes network deleted, rank: {rank}")
        else:
            network.closeConnections()
            del network
            logging.info(f'Non main process Trodes network deleted, rank: {rank}')
    exit(0)
    


logging.info('Starting up main')
main(sys.argv[1:])
logging.info('Done with main')