import time
import json
import datetime
import logging
import logging.config
import sys
import os.path
import getopt
from mpi4py import MPI

from spykshrk.realtime import (main_process, ripple_process, encoder_process,
                               decoder_process, gui_process)
from spykshrk.realtime.simulator import simulator_process

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
    prefix = config['file']['prefix']
    # the last process to reach this point will determine what the prefix is set to.
    # all record files will end up having the same prefix
    config['file']['prefix'] = datetime.datetime.now().strftime('%Y%m%d_%H%M%S_') + prefix

    # setup MPI
    comm = MPI.COMM_WORLD  # type: MPI.Comm
    size = comm.Get_size()
    rank = comm.Get_rank()
    name = MPI.Get_processor_name()
      
    logging.info(f"Rank {rank} starting up in main")

    # setup logging
    logging.config.dictConfig({
        'version': 1,
        'disable_existing_loggers': False,
        'formatters': {
            'simple': {
                'format': ('%(asctime)s.%(msecs)03d [%(levelname)s] '
                           '(MPI-{:02d}) [PID: %(process)d] %(name)s: %(message)s').format(rank),
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

    if size == 1:
        # MPI is not running or is running on a single node.  Single processor mode
        pass

    # Make sure output directory exists
    os.makedirs(os.path.join(config['files']['output_dir']), exist_ok=True)
    
    # Save config to output
    output_config = open(os.path.join(config['files']['output_dir'], config['files']['prefix'] + '.config.json'), 'w')
    json.dump(config, output_config, indent=4)

    process = None
    if rank == config["rank"]["supervisor"]:
        process = main_process.MainProcess(comm=comm, rank=rank, config=config)
    elif rank in config["rank"]["decoder"]: # Note: expect this to be a list
        process = decoder_process.DecoderProcess(comm=comm, rank=rank, config=config)
    elif rank in config["rank"]['ripples']:
        process = ripple_process.RippleProcess(comm, rank, config=config)
    elif rank in config["rank"]["encoders"]:
        process = encoder_process.EncoderProcess(comm, rank, config=config)
    elif rank == config["rank"]["gui"]:
        process = gui_process.GuiProcess(comm, rank, config)
    else:
        raise Exception(f"Could not assign rank {rank} to appropriate process")

    comm.Barrier()
    if process is not None:
        time.sleep(0.1*rank)
        logging.info(f"Rank {rank} starting main loop")
        process.main_loop()
        logging.info(f"Rank {rank} finished main loop, exiting main")
    else:
        logging.info(f"Rank {rank} could not start up in main correctly")

if __name__ == "__main__":
    main(sys.argv[1:])
