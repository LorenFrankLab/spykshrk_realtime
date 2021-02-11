#!/bin/bash
#
#./run_script_kevin.sh 8 ./config/kevin_hostfile.mpi test_single_modified.json

PYTHON=python3

sudo $PYTHON setup.py develop
env MPICC=/usr/bin/mpicc.openmpi sudo $PYTHON -m pip install mpi4py
echo ''

#mpiexec.openmpi -np 6 -hostfile ./config/kevin_hostfile.mpi -bind-to hwthread python3 -m spykshrk.realtime.simulator --config ./config/test_single_modified.json
time mpiexec.openmpi -np $1 -hostfile $2 -bind-to hwthread $PYTHON -m spykshrk.realtime.simulator --config $3

time $PYTHON -m spykshrk.realtime.postprocessing.rec_merge_hdf5 --config $3
