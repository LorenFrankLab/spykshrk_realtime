#!/bin/bash

python setup.py develop

echo ''

time mpiexec -np $1 -hostfile $2 -bind-to core python -m spykshrk.realtime.simulator --config $3

time python -m spykshrk.realtime.postprocessing.rec_merge_hdf5 --config $3
