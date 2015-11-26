#!/bin/bash

### Generic job script for all experiments.

# Usage example:
# export REMBED_FLAGS="--learning_rate 0.01 --batch_size 256"; qsub -v REMBED_FLAGS train_rembed_classifier.sh

### The below are not comments, but actual PBS flags:
#PBS -l walltime=99:00:00	### Die after four-ish days
#PBS -l mem=16000MB   ### Request 16G RAM. We don't need it, but it helps make sure we don't wind up on machines with overloaded GPUs.
#PBS -q jag   ### Run on the jagupard* queue

# Change to the submission directory.
cd $PBS_O_WORKDIR
echo Lauching from working directory: $PBS_O_WORKDIR
echo Flags: $LSTM_NLI_FLAGS
DEVICE=`python pick_gpu.py`

# Log what we're running and where.
echo `hostname` - $PBS_JOBID - $LSTM_NLI_FLAGS - $DEVICE - at `git log --pretty=format:'%h' -n 1` >> ~/lstm_nli_machine_assignments.txt

THEANO_FLAGS=allow_gc=False,exception_verbosity=high,cuda.root=/usr/bin/cuda,warn_float64=warn,device=$DEVICE,floatX=float32 python -m run.py $LSTM_NLI_FLAGS
