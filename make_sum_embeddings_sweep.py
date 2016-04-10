# Create a script to run a random hyperparameter search.
import argparse
import copy
import random
import numpy as np
import os

LIN = "LIN"
EXP = "EXP"

parser = argparse.ArgumentParser()
parser.add_argument("--host", type=str, default="", help="specific machine to run on")
parser.add_argument("--queue", type=str, default="jag")
args = parser.parse_args()

scrPath = "/afs/cs.stanford.edu/u/meric/scr/meric/LSTM-NLI"
# Instructions: Configure the variables in this block, then run
# the following on a machine with qsub access:
# python make_sweep.py > my_sweep.sh
# bash my_sweep.sh

# - #

# Non-tunable flags that must be passed in.

FIXED_PARAMETERS = {
    "trainData":    "/scr/nlp/data/snli_1.0/snli_1.0_train.jsonl",
    "trainDataStats": "/afs/cs.stanford.edu/u/meric/scr/meric/LSTM-NLI/data/train_dataStats.json",
    "valData":    "/scr/nlp/data/snli_1.0/snli_1.0_dev.jsonl",
    "valDataStats": "/afs/cs.stanford.edu/u/meric/scr/meric/LSTM-NLI/data/dev_dataStats.json",
    "testData": "/scr/nlp/data/snli_1.0/snli_1.0_test.jsonl",
    "testDataStats": "/afs/cs.stanford.edu/u/meric/scr/meric/LSTM-NLI/data/test_dataStats.json",
    "embedData": "/scr/nlp/data/glove_vecs/glove.6B.50d.txt",
    #"unrollSteps": "20",
    #"clipping_max_value":  "3.0",
    "batchSize":  "32",
    #"numExamplesToTrain": "-1",
    "numEpochs": "50",
    "regPenalty": "l2",
    "denseDim": 200,
    "numDense": 2
}

# Tunable parameters.
SWEEP_PARAMETERS = {
    "learnRate":      (EXP, 0.0005, 0.01),
    "regCoeff":         (EXP, 5e-7, 1e-4),
    "unrollSteps":      (LIN, 16, 23),
    #"L2regularization":   		  (EXP, 5e-7, 1e-4),
}

sweep_runs = 6
queue = args.queue

# - #
print "# NUM RUNS: " + str(sweep_runs)
print "# SWEEP PARAMETERS: " + str(SWEEP_PARAMETERS)
print "# FIXED_PARAMETERS: " + str(FIXED_PARAMETERS)
print

for run_id in range(sweep_runs):
    params = {}
    params.update(FIXED_PARAMETERS)
    for param in SWEEP_PARAMETERS:
        config = SWEEP_PARAMETERS[param]
        t = config[0]
        mn = config[1]
        mx = config[2]

        r = random.uniform(0, 1)
        if t == EXP:
            lmn = np.log(mn)
            lmx = np.log(mx)
            sample = np.exp(lmn + (lmx - lmn) * r)
        else:
            sample = mn + (mx - mn) * r

        if isinstance(mn, int):
            sample = int(round(sample, 0))

        params[param] = sample

    name = ""
    flags = ""
    for param in params:
        value = params[param]
        val_str = ""
        flags += " --" + param + " " + str(value)
        if param not in FIXED_PARAMETERS:
            if isinstance(value, int):
                val_disp = str(value)
            else:
                val_disp = "%.2g" % value
            name += "-" + param + val_disp


    batchSize = "batchSize" + FIXED_PARAMETERS["batchSize"]
    numEpochs = "numEpochs" +  FIXED_PARAMETERS["numEpochs"]
    learnRate = "learnRate" + "%.2g" %params["learnRate"]
    #L2reg = "L2regularization" + "%.2g" %params["L2regularization"]

    experimentName = "sweep_sum_embed_" + batchSize + "_" + numEpochs + "_" + learnRate
    experimentName = scrPath + "/log/" + experimentName + ".log"
    #flags += " --logPath" + " " + logPath
    flags += " --expName " + experimentName

    host = ""
    if args.host != "":
        host = "-l host=" + args.host

    print "export SUM_EMBED_FLAGS=\"" + flags + "\"; qsub -v SUM_EMBED_FLAGS train_sum_embed.sh " + host + " -q " + queue
    print
