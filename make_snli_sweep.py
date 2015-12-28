# Create a script to run a random hyperparameter search.

import copy
import random
import numpy as np
import os

LIN = "LIN"
EXP = "EXP"

# Instructions: Configure the variables in this block, then run
# the following on a machine with qsub access:
# python make_sweep.py > my_sweep.sh
# bash my_sweep.sh

# - #

# Non-tunable flags that must be passed in.

FIXED_PARAMETERS = {
   # "data_type":     "snli",
   # "model_type":     "Model0",
    "trainData":    "/scr/nlp/data/snli_1.0/snli_1.0_train.jsonl",
    "trainDataStats": "/afs/cs.stanford.edu/u/meric/scr/LSTM-NLI/data/train_dataStats.json",
    "valData":    "/scr/nlp/data/snli_1.0/snli_1.0_dev.jsonl",
    "valDataStats": "/afs/cs.stanford.edu/u/meric/scr/LSTM-NLI/data/dev_dataStats.json",
    "testData": "/scr/nlp/data/snli_1.0/snli_1.0_test.jsonl",
    "testDataStats": "/afs/cs.stanford.edu/u/meric/scr/LSTM-NLI/data/test_dataStats.json",
    "embedData": "/scr/nlp/data/glove_vecs/glove.6B.50d.txt",
    "dimInput": "100",
    #"dimHidden": "64",
    #"unrollSteps": "20",
    #"clipping_max_value":  "3.0",
    "batchSize":  "128",
    "numExamplesToTrain": "-1",
    "numEpochs": "15"
}

# Tunable parameters.
SWEEP_PARAMETERS = {
    "learnRate":      (EXP, 0.00005, 0.001),
    "gradMax":         (LIN, 0.5, 10.),
    "unrollSteps":      (LIN, 8, 25),
    "dimHidden":        (LIN, 64, 512),
    "L2regularization":   		  (EXP, 5e-7, 1e-4),
    "dropoutRate":          (LIN, 0.4, 1.0)
}

sweep_runs = 6
queue = "jag"

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
    dimHidden = "dimHidden" + str(params["dimHidden"])
    learnRate = "learnRate" + "%.2g" %params["learnRate"]
    dropoutRate = "dropoutRate" + "%.2g" %params["dropoutRate"]
    L2reg = "L2regularization" + "%.2g" %params["L2regularization"]

    experimentName = "sweep_snli_" + batchSize + "_" + numEpochs + "_" + dimHidden + "_" + learnRate + "_" + L2reg + "_" + dropoutRate
    logPath = os.getcwd() + "/log/" + experimentName + ".log"
    flags += " --logPath" + " " + logPath

    print "export LSTM_NLI_FLAGS=\"" + flags + "\"; qsub -v LSTM_NLI_FLAGS train_LSTM_NLI.sh -q " + queue
    print
