import collections
import time

class Stats(object):
    """
    General purpose object for recording statistics of model run including
    accuracies and cost values. Note, takes as input a logger object used
    for writing to disk periodically. Will also be used to plot appropriate graphs.
    """
    def __init__(self, logger):
        self.startTime = time.time()
        self.acc = collections.defaultdict(list)
        self.cost = []
        self.logger = logger
        self.totalNumEx = 0


    def reset(self):
        self.acc.clear()
        self.cost = []
        self.totalNumEx = 0


    def recordAcc(self, numEx, acc, dataset="train"):
        self.acc[dataset].append((numEx, acc))
        self.logger.Log("Current " + dataset + " accuracy after {0} examples:"
                                               " {1}".format(numEx, acc))


    def recordCost(self, cost):
        self.cost.append(cost)
        self.logger.Log("Current cost: {0}".format(cost))


    def recordFinalTrainingTime(self, numEx):
        self.logger.Log("Training complete after processing {1} examples! "
                        "Total training time: {0} ".format((time.time() -
                                                    self.startTime), numEx))


    def recordFinalStats(self, numEx, trainAcc, devAcc):
        # TODO: Eventually support test accuracy computation as well
        self.totalNumEx = numEx
        self.acc["train"].append((numEx, trainAcc))
        self.acc["dev"].append((numEx, devAcc))
        self.logger.Log("Final training accuracy after {0} examples: {1}".format(numEx, trainAcc))
        self.logger.Log("Final validation accuracy after {0} examples: {1}".format(numEx, devAcc))
        self.reset()