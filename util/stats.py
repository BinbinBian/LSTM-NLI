import collections
import cPickle
import matplotlib.pyplot as plt
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


    def recordCost(self, numEx, cost):
        self.cost.append((numEx, cost))
        self.logger.Log("Current cost: {0}".format(cost))


    def recordFinalTrainingTime(self, numEx):
        self.logger.Log("Training complete after processing {1} examples! "
                        "Total training time: {0} ".format((time.time() -
                                                    self.startTime), numEx))


    def saveFig(self, fileName, title, xLabel, yLabel, xCoord, yCoord):
        plt.plot(xCoord, yCoord)
        plt.xlabel(xLabel)
        plt.ylabel(yLabel)
        plt.title(title)
        plt.savefig(fileName)


    def recordFinalStats(self, numEx, trainAcc, devAcc, fileName):
        # TODO: Eventually support test accuracy computation as well
        self.totalNumEx = numEx
        self.acc["train"].append((numEx, trainAcc))
        self.acc["dev"].append((numEx, devAcc))
        self.logger.Log("Final training accuracy after {0} examples: {1}".format(numEx, trainAcc))
        self.logger.Log("Final validation accuracy after {0} examples: {1}".format(numEx, devAcc))

        # Pickle accuracy and cost
        with open(fileName+".pickle", 'w') as f:
            cPickle.dump(self.acc, f)
            cPickle.dump(self.cost, f)

        # Plot accuracies and loss function
        numEx = [stat[0] for stat in self.cost]
        cost = [stat[1] for stat in self.cost]

        trainEx = [stat[0] for stat in self.acc["train"]]
        trainAcc = [stat[1] for stat in self.acc["train"]]

        devEx = [stat[0] for stat in self.acc["dev"]]
        devAcc = [stat[1] for stat in self.acc["dev"]]

        self.saveFig(fileName+"_cost.png", "Cost vs. Num Examples", "Num Examples",
                     "Cost", numEx, cost)

        self.saveFig(fileName+"_trainAcc.png", "Train Accuracy vs. Num Examples", "Num Examples",
                     "Accuracy", trainEx, trainAcc)

        self.saveFig(fileName+"_devAcc.png", "Dev Accuracy vs. Num Examples", "Num Examples",
                     "Accuracy", devEx, devAcc)

        self.reset()