import collections
import cPickle
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import time

# Number of seconds in an hour
SEC_HOUR = 3600

class Stats(object):
    """
    General purpose object for recording statistics of model run including
    accuracies and cost values. Note, takes as input a logger object used
    for writing to disk periodically. Will also be used to plot appropriate graphs.
    """
    def __init__(self, logger, expName):
        self.startTime = time.time()
        self.acc = collections.defaultdict(list)
        self.cost = []
        self.logger = logger
        self.totalNumEx = 0
        self.expName = expName


    def reset(self):
        self.acc.clear()
        self.cost = []
        self.totalNumEx = 0


    def recordAcc(self, numEx, acc, dataset="train"):
        self.acc[dataset].append((numEx, acc))
        self.logger.Log("Current " + dataset + " accuracy after {0} examples:"
                                               " {1}".format(numEx, acc))
        if dataset == "train":
            ex = self.getTrainEx()
            acc = self.getTrainAcc()
        elif dataset == "dev":
            ex = self.getDevEx()
            acc = self.getDevAcc()

        self.plotAndSaveFig(self.expName+"_"+dataset+"Acc.png", dataset +
                            "Accuracy vs. Num Examples", "Num Examples", dataset +
                            " Accuracy", ex, acc)


    def recordCost(self, numEx, cost):
        self.cost.append((numEx, cost))
        self.logger.Log("Current cost: {0}".format(cost))
        numEx = self.getNumEx()
        cost = self.getCost()
        self.plotAndSaveFig(self.expName+"_cost.png", "Cost vs. Num Examples", "Num Examples",
                     "Cost", numEx, cost)


    def recordFinalTrainingTime(self, numEx):
        self.logger.Log("Training complete after processing {1} examples! "
                        "Total training time: {0} ".format((time.time() -
                                                    self.startTime)/SEC_HOUR, numEx))


    def plotAndSaveFig(self, fileName, title, xLabel, yLabel, xCoord, yCoord):
        plt.plot(xCoord, yCoord)
        plt.xlabel(xLabel)
        plt.ylabel(yLabel)
        plt.title(title)
        plt.savefig(fileName)
        plt.clf()


    def getNumEx(self):
        return [stat[0] for stat in self.cost]


    def getCost(self):
        return [stat[1] for stat in self.cost]


    def getTrainEx(self):
        return [stat[0] for stat in self.acc["train"]]


    def getTrainAcc(self):
        return [stat[1] for stat in self.acc["train"]]


    def getDevEx(self):
        return [stat[0] for stat in self.acc["dev"]]


    def getDevAcc(self):
        return [stat[1] for stat in self.acc["dev"]]


    def recordFinalStats(self, numEx, trainAcc, devAcc):
        # TODO: Eventually support test accuracy computation as well
        self.totalNumEx = numEx
        self.acc["train"].append((numEx, trainAcc))
        self.acc["dev"].append((numEx, devAcc))
        self.logger.Log("Final training accuracy after {0} examples: {1}".format(numEx, trainAcc))
        self.logger.Log("Final validation accuracy after {0} examples: {1}".format(numEx, devAcc))

        # Pickle accuracy and cost
        with open(self.expName+".pickle", 'w') as f:
            cPickle.dump(self.acc, f)
            cPickle.dump(self.cost, f)

        # Plot accuracies and loss function
        numEx = self.getNumEx()
        cost = self.getCost()

        trainEx = self.getTrainEx()
        trainAcc = self.getTrainAcc()

        devEx = self.getDevEx()
        devAcc = self.getDevAcc()

        self.plotAndSaveFig(self.expName+"_cost.png", "Cost vs. Num Examples", "Num Examples",
                     "Cost", numEx, cost)

        self.plotAndSaveFig(self.expName+"_trainAcc.png", "Train Accuracy vs. Num Examples", "Num Examples",
                     "Accuracy", trainEx, trainAcc)

        self.plotAndSaveFig(self.expName+"_devAcc.png", "Dev Accuracy vs. Num Examples", "Num Examples",
                     "Accuracy", devEx, devAcc)

        self.reset()
