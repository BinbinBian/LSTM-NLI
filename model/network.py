import numpy as np
import os
import theano
# Hacky way to ensure that theano can find NVCC compiler
os.environ["PATH"] += ":/usr/local/cuda/bin"

from util.afs_safe_logger import Logger

# Set random seed for deterministic runs
SEED = 100
np.random.seed(SEED)

currDir = os.path.dirname(os.path.dirname(__file__))


class Network(object):
    """
    Generic network class from which other specific model architectures will inherit.
    """
    def __init__(self, logPath, trainData, trainDataStats, valData, valDataStats,
                 testData, testDataStats):
        self.logger = Logger(log_path=logPath)
        # All layers in model
        self.layers = []

        self.trainData = trainData
        self.trainDataStats = trainDataStats
        self.valData = valData
        self.valDataStats = valDataStats
        self.testData = testData
        self.testDataStats = testDataStats


    def buildModel(self):
        raise NotImplementedError


    def printNetworkParams(self):
        """
        Print all params of network.
        """
        for layer in self.layers:
            self.logger.Log("Current parameter values for %s" %(layer.layerName))
            self.logger.Log("-" * 50)
            for pName, pValue in layer.params.iteritems():
                self.logger.Log(pName+" : "+str(np.asarray(pValue.eval())))

            self.logger.Log("-" * 50)


    def extractParams(self):
        """
        Extracts the numerical value of the model params and
        stored in model variable
        """
        for layer in self.layers:
            for paramName, paramVar in layer.params.iteritems():
                self.numericalParams[paramName] = paramVar.get_value()

        # TODO: Test that params are properly extracted


    def saveModel(self, modelFileName):
        """
        Saves the parameters of the model to disk.
        """
        with open(modelFileName, 'w') as f:
            np.savez(f, **self.numericalParams)


    def loadModel(self, modelFileName):
        """
        Loads the given model and sets the parameters of the network to the
        loaded parameter values
        :param modelFileName:
        """
        raise NotImplementedError


    def convertIdxToLabel(self, labelIdx):
        """
        Converts an idx to a label from our classification categories.
        :param idx:
        :return: List of all label categories
        """
        categories = ["entailment", "contradiction", "neutral"]
        labelCategories = []
        for idx in labelIdx:
            labelCategories.append(categories[idx])

        self.logger.Log("Labels of examples: {0}".format(labelCategories))

        return labelCategories

    def trainFunc(self):
        raise NotImplementedError


    def train(self):
        raise NotImplementedError


    def predictFunc(self):
        raise NotImplementedError


    def predict(self, premiseSent, hypothesisSent, predictFunc):
        """
        Output Labels for given premise/hypothesis sentences pair.
        :param premiseSent:
        :param hypothesisSent:
        :param predictFunc:
        :return: Label category from among "entailment", "contradiction", "neutral"
        """
        categories = ["entailment", "contradiction", "neutral"]
        labelIdx = predictFunc(premiseSent, hypothesisSent)
        labelCategories = []
        for idx in labelIdx:
            labelCategories.append(categories[idx])

        return labelCategories