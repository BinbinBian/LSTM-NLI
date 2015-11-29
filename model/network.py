""" Defines the entire network
"""

import layers
import numpy as np
import os
# Hacky way to ensure that theano can find NVCC compiler
os.environ["PATH"] += ":/usr/local/cuda/bin"
import theano
import theano.tensor as T
import time

from model.embeddings import EmbeddingTable
from model.layers import HiddenLayer
from util.afs_safe_logger import Logger
from util.utils import convertLabelsToMat


# Set random seed for deterministic runs
SEED = 100
np.random.seed(SEED)

class Network(object):
    """
    Represents entire network.
    """
    def __init__(self, embedData, trainData, trainDataStats, valData, valDataStats,
                 testData, testDataStats, logPath, dimHidden=2,
                 dimInput=2, numTimestepsPremise=1, numTimestepsHypothesis=1):
        """
        :param numTimesteps: Number of timesteps to unroll network for.
        :param dataPath: Path to file with precomputed word embeddings
        :param batchSize: Number of samples to use in each iteration of
                         training
        :param optimizer: Optimization algorithm to use for training. Will
                          eventually support adagrad, etc.
        """
        self.numTimestepsPremise = numTimestepsPremise
        self.numTimestepsHypothesis = numTimestepsHypothesis
        self.embeddingTable = EmbeddingTable(embedData)

        # Paths to all data files
        self.trainData = trainData
        self.trainDataStats = trainDataStats
        self.valData = valData
        self.valDataStats = valDataStats
        self.testData = testData
        self.testDataStats = testDataStats
        self.logger = Logger(log_path=logPath)

        # Dimension of word embeddings at input
        self.dimEmbedding = self.embeddingTable.dimEmbeddings

        # Desired dimension of input to hidden layer
        self.dimInput = dimInput

        self.dimHidden = dimHidden
        # To store matrix of data input
        self.inputMat = None

        self.premiseLSTMName = "premiseLayer"
        self.hypothesisLSTMName = "hypothesisLayer"

        self.numericalParams = {} # Will store the numerical values of the
                        # theano variables that represent the params of the
                        # model; stored as dict of (name, value) pairs

        self.buildModel()


    def printNetworkParams(self):
        """
        Print all params of network.
        """
        for layer in [self.hiddenLayerPremise, self.hiddenLayerHypothesis]:
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
        for paramName, paramVar in self.hiddenLayerHypothesis.params.iteritems():
            self.numericalParams[paramName] = paramVar.get_value()


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
        with open(modelFileName, 'r') as f:
            params = np.load(f)
            premiseParams = {}
            for paramName, paramVal in params.iteritems():
                paramPrefix, layerName = paramName.split("_")

                # Set hypothesis params
                try:
                    self.hiddenLayerHypothesis.params[paramName].set_value(paramVal)
                except:
                    if paramPrefix[0:4] == "bias": # Hacky
                        self.hiddenLayerHypothesis.params[paramName] = \
                            theano.shared(paramVal, broadcastable=(True, False))
                    else:
                        self.hiddenLayerHypothesis.params[paramName] = \
                            theano.shared(paramVal)

                # Find params of premise layer
                if layerName == "premiseLayer":
                    premiseParams[paramName] = paramVal

            # Set premise params
            for paramName, paramVal in premiseParams.iteritems():
                self.hiddenLayerPremise.params[paramName].set_value(paramVal)


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


    @staticmethod
    def getMinibatchesIdx(numDataPoints, minibatchSize, shuffle=False):
        """
        Used to shuffle the dataset at each iteration. Return list of
        (batch #, batch) pairs.
        """

        idxList = np.arange(numDataPoints, dtype="int32")

        if shuffle:
            np.random.shuffle(idxList)

        minibatches = []
        minibatchStart = 0
        for i in xrange(numDataPoints // minibatchSize):
            minibatches.append(idxList[minibatchStart:
                                        minibatchStart + minibatchSize])
            minibatchStart += minibatchSize

        if (minibatchStart != numDataPoints):
            # Make a minibatch out of what is left
            minibatches.append(idxList[minibatchStart:])

        return zip(range(len(minibatches)), minibatches)


    def computeAccuracy(self, dataPremiseMat, dataHypothesisMat, dataTarget):
        """
        Computes the accuracy for the given network on a certain dataset.
        """
        numExamples = len(dataTarget)
        correctPredictions = 0.

        inputPremise = T.ftensor3("inputPremise")
        inputHypothesis = T.ftensor3("inputHypothesis")
        predictFn = self.predictFunc(inputPremise, inputHypothesis)

        # Arbitary batch size set
        minibatches = Network.getMinibatchesIdx(len(dataTarget), 10)

        for _, minibatch in minibatches:
            batchPremise = dataPremiseMat[0:self.numTimestepsPremise, minibatch, :]
            batchPremiseTensor = self.embeddingTable.convertIdxMatToIdxTensor(batchPremise)
            batchHypothesis = dataHypothesisMat[0:self.numTimestepsHypothesis, minibatch, :]
            batchHypothesisTensor = self.embeddingTable.convertIdxMatToIdxTensor(batchHypothesis)

            prediction = predictFn(batchPremiseTensor, batchHypothesisTensor)
            batchLabels = dataTarget[minibatch]
            batchGoldIdx = [ex.argmax(axis=0) for ex in batchLabels]

            correctPredictions += (np.array(prediction) ==
                                   np.array(batchGoldIdx)).sum()

        return correctPredictions/numExamples


    def buildModel(self):
        """
        Handles building of model, including initializing necessary parameters, defining
        loss functions, etc.
        """
        self.hiddenLayerPremise = HiddenLayer(self.dimInput, self.dimHidden,
                                              self.dimEmbedding, "premiseLayer")

        # Need to make sure not differentiating with respect to Wcat of premise
        # May want to find cleaner way to deal with this later
        del self.hiddenLayerPremise.params["weightsCat_premiseLayer"]
        del self.hiddenLayerPremise.params["biasCat_premiseLayer"]

        self.hiddenLayerHypothesis = HiddenLayer(self.dimInput, self.dimHidden,
                                        self.dimEmbedding, "hypothesisLayer")


    def trainFunc(self, inputPremise, inputHypothesis, yTarget, learnRate):
        """
        Defines theano training function for layer, including forward runs and backpropagation.
        Takes as input the necessary symbolic variables.
        """
        self.hiddenLayerPremise.forwardRun(inputPremise, 1)
        premiseOutputHidden = self.hiddenLayerPremise.finalHiddenVal
        premiseOutputCandidate = self.hiddenLayerPremise.finalCandidateVal

        self.hiddenLayerHypothesis.setInitialLayerParams(premiseOutputHidden, premiseOutputCandidate)
        cost, costFn = self.hiddenLayerHypothesis.costFunc(inputPremise,
                                    inputHypothesis, yTarget, "hypothesis", 1)

        grads, gradsFn = self.hiddenLayerHypothesis.computeGrads(inputPremise,
                                                inputHypothesis, yTarget, cost)
        paramUpdates = self.hiddenLayerHypothesis.sgd(grads, learnRate)
        trainFn = theano.function([inputPremise, inputHypothesis, yTarget, learnRate], updates=paramUpdates, name='trainNet')

        return trainFn, costFn, gradsFn


    def train(self, numEpochs=1, batchSize=5, learnRateVal=0.1):
        """
        Takes care of training model, including propagation of errors and updating of
        parameters.
        """
        # trainPremiseIdxMat, trainHypothesisIdxMat = self.embeddingTable.convertDataToIdxMatrices(
        #                         self.trainData, self.trainDataStats)
        # trainGoldLabel = convertLabelsToMat(self.trainLabels)

        valPremiseIdxMat, valHypothesisIdxMat = self.embeddingTable.convertDataToIdxMatrices(
                                self.valData, self.valDataStats)
        valGoldLabel = convertLabelsToMat(self.valData)


        inputPremise = T.ftensor3(name="inputPremise")
        inputHypothesis = T.ftensor3(name="inputHypothesis")
        yTarget = T.fmatrix(name="yTarget")
        learnRate = T.scalar(name="learnRate", dtype='float32')

        self.hiddenLayerHypothesis.appendParams(self.hiddenLayerPremise.params)
        trainNetFunc, costFn, gradsFn = self.trainFunc(inputPremise,
                                        inputHypothesis, yTarget, learnRate)

        totalExamples = 0

        startTime = time.time()

        # Training
        self.logger.Log("Starting training with {0} epochs, {1} batchSize, and"
                " {2} sgd learning rate".format(numEpochs, batchSize, learnRateVal))
        for epoch in xrange(numEpochs):
            self.logger.Log("Epoch number: %d" %(epoch))

            minibatches = Network.getMinibatchesIdx(len(valGoldLabel), batchSize)
            numExamples = 0
            for _, minibatch in minibatches:
                numExamples += len(minibatch)
                totalExamples += len(minibatch)

                self.logger.Log("Processed {0} examples".format(str(numExamples)))

                batchPremise = valPremiseIdxMat[0:self.numTimestepsPremise, minibatch, :]
                batchPremiseTensor = self.embeddingTable.convertIdxMatToIdxTensor(batchPremise)
                batchHypothesis = valHypothesisIdxMat[0:self.numTimestepsHypothesis, minibatch, :]
                batchHypothesisTensor = self.embeddingTable.convertIdxMatToIdxTensor(batchHypothesis)

                batchLabels = valGoldLabel[minibatch]


                #self.hiddenLayerHypothesis.printParams()
                gradOut = trainNetFunc(batchPremiseTensor,
                                       batchHypothesisTensor, batchLabels, learnRateVal)
                newPremiseGrads = self.hiddenLayerHypothesis.getPremiseGrads()
                self.hiddenLayerPremise.updateParams(newPremiseGrads)
                #self.hiddenLayerHypothesis.printParams()

                # Note '15' below is completely arbitrary
                if numExamples%(batchSize * 15) == 0:
                    valAccuracy = self.computeAccuracy(valPremiseIdxMat,
                                    valHypothesisIdxMat, valGoldLabel)
                    self.logger.Log("Current validation accuracy after {0} examples: {1}".\
                                            format(totalExamples, valAccuracy))


                self.hiddenLayerHypothesis.appendParams(self.hiddenLayerPremise.params)

        self.logger.Log("Training complete! Total training time: {0}".format
                    (time.time() - startTime))

        # Save model to disk
        self.logger.Log("Saving model...")
        self.extractParams()
        configString = "batch={0},epoch={1},learnR={2}".format(str(batchSize),
                                            str(numEpochs), str(learnRateVal))
        self.saveModel("basicLSTM_"+configString+".npz")
        self.logger.Log("Model saved!")

        # Train Accuracy
        # trainAccuracy = self.computeAccuracy(trainPremiseIdxMat,
        #                             trainHypothesisIdxMat, trainGoldLabel)
        # print "Training accuracy: {0}".format(trainAccuracy)

        # Val Accuracy
        valAccuracy = self.computeAccuracy(valPremiseIdxMat,
                                    valHypothesisIdxMat, valGoldLabel)
        self.logger.Log("Final validation accuracy: {0}".format(valAccuracy))


    def predictFunc(self, symPremise, symHypothesis):
        """
        Produces a theano prediction function for outputting the label of a given input.
        Takes as input a symbolic premise and a symbolic hypothesis.
        :return: Theano function for generating probability distribution over labels.
        """
        self.hiddenLayerPremise.forwardRun(symPremise, 1)
        premiseOutputHidden = self.hiddenLayerPremise.finalHiddenVal
        premiseOutputCandidate = self.hiddenLayerPremise.finalCandidateVal

        # Run through hypothesis LSTM
        self.hiddenLayerHypothesis.setInitialLayerParams(premiseOutputHidden, premiseOutputCandidate)
        self.hiddenLayerHypothesis.forwardRun(symHypothesis, 1)
        catOutput = self.hiddenLayerHypothesis.projectToCategories()
        softMaxOut = self.hiddenLayerHypothesis.applySoftmax(catOutput)
        labelIdx = softMaxOut.argmax(axis=1)

        return theano.function([symPremise, symHypothesis], labelIdx, name="predictLabelsFunction")


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
