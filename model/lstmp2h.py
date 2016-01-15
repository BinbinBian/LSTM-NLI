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
from model.network import Network
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from util.afs_safe_logger import Logger
from util.stats import Stats
from util.utils import convertLabelsToMat, convertMatsToLabel, getMinibatchesIdx, \
                        convertDataToTrainingBatch

# Set random seed for deterministic runs
SEED = 100
np.random.seed(SEED)
currDir = os.path.dirname(os.path.dirname(__file__))


class LSTMP2H(Network):
    """
    Represents single layer premise LSTM to hypothesis LSTM network.
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
        super(LSTMP2H, self).__init__(logPath, trainData, trainDataStats, valData,
                                      valDataStats, testData, testDataStats)

        self.configs = locals()

        self.numTimestepsPremise = numTimestepsPremise
        self.numTimestepsHypothesis = numTimestepsHypothesis
        self.embeddingTable = EmbeddingTable(embedData)

        # Dimension of word embeddings at input
        self.dimEmbedding = self.embeddingTable.dimEmbeddings

        # Desired dimension of input to hidden layer
        self.dimInput = dimInput

        self.dimHidden = dimHidden

        # shared variable to keep track of whether to apply dropout in training/testing
        # 0. = testing; 1. = training
        self.dropoutMode = theano.shared(0.)

        # self.numericalParams = {} # Will store the numerical values of the
        #                 # theano variables that represent the params of the
        #                 # model; stored as dict of (name, value) pairs

        self.buildModel()


    # TODO: Check that this actually works-- I'm skeptical
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


    def computeAccuracy(self, dataPremiseMat, dataHypothesisMat, dataTarget,
                        predictFunc):
        """
        Computes the accuracy for the given network on a certain dataset.
        """
        numExamples = len(dataTarget)
        correctPredictions = 0.

        # Arbitrary batch size set
        minibatches = getMinibatchesIdx(len(dataTarget), 1)

        for _, minibatch in minibatches:
            batchPremiseTensor, batchHypothesisTensor, batchLabels = \
                    convertDataToTrainingBatch(dataPremiseMat, self.numTimestepsPremise, dataHypothesisMat,
                                               self.numTimestepsHypothesis, self.embeddingTable,
                                               dataTarget, minibatch)
            prediction = predictFunc(batchPremiseTensor, batchHypothesisTensor)
            batchGoldIdx = [ex.argmax(axis=0) for ex in batchLabels]

            correctPredictions += (np.array(prediction) ==
                                   np.array(batchGoldIdx)).sum()

        return correctPredictions/numExamples


    def buildModel(self):
        """
        Handles building of model, including initializing necessary parameters, etc.
        """
        self.hiddenLayerPremise = HiddenLayer(self.dimInput, self.dimHidden,
                                              self.dimEmbedding, "premiseLayer",
                                              self.dropoutMode)

        # Need to make sure not differentiating with respect to Wcat of premise
        # May want to find cleaner way to deal with this later
        del self.hiddenLayerPremise.params["weightsCat_premiseLayer"]
        del self.hiddenLayerPremise.params["biasCat_premiseLayer"]

        self.hiddenLayerHypothesis = HiddenLayer(self.dimInput, self.dimHidden,
                                        self.dimEmbedding, "hypothesisLayer",
                                        self.dropoutMode)

        # TODO: add above layers to self.layers
        self.layers.extend((self.hiddenLayerPremise, self.hiddenLayerHypothesis))


    def trainFunc(self, inputPremise, inputHypothesis, yTarget, learnRate, gradMax,
                  L2regularization, dropoutRate, sentenceAttention, wordwiseAttention,
                  batchSize, optimizer="rmsprop"):
        """
        Defines theano training function for layer, including forward runs and backpropagation.
        Takes as input the necessary symbolic variables.
        """
        if sentenceAttention:
            self.hiddenLayerHypothesis.initSentAttnParams()

        if wordwiseAttention:
            self.hiddenLayerHypothesis.initWordwiseAttnParams()

        self.hiddenLayerPremise.forwardRun(inputPremise, timeSteps=self.numTimestepsPremise) # Set numtimesteps here
        premiseOutputVal = self.hiddenLayerPremise.finalOutputVal
        premiseOutputCellState = self.hiddenLayerPremise.finalCellState

        self.hiddenLayerHypothesis.setInitialLayerParams(premiseOutputVal, premiseOutputCellState)
        cost, costFn = self.hiddenLayerHypothesis.costFunc(inputPremise,
                                    inputHypothesis, yTarget, "hypothesis",
                                    L2regularization, dropoutRate, self.hiddenLayerPremise.allOutputs, batchSize,
                                    sentenceAttention=sentenceAttention,
                                    wordwiseAttention=wordwiseAttention,
                                    numTimestepsHypothesis=self.numTimestepsHypothesis,
                                    numTimestepsPremise=self.numTimestepsPremise)

        gradsHypothesis, gradsHypothesisFn = self.hiddenLayerHypothesis.computeGrads(inputPremise,
                                                inputHypothesis, yTarget, cost, gradMax)

        gradsPremise, gradsPremiseFn = self.hiddenLayerPremise.computeGrads(inputPremise,
                                                inputHypothesis, yTarget, cost, gradMax)

        fGradSharedHypothesis, fUpdateHypothesis = self.hiddenLayerHypothesis.rmsprop(
            gradsHypothesis, learnRate, inputPremise, inputHypothesis, yTarget, cost)

        fGradSharedPremise, fUpdatePremise = self.hiddenLayerPremise.rmsprop(
            gradsPremise, learnRate, inputPremise, inputHypothesis, yTarget, cost)


        return (fGradSharedPremise, fGradSharedHypothesis, fUpdatePremise,
                fUpdateHypothesis, costFn, gradsHypothesisFn, gradsPremiseFn)


    def train(self, numEpochs=1, batchSize=5, learnRateVal=0.1, numExamplesToTrain=-1, gradMax=3.,
                L2regularization=0.0, dropoutRate=0.0, sentenceAttention=False,
                wordwiseAttention=False):
        """
        Takes care of training model, including propagation of errors and updating of
        parameters.
        """
        expName = "Epochs_{0}_LRate_{1}_L2Reg_{2}_dropout_{3}_sentAttn_{4}_" \
                       "wordAttn_{5}".format(str(numEpochs), str(learnRateVal),
                                             str(L2regularization), str(dropoutRate),
                                             str(sentenceAttention), str(wordwiseAttention))
        self.configs.update(locals())
        # trainPremiseIdxMat, trainHypothesisIdxMat = self.embeddingTable.convertDataToIdxMatrices(
        #                          self.trainData, self.trainDataStats)
        # trainGoldLabel = convertLabelsToMat(self.trainData)

        valPremiseIdxMat, valHypothesisIdxMat = self.embeddingTable.convertDataToIdxMatrices(
                                self.valData, self.valDataStats)
        valGoldLabel = convertLabelsToMat(self.valData)

        # If you want to train on less than full dataset
        if numExamplesToTrain > 0:
            valPremiseIdxMat = valPremiseIdxMat[:, range(numExamplesToTrain), :]
            valHypothesisIdxMat = valHypothesisIdxMat[:, range(numExamplesToTrain), :]
            valGoldLabel = valGoldLabel[range(numExamplesToTrain)]


        inputPremise = T.ftensor3(name="inputPremise")
        inputHypothesis = T.ftensor3(name="inputHypothesis")
        yTarget = T.fmatrix(name="yTarget")
        learnRate = T.scalar(name="learnRate", dtype='float32')


        fGradSharedHypothesis, fGradSharedPremise, fUpdatePremise, \
            fUpdateHypothesis, costFn, _, _ = self.trainFunc(inputPremise,
                                            inputHypothesis, yTarget, learnRate, gradMax,
                                            L2regularization, dropoutRate, sentenceAttention,
                                            wordwiseAttention, batchSize)

        totalExamples = 0
        stats = Stats(self.logger)

        # Training
        self.logger.Log("Model configs: {0}".format(self.configs))
        self.logger.Log("Starting training with {0} epochs, {1} batchSize,"
                " {2} learning rate, {3} L2regularization coefficient, and {4} dropout rate".format(
            numEpochs, batchSize, learnRateVal, L2regularization, dropoutRate))


        predictFunc = self.predictFunc(inputPremise, inputHypothesis, dropoutRate)

        for epoch in xrange(numEpochs):
            self.logger.Log("Epoch number: %d" %(epoch))

            if numExamplesToTrain > 0:
                # To see if can overfit on small dataset
                minibatches = getMinibatchesIdx(numExamplesToTrain, batchSize)
            else:
                minibatches = getMinibatchesIdx(len(valGoldLabel), batchSize)

            numExamples = 0
            for _, minibatch in minibatches:
                self.dropoutMode.set_value(1.)
                numExamples += len(minibatch)
                totalExamples += len(minibatch)

                self.logger.Log("Processed {0} examples in current epoch".
                                format(str(numExamples)))

                batchPremiseTensor, batchHypothesisTensor, batchLabels = \
                    convertDataToTrainingBatch(valPremiseIdxMat, self.numTimestepsPremise, valHypothesisIdxMat,
                                               self.numTimestepsHypothesis, self.embeddingTable,
                                               valGoldLabel, minibatch)

                gradHypothesisOut = fGradSharedHypothesis(batchPremiseTensor,
                                       batchHypothesisTensor, batchLabels)
                gradPremiseOut = fGradSharedPremise(batchPremiseTensor,
                                       batchHypothesisTensor, batchLabels)
                fUpdatePremise(learnRateVal)
                fUpdateHypothesis(learnRateVal)

                predictLabels = self.predict(batchPremiseTensor, batchHypothesisTensor, predictFunc)
                self.logger.Log("Labels in epoch {0}: {1}".format(epoch, str(predictLabels)))


                cost = costFn(batchPremiseTensor, batchHypothesisTensor, batchLabels)
                stats.recordCost(cost)

                # Periodically print val accuracy
                # Note: Big time sink happens here
                if totalExamples%(10) == 0:
                    self.dropoutMode.set_value(0.)
                    devAccuracy = self.computeAccuracy(valPremiseIdxMat,
                                                       valHypothesisIdxMat, valGoldLabel, predictFunc)
                    stats.recordAcc(totalExamples, devAccuracy, "dev")


        stats.recordFinalTrainingTime(totalExamples)

        # Save model to disk
        self.logger.Log("Saving model...")
        self.extractParams()
        configString = "batch={0},epoch={1},learnRate={2},dimHidden={3},dimInput={4}".format(str(batchSize),
                                            str(numEpochs), str(learnRateVal),
                                            str(self.dimHidden), str(self.dimInput))
        self.saveModel(currDir + "/savedmodels/basicLSTM_"+configString+".npz")
        self.logger.Log("Model saved!")

        # Set dropout to 0. again for testing
        self.dropoutMode.set_value(0.)

        #Train Accuracy
        # trainAccuracy = self.computeAccuracy(trainPremiseIdxMat,
        #                             trainHypothesisIdxMat, trainGoldLabel, predictFunc)
        # self.logger.Log("Final training accuracy: {0}".format(trainAccuracy))

        # Val Accuracy
        valAccuracy = self.computeAccuracy(valPremiseIdxMat,
                                    valHypothesisIdxMat, valGoldLabel, predictFunc)
        # TODO: change -1 for training acc to actual value when I enable train computation
        stats.recordFinalStats(totalExamples, -1, valAccuracy, expName)


    def predictFunc(self, symPremise, symHypothesis, dropoutRate):
        """
        Produces a theano prediction function for outputting the label of a given input.
        Takes as input a symbolic premise and a symbolic hypothesis.
        :return: Theano function for generating probability distribution over labels.
        """
        self.hiddenLayerPremise.forwardRun(symPremise, timeSteps=self.numTimestepsPremise)
        premiseOutputVal = self.hiddenLayerPremise.finalOutputVal
        premiseOutputCellState = self.hiddenLayerPremise.finalCellState

        # Run through hypothesis LSTM
        self.hiddenLayerHypothesis.setInitialLayerParams(premiseOutputVal, premiseOutputCellState)
        self.hiddenLayerHypothesis.forwardRun(symHypothesis, timeSteps=self.numTimestepsHypothesis)

        # Apply dropout here
        self.hiddenLayerHypothesis.finalOutputVal = self.hiddenLayerHypothesis.applyDropout(
                                self.hiddenLayerHypothesis.finalOutputVal, self.dropoutMode,
                                dropoutRate)
        catOutput = self.hiddenLayerHypothesis.projectToCategories()
        softMaxOut = T.nnet.softmax(catOutput)

        labelIdx = softMaxOut.argmax(axis=1)

        return theano.function([symPremise, symHypothesis], labelIdx, name="predictLabelsFunction")
