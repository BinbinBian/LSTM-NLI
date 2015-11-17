""" Defines the entire network
"""

import layers
import numpy as np
import theano
import theano.tensor as T

from model.embeddings import EmbeddingTable
from model.layers import HiddenLayer


class Network(object):
    """
    Represents entire network.
    """
    def __init__(self, numTimesteps=1, dimHidden=2, dataPath=None, optimizer="sgd"):
        """
        :param numTimesteps: Number of timesteps to unroll network for.
        :param dataPath: Path to file with precomputed word embeddings
        :param batchSize: Number of samples to use in each iteration of
                         training
        :param optimizer: Optimization algorithm to use for training. Will
                          eventually support adagrad, etc.
        """
        self.numTimesteps = numTimesteps
        self.embeddingTable = EmbeddingTable(dataPath)
        #self.dimInput = self.embeddingTable.dimEmbeddings
        self.dimInput = 2
        self.dimHidden = dimHidden
        self.inputMat = None # To store matrix of data input

    def buildModel(self):
        """
        Handles building of model, including initializing necessary parameters, defining
        loss functions, etc.
        """
        self.hiddenLayerPremise = HiddenLayer(self.dimHidden, self.dimInput, "premiseLayer")
        self.hiddenLayerHypothesis = HiddenLayer(self.dimHidden, self.dimInput, "hypothesisLayer")

        self.hiddenLayerHypothesis.updateParams(self.hiddenLayerPremise.params)


    def trainFunc(self):
        inputPremise = T.dtensor3("inputPremise")
        inputHypothesis = T.dtensor3("inputHypothesis")
        yTarget = T.dmatrix("yTarget")

        # self.hiddenLayerHypothesis.printParams()
        # del self.hiddenLayerHypothesis.params["weightsXo_hypothesisLayer"]
        # del self.hiddenLayerHypothesis.params["weightsCat_premiseLayer"]
        # del self.hiddenLayerHypothesis.params["biasO_hypothesisLayer"]

        # self.hiddenLayerHypothesis.printParams()

        self.hiddenLayerPremise.forwardRun(inputPremise, 1)
        premiseOutputHidden = self.hiddenLayerPremise.finalHiddenVal
        premiseOutputCandidate = self.hiddenLayerPremise.finalCandidateVal

        #print "premise Output Hidden: ", premiseOutputHidden

        self.hiddenLayerHypothesis.setInitialLayerParams(premiseOutputHidden, premiseOutputCandidate)
        self.hiddenLayerHypothesis.forwardRun(inputHypothesis, 1)
        hypothesisOutput = self.hiddenLayerHypothesis.finalHiddenVal
        cost = self.hiddenLayerHypothesis.costFunc(inputHypothesis, yTarget)
        grads = self.hiddenLayerHypothesis.computeGrads(inputHypothesis, yTarget, cost)
        trainF = theano.function([inputPremise, inputHypothesis, yTarget], grads)


        premiseSent = np.random.randn(1,1,2)
        hypothesisSent = np.random.randn(1,1,2)
        xNP = np.array([[[0.3, 0.04]]], dtype = np.float64)
        yTargetVal = np.array([[0., 1., 0.]], dtype=np.float64)

        print "Train out: ", trainF(premiseSent, hypothesisSent, yTargetVal)


    def train(self, numEpochs=5, batchSize=1):
        """
        Takes care of training model, including propagation of errors and updating of
        parameters.
        """
        #for epoch in xrange(numEpochs):

        premiseSent = T.as_tensor_variable(np.random.randn(1,1,2))
        hypothesisSent= T.as_tensor_variable(np.random.randn(1,1,2))






