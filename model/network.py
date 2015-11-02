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
    def __init__(self, numTimesteps=1, dimHidden=5, dataPath=None, optimizer="sgd"):
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
        self.dimInput = self.embeddingTable.dimEmbeddings
        self.dimHidden = dimHidden
        self.optimizer = optimizer

    def _buildModel(self):
        """
        Handles building of model, including initializing necessary parameters, defining
        loss functions, etc.
        """
        self.hiddenLayer = HiddenLayer(self.dimHidden, self.dimInput)
        #self.hiddenLayer.forwardRun()

        prediction = T.nnet.softmax(T.dot(self.hiddenLayer.outputs,
                                        self.hiddenLayer.weightsOutputTransform)
                                        + self.hiddenLayer.biasOutputTransform)
        predictionProbs = theano.function([input], prediction, name='predictionProbs')

        loss = -1 * T.log(prediction[T.arange(10), np.arange(10)]).mean() # TODO: need to fix this

         
        raise NotImplementedError

    def train(self, numEpochs=5, batchSize=10):
        """
        Takes care of training model, including propagation of errors and updating of
        parameters.
        """
        raise NotImplementedError



