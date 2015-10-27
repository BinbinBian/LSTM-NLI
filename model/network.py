""" Defines the entire network
"""

import layers
import numpy as np
import theano
import theano.tensor as T

from model.embeddings import EmbeddingTable



class Network(object):
    """
    Represents entire network.
    """
    def __init__(self, numTimesteps=1, dataPath=None, batchSize=5):
        """
        :param numTimesteps: Number of timesteps to unroll network for.
        :param dataPath: Path to file with precomputed word embeddings
        :param batchSize: Number of samples to use in each iteration of
                         training
        """
        self.numTimesteps = numTimesteps
        self.batchSize = batchSize
        self.embeddingTable = EmbeddingTable(dataPath)

    def initializeInputMat(self):
        pass

    def train(self):
        """
        Takes care of training model, including propagation of errors and updating of
        parameters.
        """
        pass



