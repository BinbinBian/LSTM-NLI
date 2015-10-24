""" Defines the entire network
"""

import layers
import numpy as np
import theano
import theano.tensor as T

from model.embeddings import EmbeddingTable



class Network(object):
    """
    Represents entire network object.
    """
    def __init__(self, numTimesteps=1, dataPath=None, batchSize=5):
        """
        :param numTimesteps: Number of timesteps to unroll network for.
        :param dataPath: Path to file with precomputed word embeddings
        """
        self.numTimesteps = numTimesteps
        self.batchSize = batchSize
        self.embeddingTable = EmbeddingTable(dataPath)
