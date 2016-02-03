import numpy as np
import os
# Hacky way to ensure that theano can find NVCC compiler
os.environ["PATH"] += ":/usr/local/cuda/bin"
import theano
import theano.tensor as T

from model.embeddings import EmbeddingTable
from model.network import Network
from util.trainingUtils import computeGrads, rmsprop
from util.utils import HeKaimingInitializer, GaussianDefaultInitializer, computeParamNorms


heka = HeKaimingInitializer()
normal = GaussianDefaultInitializer()

class SumEmbeddings(Network):
    """
    Simple baseline model for checking pipeline. Sum linear function of embeddings
    in premise/hypothesis respectively. Concatenate sentence representations, feed
    through tanh/relu layer and then apply softmax
    :param dimProject: dimension that word embeddings of sentence are projected to
    """

    def __init__(self, embedData, trainData, trainDataStats, valData, valDataStats,
                 testData, testDataStats, logPath, dimProject):
        super(SumEmbeddings, self).__init__(embedData, logPath, trainData, trainDataStats, valData,
                                      valDataStats, testData, testDataStats,
                                      numTimestepsPremise=None, numTimestepsHypothesis=None)
        self.configs = locals()
        self.params = {}
        self.dimProject = dimProject

        self.buildModel()


    def buildModel(self):
        # Note initialization scheme for weights; may want to change if not working
        self.W_proj = theano.shared(heka((self.dimEmbedding, self.dimProject)), name="weightsProject")
        self.b_proj = theano.shared(heka((1, self.dimProject)), name="biasProject")

        self.params["W_proj"] = self.W_proj
        self.params["b_proj"] = self.b_proj


    def trainFunc(self, inputPremise, inputHypothesis, yTarget, learnRate, gradMax,
                  L2regularization):
        premProject = T.dot(inputPremise, self.W_proj)
        hypoProject = T.dot(inputHypothesis, self.W_proj)

        sumPrem = premProject.sum(axis=1) + self.b_proj
        sumHypo = hypoProject.sum(axis=1) + self.b_proj # Should be dim (n, dimProject) where n is batch size

        concatVec = T.concatenate([sumPrem, sumHypo], axis=1)

        activeVec = T.tanh(concatVec)

        yPred = T.nnet.softmax(activeVec)
        entropy = T.nnet.categorical_crossentropy(yPred, yTarget).mean()
        cost = entropy + computeParamNorms([self.W_proj], L2regularization)

        costFunc = theano.function([inputPremise, inputHypothesis, yTarget], cost)

        grads, _ = computeGrads(inputPremise, inputHypothesis, yTarget,
                                cost, gradMax, self.params.values())

        fGradShared, fUpdate = rmsprop(grads, learnRate, inputPremise,
                                       inputHypothesis, yTarget, cost, self.params)

        return fGradShared, fUpdate, costFunc


    def train(self, numEpochs=1, batchSize=5, learnRateVal=0.1, gradMax=3.,
                L2regularization=0.0):
        pass


    def predictFunc(self):
        pass


    def predict(self):
        pass
