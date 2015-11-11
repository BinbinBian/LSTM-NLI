""" A series of functionality tests for various components
of system.
"""

import numpy as np
import theano
import theano.tensor as T

from model.embeddings import EmbeddingTable
from model.layers import HiddenLayer

dataPath = "/Users/mihaileric/Documents/Research/LSTM-NLI/data/"


def testEmbeddings():
    table = EmbeddingTable(dataPath+"glove.6B.50d.txt.gz")
    print table.getEmbeddingFromWord("cat")
    print table.getEmbeddingFromWord("dog")
    print table.getEmbeddingFromWord("asssad")


def testSentToIdxMat():
    table = EmbeddingTable(dataPath+"glove.6B.50d.txt.gz")
    testSent1 = "The cat is blue"
    idxMat1 = table.convertSentToIdxMatrix(testSent1)
    print idxMat1

    testSent2 = "More dogs are happy"
    idxMat2 = table.convertSentToIdxMatrix(testSent2)
    print idxMat2


def testIdxListToEmbedList():
    table = EmbeddingTable(dataPath+"glove.6B.50d.txt.gz")
    idxList  = [[1], [4], [8]]
    print table.convertIdxMatrixToEmbeddingList(idxList)


def testHiddenLayerStep():
    hLayer = HiddenLayer(2,2, "blah")
    print hLayer.b_f.eval()
    print hLayer.W_f.eval() # Should give same values across runs

    input = T.alloc(np.array([[1,2],[3,4], [3,4]], dtype=np.float64), 3, 2)
    prevHidden = T.alloc(np.array([[4, 3],[0.2, -0.1], [0.2, -0.1]], dtype=np.float64), 3, 2)
    prevCell = T.alloc(np.array([[5, 1], [0.4, 2], [0.4, 2]], dtype=np.float64), 3, 2)
    nextHidden, nextCell = hLayer._step(input, prevHidden, prevCell)

    input = T.alloc(np.array([1,2], dtype=np.float64), 1, 2)
    prevHidden = T.alloc(np.array([4, 3], dtype=np.float64), 1, 2)
    prevCell = T.alloc(np.array([5, 0.4], dtype=np.float64), 1, 2)
    nextHidden, nextCell = hLayer._step(input, prevHidden, prevCell)

    input = T.alloc(np.array([[1,2],[3,4]], dtype=np.float64), 2, 2)
    prevHidden = T.alloc(np.array([[4, 3],[0.2, -0.1]], dtype=np.float64), 2, 2)
    prevCell = T.alloc(np.array([[5, 1], [0.4, 2]], dtype=np.float64), 2, 2)
    nextHidden, nextCell = hLayer._step(input, prevHidden, prevCell)

    print "-" * 100
    nextnextHidden, nextnextCell = hLayer._step(input, nextHidden, nextCell)
    print "-" * 100
    hLayer._step(input, nextnextHidden, nextnextCell)


def testHiddenLayerScan():
    hLayer = HiddenLayer(2, 2, "testHidden")
    inputMat = T.as_tensor_variable(np.random.randn(2,1,2)) #(numTimeSteps, numSamples, dimHidden)
    hLayer.forwardRun(inputMat, 2, 100)

if __name__ == "__main__":
  # testEmbeddings()
  # testHiddenLayer()
  #testSentToIdxMat()
  #testIdxListToEmbedList()
  #testHiddenLayerStep()
  testHiddenLayerScan()