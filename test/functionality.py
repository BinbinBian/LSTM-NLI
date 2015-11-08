""" A series of functionality tests for various components
of system.
"""

from model.embeddings import EmbeddingTable
from model.layers import HiddenLayer

dataPath = "/Users/mihaileric/Documents/Research/LSTM-NLI/data/"
table = EmbeddingTable(dataPath+"glove.6B.50d.txt.gz")

def testEmbeddings():
    table = EmbeddingTable(dataPath+"glove.6B.50d.txt.gz")
    print table.getEmbeddingFromWord("cat")
    print table.getEmbeddingFromWord("dog")
    print table.getEmbeddingFromWord("asssad")


def testSentToIdxMat():
    testSent1 = "The cat is blue"
    idxMat1 = table.convertSentToIdxMatrix(testSent1)
    print idxMat1

    testSent2 = "More dogs are happy"
    idxMat2 = table.convertSentToIdxMatrix(testSent2)
    print idxMat2


def testIdxListToEmbedList():
    idxList  = [[1], [4], [8]]
    print table.convertIdxMatrixToEmbeddingList(idxList)


def testHiddenLayer():
    pass


if __name__ == "__main__":
  # testEmbeddings()
  # testHiddenLayer()
  #testSentToIdxMat()
  testIdxListToEmbedList()