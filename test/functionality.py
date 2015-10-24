""" A series of functionality tests for various components
of system.
"""

from model.embeddings import EmbeddingTable
from model.layers import HiddenLayer

dataPath = "/Users/mihaileric/Documents/Research/LSTM-NLI/data/"


def testEmbeddings():
    table = EmbeddingTable(dataPath+"glove.6B.50d.txt.gz")
    print table.getWordEmbedding("cat")
    print table.getWordEmbedding("dog")
    print table.getWordEmbedding("asssad")

def testHiddenLayer():
    pass

if __name__ == "__main__":
  # testEmbeddings()
    testHiddenLayer()