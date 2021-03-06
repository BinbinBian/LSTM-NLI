import numpy as np
import sys
import time

# Otherwise PyCharm complains
sys.path.append("/Users/mihaileric/Documents/Research/keras")

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from model.embeddings import EmbeddingTable

# TODO: Implement softmax in keras as final output layer (activation?)
dataPath = "/Users/mihaileric/Documents/Research/LSTM-NLI/data/"
embeddingTable = EmbeddingTable(dataPath+"glove.6B.50d.txt.gz")


def buildAndEvaluateModel():
    start = time.time()

    testSent = "The cat is blue"
    idxMat = embeddingTable.convertSentToIdxMatrix(testSent)

    maxFeatures = 5 # Will iterate over data to compute vocabulary size
    inputLength = 7 # Will compute the max length of sentences
    Xtrain = np.random.choice(np.arange(0,maxFeatures), (5,5))
    Xtest = np.random.choice(np.arange(0,maxFeatures), (5,5))
    Ytrain = np.random.choice(np.arange(0,3), 5)
    Ytest = np.random.choice(np.arange(0,3), 5)
    model = Sequential()
    model.add(Embedding(maxFeatures, 256, input_length=inputLength))
    model.add(LSTM(output_dim=128, activation='sigmoid', inner_activation='hard_sigmoid')) # Set masking to 0?
    model.add(Dropout(0.5))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='rmsprop')
    print "Fitting model..."
    model.fit(Xtrain, Ytrain, batch_size=16, nb_epoch=10)
    print "Scoring model..."
    score = model.evaluate(Xtest, Ytest, batch_size=16)

    print "Done"
    print "Time elapsed: ", time.time() - start

if __name__ == "__main__":
    buildAndEvaluateModel()