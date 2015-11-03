import numpy as np

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM



def buildAndEvaluateModel():
    maxFeatures = 100 # Will iterate over data to compute vocabulary size
    inputLength = 10 # Will compute the max length of sentences
    Xtrain, Ytrain, Xtest, Ytest = np.random.randn(10,10)
    model = Sequential()
    model.add(Embedding(maxFeatures, 256, input_length=inputLength))
    model.add(LSTM(output_dim=128, activation='sigmoid', inner_activation='hard_sigmoid'))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='rmsprop')

    model.fit(Xtrain, Ytrain, batch_size=16, nb_epoch=10)
    score = model.evaluate(Xtest, Ytest, batch_size=16)