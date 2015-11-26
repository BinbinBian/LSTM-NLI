""" Handles building, training, and testing the model.
"""
import argparse
import sys
import os

# May need to change this path
sys.path.append("/Users/mihaileric/Documents/Research/LSTM-NLI/")

from model.network import Network

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="argument parser for neural model")

    parser.add_argument("--batchSize", type=int,
                        help="batch size for training")
    parser.add_argument("--hiddenDim", type=int,
                        help="dimension of hidden layer")
    parser.add_argument("--numEpochs", type=int, default=4,
                        help="number of epochs to use for training")
    # TODO: Parse additional arguments as necessary
    args = parser.parse_args()

    network = Network() # Will pass arguments parsed here
    network.train(args.numEpochs, args.batchSize)