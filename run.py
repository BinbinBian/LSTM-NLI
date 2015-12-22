""" Handles building, training, and testing the model.
"""
import argparse
import os
import sys

# NOTE: May need to change this path
sys.path.append("/Users/mihaileric/Documents/Research/LSTM-NLI/")

from model.network import Network


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="argument parser for neural "
                                                 "model")
    parser.add_argument("--embedData", type=str,
                        help="path to precomputed word embeddings")
    parser.add_argument("--trainData", type=str,
                        help="path to train data")
    parser.add_argument("--trainDataStats", type=str,
                        help="path to stats about train data")
    parser.add_argument("--valData", type=str,
                        help="path to validation data")
    parser.add_argument("--valDataStats", type=str,
                        help="path to stats about validation data")
    parser.add_argument("--testData", type=str,
                        help="path to test data")
    parser.add_argument("--testDataStats", type=str,
                        help="path to stats about test data")
    parser.add_argument("--logPath", type=str,
                        help="path to file where model outputs will be logged")
    parser.add_argument("--batchSize", type=int,
                        help="batch size for training")
    parser.add_argument("--dimHidden", type=int, default=64,
                        help="dimension of hidden layer")
    parser.add_argument("--dimInput", type=int,
                        help="dimension of input to network")
    parser.add_argument("--numEpochs", type=int, default=4,
                        help="number of epochs to use for training")
    parser.add_argument("--learnRate", type=float,
                        help="learning rate used training")
    parser.add_argument("--unrollSteps", type=int,
                        help="number of steps to unroll LSTM layer")
    parser.add_argument("--numExamplesToTrain", type=int,
                        default=-1, help="number of examples to use for training"
                                         "if you don't want to use full data")
    parser.add_argument("--gradMax", type=float,
                        default=3., help="maximum gradient magnitude to use for "
                                         "gradient clipping")
    parser.add_argument("--regularization", type=float,
                        default=0., help="L2/L1 regularization coefficient")
    args = parser.parse_args()

    network = Network(args.embedData, args.trainData, args.trainDataStats,
                      args.valData, args.valDataStats, args.testData,
                      args.testDataStats, args.logPath, dimHidden=args.dimHidden,
                      dimInput=args.dimInput, numTimestepsPremise=args.unrollSteps,
                      numTimestepsHypothesis=args.unrollSteps)
    network.train(args.numEpochs, args.batchSize, args.learnRate, args.numExamplesToTrain,
                  args.gradMax)
