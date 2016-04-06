""" Handles building, training, and testing the model.
"""
import argparse
import os
import sys
# NOTE: May need to change this path
sys.path.append("/Users/mihaileric/Documents/Research/LSTM-NLI/")

from model import sum_embeddings
from util.utils import HeKaimingInitializer, GaussianDefaultInitializer


if __name__ == "__main__":
    heka = HeKaimingInitializer()

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
    parser.add_argument("--numEpochs", type=int, default=4,
                        help="number of epochs to use for training")
    parser.add_argument("--learnRate", type=float,
                        help="learning rate used training")
    parser.add_argument("--numExamplesToTrain", type=int,
                        default=-1, help="number of examples to use for training"
                                         "if you don't want to use full data")
    parser.add_argument("--gradMax", type=float,
                        default=3.0, help="maximum gradient magnitude to use for "
                                         "gradient clipping")
    parser.add_argument("--L2regularization", type=float,
                        default=0.0, help="L2/L1 regularization coefficient")
    parser.add_argument("--unrollSteps", type=int,
                        help="number of steps to unroll LSTM layer")
    parser.add_argument("--expName", type=str,
                        help="full path to experiment name")
    args = parser.parse_args()

    network = sum_embeddings.main(args.expName, args.embedData, args.trainData, args.trainDataStats,
                      args.valData, args.valDataStats, args.testData,
                      args.testDataStats, args.logPath, args.batchSize, args.numEpochs, args.unrollSteps, args.learnRate)
