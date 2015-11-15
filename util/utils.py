"""Defines a series of useful utility functions for various modules."""

import csv
import json
import numpy as np
import os
import re
import sys

"""Add root directory path"""
root_dir = os.path.dirname(os.path.dirname(__file__))
sys.path.append(root_dir)


WORD_RE = re.compile(r"([^ \(\)]+)", re.UNICODE)

def str2tree(s):
    """Turns labeled bracketing s into a tree structure (tuple of tuples)"""
    s = WORD_RE.sub(r'"\1",', s)
    s = s.replace(")", "),").strip(",")
    s = s.strip(",")
    return eval(s)

def leaves(t):
    """Returns all of the words (terminal nodes) in tree t"""
    words = []
    for x in t:
        if isinstance(x, str):
            words.append(x)
        else:
            words += leaves(x)
    return words

os.chdir("../")
data_dir = os.getcwd() + "/data/"

def sick_reader(src_filename):
    count = 1
    for example in csv.reader(file(src_filename), delimiter="\t"):
        print "Sentence count: ", count
        label, t1, t2 = example[:3]
        if not label.startswith('%') and not label=="gold_label": # Some files use leading % for comments.
            yield (label, str2tree(t1), str2tree(t2))
        count += 1


#Readers for processing SICK datasets
def sick_train_reader():
    return sick_reader(src_filename=data_dir+"SICK_train_parsed.txt", semafor_filename=data_dir+"semafor_train.xml")


def sick_dev_reader():
    return sick_reader(src_filename=data_dir+"SICK_dev_parsed.txt", semafor_filename=data_dir+"semafor_dev.xml")


def sick_test_reader():
    return sick_reader(src_filename=data_dir+"SICK_test_parsed.txt", semafor_filename=data_dir+"semafor_test.xml")


def sick_train_dev_reader():
    return sick_reader(src_filename=data_dir+"SICK_train+dev_parsed.txt", semafor_filename=data_dir+"semafor_traindev.xml")


def snli_reader(src_filename):
    for example in csv.reader(file(src_filename), delimiter="\t"):
        label, t1, t2 = example[:3]
        if not label.startswith('%'): # Some files use leading % for comments.
            yield (label, str2tree(t1), str2tree(t2))


#Readers for processing SICK datasets
def snli_train_reader():
    return sick_reader(src_filename=data_dir+"snli_1.0rc3_train.txt")


def snli_dev_reader():
    return sick_reader(src_filename=data_dir+"snli_1.0rc3_dev.txt")


def snli_test_reader():
    return sick_reader(src_filename=data_dir+"snli_1.0rc3_test.txt")


def computeDataStatistics(dataSet="dev"):
    """
    Iterate over data using provided reader and compute statistics. Output values to JSON files.
    :param dataSet: Name of dataset to compute statistics from among 'train', 'dev', or 'test'
    """
    reader = None
    if dataSet == "train":
        reader = snli_train_reader
    elif dataSet == "dev":
        reader = snli_dev_reader
    else:
        reader = snli_test_reader

    # TODO: Fix error in "-" appearing as a label in dataset

    sentences = list()
    labels = list()
    vocab = set()
    minSenLength = float("inf")
    maxSenLength = float("-inf")
    allLabels = ['entailment', 'contradiction', 'neutral']

    for label, t1Tree, t2Tree in reader():
        if label not in allLabels:
            continue
        labels.append(label)

        t1Tokens = leaves(t1Tree)
        t2Tokens = leaves(t2Tree)

        if len(t1Tokens) > maxSenLength:
            maxSenLength = len(t1Tokens)
        if len(t1Tokens) < minSenLength:
            minSenLength = len(t1Tokens)

        if len(t2Tokens) > maxSenLength:
            maxSenLength = len(t2Tokens)
        if len(t2Tokens) < minSenLength:
            minSenLength = len(t2Tokens)

        vocab.update(set(t1Tokens))
        vocab.update(set(t2Tokens))

        # Append both premise and hypothesis as single list
        # TODO: ensure that sentences cleaned, lower-cased appropriately
        sentences.append([t1Tokens, t2Tokens])

    # Output results to JSON file
    with open(dataSet+"_labels.json", "w") as labelsFile:
        json.dump({'labels': labels}, labelsFile)

    with open(dataSet+"_dataStats.json", "w") as dataStatsFile:
        json.dump({"vocabSize": len(vocab), "minSentLen": minSenLength,
                   "maxSentLen": maxSenLength}, dataStatsFile)

    with open(dataSet+"_sentences.json", "w") as sentenceFile:
        json.dump({"sentences": sentences}, sentenceFile)


    return vocab, sentences, labels, minSenLength, maxSenLength

def convertLabelsToMat(labelFile):
    """
    Converts json file of labels to a (numSamples, 3) matrix with a 1 in the column
    corresponding to the label.
    :param labelFile:
    :return: numpy matrix corresponding to the labels
    """
    labelsList = ["entailment", "contradiction", "neutral"]
    with open(labelFile, "r") as f:
        labelsObj = json.load(f)
        labels = labelsObj["labels"]

        labelsMat = np.zeros((len(labels), 3), dtype=np.float64)
        for idx, label in enumerate(labels):
            labelIdx = labelsList.index(label)
            labelsMat[idx][labelIdx] = 1.

    return labelsMat