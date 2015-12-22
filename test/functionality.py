""" A series of functionality tests for various components
of system.
"""

import numpy as np
import os
# Hacky way to ensure that theano can find NVCC compiler
os.environ["PATH"] += ":/usr/local/cuda/bin"

import theano
import theano.tensor as T
import time

from model.embeddings import EmbeddingTable
from model.layers import HiddenLayer
from model.network import Network
from util.utils import convertLabelsToMat, computeParamNorms

dataPath = "/Users/mihaileric/Documents/Research/LSTM-NLI/data/"
# Set random seed for deterministic runs
SEED = 100
np.random.seed(SEED)


def testDeterministicParams():
    layer1 = HiddenLayer(2, 2, 2, "test")
    layer1.printLayerParams()

    layer2 = HiddenLayer(2, 2, 2, "test")
    layer2.printLayerParams()


def testLabelsMat():
    labelsMat = convertLabelsToMat("/Users/mihaileric/Documents/Research/"
                                   "LSTM-NLI/data/snli_1.0_dev.jsonl")
    print labelsMat


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
    hLayer = HiddenLayer(2, 2, 2, "blah")

    # input = T.alloc(np.array([[1,2],[3,4], [3,4]], dtype=np.float64), 3, 2)
    # prevHidden = T.alloc(np.array([[4, 3],[0.2, -0.1], [0.2, -0.1]], dtype=np.float64), 3, 2)
    # prevCell = T.alloc(np.array([[5, 1], [0.4, 2], [0.4, 2]], dtype=np.float64), 3, 2)
    # nextHidden, nextCell = hLayer._step(input, prevHidden, prevCell)

    input = T.alloc(np.array([1,2], dtype=np.float64), 1, 2)
    prevHidden = T.alloc(np.array([4, 3], dtype=np.float64), 1, 2)
    prevCell = T.alloc(np.array([5, 0.4], dtype=np.float64), 1, 2)
    nextHidden, nextCell = hLayer._step(input, prevHidden, prevCell)

    # input = T.alloc(np.array([[1,2],[3,4]], dtype=np.float64), 2, 2)
    # prevHidden = T.alloc(np.array([[4, 3],[0.2, -0.1]], dtype=np.float64), 2, 2)
    # prevCell = T.alloc(np.array([[5, 1], [0.4, 2]], dtype=np.float64), 2, 2)
    # nextHidden, nextCell = hLayer._step(input, prevHidden, prevCell)

    #print np.asarray(nextHidden.eval())
    # print "-" * 100
    # nextnextHidden, nextnextCell = hLayer._step(input, nextHidden, nextCell)
    # print "-" * 100
    # hLayer._step(input, nextnextHidden, nextnextCell)


def testHiddenLayerScan():
    hLayer = HiddenLayer(2, 2, 2, "testHidden")
    inputMat = T.as_tensor_variable(np.random.randn(3,1,2).astype(np.float32)) #(numTimeSteps, numSamples, dimHidden)
    modelOut, updates = hLayer.forwardRun(inputMat, 3)

    for idx, item in enumerate(modelOut):
        print "Item: {0}".format(idx)
        print np.asarray(item[0].eval())
        print np.asarray(item[1].eval())
        print np.asarray(item[-1].eval())
        print "----"


def testCatProjection():
    hLayer = HiddenLayer(2, 2, "testHidden")
    inputMat = T.as_tensor_variable(np.random.randn(2,1,2)) #(numTimeSteps, numSamples, dimHidden)
    hLayer.forwardRun(inputMat, 2, 100)
    catOutput = hLayer.projectToCategories()

    print "Final Hidden Val: ", hLayer.finalHiddenVal.eval()
    print "W cat: ", hLayer.W_cat.eval()
    print "b cat: ", hLayer.b_cat.eval()
    print "Cat Out: ", catOutput.eval()


def testGetPrediction():
    hLayer = HiddenLayer(2, 2, "testHidden")
    inputMat = T.as_tensor_variable(np.random.randn(1,1,2)) #(numTimeSteps, numSamples, dimHidden)
    hLayer.forwardRun(inputMat, 1, 100)
    catOutput = hLayer.projectToCategories()

    print "Cat Out: ", catOutput.eval()
    softmaxOut = hLayer.applySoftmax(catOutput)
    print "Softmax out: ", softmaxOut.eval()

    # cat2Output = T.alloc(np.array([[3,4,1],[2,3,4]], dtype=np.float64), 2, 3)
    # print "Cat 2 out: ", hLayer._getPredictions(cat2Output).eval()


def testCrossEntropyLoss():
    hLayer = HiddenLayer(2, 2, "testHidden")
    yPred = T.as_tensor_variable(np.array([0.3, 0.5, 0.2], dtype=np.float64))
    yTarget = T.as_tensor_variable(np.array([0., 1., 0.], dtype=np.float64))
    loss = hLayer.computeCrossEntropyCost(yPred, yTarget)

    print "Cross Entropy loss: ", loss.eval()

    yPred2 = T.as_tensor_variable(np.array([[0.3, 0.5, 0.2], [0.4, 0.3, 0.3]], dtype=np.float64))
    yTarget2 = T.as_tensor_variable(np.array([[0., 1., 0.], [1, 0, 0]], dtype=np.float64))
    loss2 = hLayer.computeCrossEntropyCost(yPred2, yTarget2)

    print "Second cross entropy loss: ", loss2.eval()


def testCostFuncPipeline():
    hLayer = HiddenLayer(dimInput=2, dimHiddenState=2, layerName="testHidden")

    x = T.dtensor3("testX")
    yTarget = T.dmatrix("testyTarget")

    xNP = np.array([[[0.5, 0.6]], [[0.3, 0.8]]], dtype = np.float32)
    yTargetNP = np.array([[0., 1., 0.]], dtype=np.float32)
    cost, costFunc = hLayer.costFunc(x, yTarget, numTimesteps=2)
    print "Cost: ", costFunc(xNP, yTargetNP)

    xNP = np.array([[[0.5, 0.6]]], dtype=np.float32)
    yTargetNP = np.array([[0., 1., 0.]], dtype=np.float32)
    costFunc = hLayer.costFunc(x, yTarget, numTimesteps=1)
    print "Cost: ", costFunc(xNP, yTargetNP)

    xNP = np.array([[[0.5, 0.6]], [[0.3, 0.8]]], dtype = np.float32)
    yTargetNP = np.array([[1., 1., 0.]], dtype=np.float32)
    cost, costFunc = hLayer.costFunc(x, yTarget, numTimesteps=2)
    print "Cost: ", costFunc(xNP, yTargetNP)


def testGradComputation():
    hLayer = HiddenLayer(dimInput=2, dimHiddenState=2, dimEmbedding=2,
                         layerName="testHidden")
    symPremise = T.ftensor3("testP")
    symHypothesis = T.ftensor3("testH")
    yTarget = T.fmatrix("testyTarget")

    inputPremise = np.array([[[0.5, 0.6]], [[0.3, 0.8]]], dtype=np.float32)
    inputHypothesis = np.array([[[0.2, 0.1]], [[0.2, 0.1]]], dtype=np.float32)
    yTargetNP = np.array([[0., 1., 0.]], dtype=np.float32)
    cost, costFn = hLayer.costFunc(symPremise, symHypothesis, yTarget, "hypothesis", numTimesteps=2)
    grads, gradsFn, gradsFnClipped = hLayer.computeGrads(symPremise, symHypothesis, yTarget, cost)
    gradOut = gradsFn(inputPremise, inputHypothesis, yTargetNP)
    gradOutClipped= gradsFnClipped(inputPremise, inputHypothesis, yTargetNP)
    for g in gradOut:
        print "Grad: ", np.asarray(g)

    for g in gradOutClipped:
        print "Clipped grad: ", np.asarray(g)


def testSGD():
    hLayer = HiddenLayer(dimInput=2, dimHiddenState=2, dimEmbedding=2,
                         layerName="testHidden")
    x = T.dtensor3("testX")
    yTarget = T.dmatrix("testyTarget")

    xNP = np.array([[[0.5, 0.6]], [[0.3, 0.8]]], dtype = np.float32)
    yTargetNP = np.array([[0., 1., 0.]], dtype=np.float32)
    cost, costFunc = hLayer.costFunc(x, yTarget, numTimesteps=2)
    grads, gradFunc = hLayer.computeGrads(x, yTarget, cost)
    gradsVals = gradFunc(xNP, yTargetNP)
    print "Grads: ", gradsVals

    print "Former param values: "
    for name, param in hLayer.params.iteritems():
        print name, ": ", param.eval()

    learnRate = T.scalar(name="learnRate")
    sgdFunc = hLayer.sgd(gradsVals, learnRate)
    sgdFunc(0.4)
    print "-"*100
    print "Updated param values: "
    for name, param in hLayer.params.iteritems():
        print name, ": ", param.eval()


def testNetworkSetup():
    network = Network()
    network.hiddenLayerHypothesis.params["biasO_premiseLayer"] += T.as_tensor_variable(np.array([1, 2]))
    network.hiddenLayerPremise.printParams()
    network.hiddenLayerHypothesis.printParams()


def testParamsBackPropUpdate():
    """
    Test to ensure that the parameters of premise are updated after backprop.
    """
    network = Network()
    network.train()


def testPredictFunc():
    """
    Test the network predict function
    """
    network = Network()

    symPremise = T.dtensor3("inputPremise")
    symHypothesis = T.dtensor3("inputHypothesis")
    premiseSent = np.random.randn(1,1,2)
    hypothesisSent = np.random.randn(1,1,2)

    predictFunc = network.predictFunc(symPremise, symHypothesis)
    labels = network.predict(premiseSent, hypothesisSent, predictFunc)

    for l in labels:
        print "Label: %s" %(l)


def testConvertToIdxMatrices():
    """
    Test conversion of data to embedding idx matrix.
    """
    table = EmbeddingTable(dataPath+"glove.6B.50d.txt.gz")
    dataStats= "/Users/mihaileric/Documents/Research/LSTM-NLI/data/dev_dataStats.json"
    dataJSONFile= "/Users/mihaileric/Documents/Research/LSTM-NLI/data/snli_1.0_dev.jsonl"
    premiseIdxMatrix, hypothesisIdxMatrix = table.convertDataToIdxMatrices(
                                                dataJSONFile, dataStats)


def testConvertIdxMatToIdxTensor():
    """
    Test conversion from idxMat to IdxTensor.
    """
    table = EmbeddingTable(dataPath+"glove.6B.50d.txt.gz")
    idxMat = np.array([[[3], [5], [-1]]])
    idxTensor = table.convertIdxMatToIdxTensor(idxMat)

    idxMat2 = np.zeros((2, 3, 4))
    idxMat2.fill(-1)
    idxTensor2 = table.convertIdxMatToIdxTensor(idxMat2)
    print 'hi'



def testSNLIExample():
    """
    Test an example actually taken from SNLI dataset on LSTM pipeline.
    """
    start = time.time()
    table = EmbeddingTable(dataPath+"glove.6B.50d.txt.gz")
    dataStats= "/Users/mihaileric/Documents/Research/LSTM-NLI/data/" \
               "test_dataStats.json"
    dataJSONFile= "/Users/mihaileric/Documents/Research/LSTM-NLI/data/" \
                  "snli_1.0_test.jsonl"
    premiseTensor, hypothesisTensor = table.convertDataToEmbeddingTensors(
                                                dataJSONFile, dataStats)

    symPremise = T.ftensor3("inputPremise")
    symHypothesis = T.ftensor3("inputHypothesis")

    premiseSent = premiseTensor[:, 0:3, :]
    hypothesisSent = hypothesisTensor[:, 0:3, :]

    network = Network(numTimestepsPremise=57, numTimestepsHypothesis=30,
                      dimInput=10, embedData="/Users/mihaileric/Documents/Research/"
                                             "LSTM-NLI/data/glove.6B.50d.txt.gz")
    network.printNetworkParams()

    predictFunc = network.predictFunc(symPremise, symHypothesis)
    labels = network.predict(premiseSent, hypothesisSent, predictFunc)

    for l in labels:
        print "Label: %s" %(l)

    print "Time for evaluation: %f" %(time.time() - start)


embedData = "/Users/mihaileric/Documents/Research/LSTM-NLI/data/glove.6B.50d.txt.gz"
trainData = "/Users/mihaileric/Documents/Research/LSTM-NLI/data/snli_1.0_train.jsonl"
trainDataStats = "/Users/mihaileric/Documents/Research/LSTM-NLI/data/train_dataStats.json"
trainLabels = "train_labels.json"
valData = "/Users/mihaileric/Documents/Research/LSTM-NLI/data/snli_1.0_dev.jsonl"
valDataStats = "/Users/mihaileric/Documents/Research/LSTM-NLI/data/dev_dataStats.json"
valLabels = "/Users/mihaileric/Documents/Research/LSTM-NLI/dev_labels.json"
testData = "/Users/mihaileric/Documents/Research/LSTM-NLI/data/snli_1.0_test.jsonl"
testDataStats = "/Users/mihaileric/Documents/Research/LSTM-NLI/data/test_dataStats.json"
logPath = "/Users/mihaileric/Documents/Research/LSTM-NLI/log/runOutput"

def testTrainFunctionality():
    start = time.time()
    network = Network(embedData, trainData, trainDataStats, valData, valDataStats, testData,
                testDataStats, logPath, dimInput=100, dimHidden=256,
                numTimestepsPremise=10, numTimestepsHypothesis=10)
    #network.printNetworkParams()
    network.train(numEpochs=10, batchSize=10, learnRateVal=0.0007, numExamplesToTrain=30,
                    gradMax=3., regularization=0.)
    print "Total time for training functionality test: {0}".format(time.time() - start)


def testExtractParamsAndSaveModel():
    network = Network(numTimestepsPremise=57, numTimestepsHypothesis=30,
                      dimInput=50, embedData=embedData, trainData=trainData,
                    trainLabels=trainLabels, trainDataStats=trainDataStats,
                    valData=valData, valDataStats=valDataStats, valLabels=valLabels)
    network.extractParams()
    network.saveModel("savedParamsFile.npz")


def testSaveLoadModel():
    network = Network(numTimestepsPremise=57, numTimestepsHypothesis=30,
                      dimInput=50, embedData=embedData, trainData=trainData,
                    trainDataStats=trainDataStats,
                    valData=valData, valDataStats=valDataStats)
    network.train()

    network2 = Network(numTimestepsPremise=57, numTimestepsHypothesis=30,
                      dimInput=50, embedData=embedData, trainData=trainData,
                    trainLabels=trainLabels, trainDataStats=trainDataStats,
                    valData=valData, valDataStats=valDataStats, valLabels=valLabels)
    network2.loadModel("basicLSTM_batch=5,epoch=1,learnR=0.1.npz")
    network2.printNetworkParams()


def testAccuracyComputation():
    network = Network(numTimestepsPremise=57, numTimestepsHypothesis=30,
                      dimInput=50, embedData=embedData, trainData=trainData,
                    trainLabels=trainLabels, trainDataStats=trainDataStats,
                    valData=valData, valDataStats=valDataStats, valLabels=valLabels)
    valPremiseIdxMat, valHypothesisIdxMat = network.embeddingTable.convertDataToIdxMatrices(
                                network.valData, network.valDataStats)
    valGoldLabel = convertLabelsToMat(network.valLabels)

    accuracy = network.computeAccuracy(valPremiseIdxMat,
                                       valHypothesisIdxMat, valGoldLabel)

    print "Number of examples: {0}".format(len(valGoldLabel))

    print "Accuracy computed: {0}".format(accuracy)


def testRegularization():
    layer = HiddenLayer(2, 2, 2, "test", numCategories=3)

    premise = T.ftensor3("testP")
    hypothesis = T.ftensor3("testH")
    yTarget = T.fmatrix("testyTarget")

    hyp = np.array([[[0.5, 0.6]], [[0.3, 0.8]]], dtype = np.float32)
    prem = np.array([[[0.5, 0.6]], [[0.3, 0.8]]], dtype = np.float32)
    yTargetNP = np.array([[0., 1., 0.]], dtype=np.float32)

    layer.printLayerParams()
    cost, fn = layer.costFunc(premise, hypothesis, yTarget, "hypothesis", 0.0, 1)
    costValue = fn(prem, hyp, yTargetNP)
    print "Cost: ", costValue

    LSTMparams = [layer.params[cParam] for cParam in layer.LSTMcellParams]
    print "L2 norm all params: ", computeParamNorms(layer.params.values(), 0.5).eval()
    print "L2 norm LSTM cell params: ", computeParamNorms(LSTMparams, 0.5).eval()


if __name__ == "__main__":
  #testLabelsMat()
  # testEmbeddings()
  # testHiddenLayer()
  #testSentToIdxMat()
  #testIdxListToEmbedList()
  #testHiddenLayerStep()
  #testHiddenLayerScan()
  #testCatProjection()
  #testGetPrediction()
  #testCrossEntropyLoss()
  #testCostFuncPipeline()
  #testGradComputation()
   # testSGD()
   #testNetworkSetup()
   #testParamsBackPropUpdate()
   #testPredictFunc()
   #testSNLIExample()
   #testConvertToIdxMatrices()
   #testConvertIdxMatToIdxTensor()
   testTrainFunctionality()
   #testExtractParamsAndSaveModel()
   #testSaveLoadModel()
   #testAccuracyComputation()
   #testDeterministicParams()
   #testRegularization()
