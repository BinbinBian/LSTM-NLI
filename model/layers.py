""" Defines the hidden layers used by the model.
"""
import numpy as np
import theano
import theano.tensor as T

from util.afs_safe_logger import Logger


# Set random seed for deterministic runs
SEED = 100
np.random.seed(SEED)

logger = Logger(log_path="/Users/mihaileric/Documents/Research/LSTM-NLI/log/"
                         "experimentLog.txt")

class HiddenLayer(object):
    def __init__(self, dimInput, dimHiddenState, dimEmbedding, layerName, numCategories=3):
        """
        :param inputMat: Matrix of input vectors to use for unraveling
                         hidden layer.
        :param dimInput: Dimension of vector of input to hidden cell.
        :param dimHiddenState: Dimension of hidden state.
        :param layerName: Name of current LSTM layer ('premise', 'hypothesis')
        """
        # Dictionary of model parameters.
        self.params = {}

        self.layerName = layerName
        self.inputDim = dimInput
        self.dimHidden = dimHiddenState
        self.dimEmbedding = dimEmbedding

        # Represents number of categories used for classification
        self.numLabels = numCategories

        # Initialization values for hidden layer compute unit parameters
        self.hiddenInit = None
        self.candidateValInit = None

        # TODO: what to use for initializing parameters (random?) , Addendum: Kaiming He initialization!!!

        # Parameters for projecting from embedding size to input dimension
        self.b_toInput = theano.shared(np.random.randn(1, dimInput).astype(
                    np.float32), name="biasToInput_"+layerName,
                    broadcastable=(True, False))
        self.W_toInput = theano.shared(np.random.randn(dimInput,
                                            dimEmbedding).astype(np.float32),
                                            name="weightsXtoInput_"+layerName)

        # Parameters for forget gate
        self.b_f = theano.shared(np.random.randn(1, dimHiddenState).astype(
            np.float32), name="biasF_"+layerName, broadcastable=(True, False))
        self.W_f = theano.shared(np.random.randn(dimHiddenState,
                                                 dimInput).astype(np.float32),
                                                name="weightsXf_"+layerName)
        self.U_f = theano.shared(np.random.randn(dimHiddenState,
                                        dimHiddenState).astype(np.float32),
                                        name="weightsHf_"+layerName)

        # Parameters for input gate
        self.b_i = theano.shared(np.random.randn(1, dimHiddenState).astype(
            np.float32), name="biasI_"+layerName, broadcastable=(True, False))
        self.W_i = theano.shared(np.random.randn(dimHiddenState,
                                                dimInput).astype(np.float32),
                                                name="weightsXi_"+layerName)
        self.U_i = theano.shared(np.random.randn(dimHiddenState,
                                        dimHiddenState).astype(np.float32),
                                        name="weightsHi_"+layerName)


        # Parameters for candidate values
        self.b_c = theano.shared(np.random.randn(1, dimHiddenState).astype(
            np.float32), name="biasC_"+layerName, broadcastable=(True, False))
        self.W_c = theano.shared(np.random.randn(dimHiddenState,
                                               dimInput).astype(np.float32),
                                               name="weightsXc_"+layerName)
        self.U_c = theano.shared(np.random.randn(dimHiddenState,
                                         dimHiddenState).astype(np.float32),
                                         name="weightsHc_"+layerName)

        # Parameters for final output vector transform (for final
        # classification)
        self.b_o = theano.shared(np.random.randn(1, dimHiddenState).astype(
            np.float32), name="biasO_"+layerName, broadcastable=(True, False))
        self.W_o = theano.shared(np.random.randn(dimHiddenState,
                                               dimInput).astype(np.float32),
                                               name="weightsXo_"+layerName)
        self.U_o = theano.shared(np.random.randn(dimHiddenState,
                                         dimHiddenState).astype(np.float32),
                                         name="weightsHo_"+layerName)


        # Parameters for linear projection from output of forward pass to a
        # vector with dimension equal to number of categories being classified
        # via one more softmax
        self.b_cat = theano.shared(np.random.randn(1, self.numLabels).astype(
            np.float32), name="biasCat_"+layerName, broadcastable=(True, False))
        self.W_cat = theano.shared(np.random.randn(dimHiddenState,
                                            self.numLabels).astype(np.float32),
                                            name="weightsCat_"+layerName)


        self.finalCandidateVal = None # Stores final cell state from scan in forwardPass
        self.finalHiddenVal = None  # Stores final hidden state from scan in forwardPass

        # Add shared vars to params dict
        self.params["biasToInput_"+layerName] = self.b_toInput
        self.params["weightsToInput_"+layerName] = self.W_toInput

        self.params["biasI_"+layerName] = self.b_i
        self.params["weightsXi_"+layerName] = self.W_i
        self.params["weightsHi_"+layerName] = self.U_i

        self.params["biasF_"+layerName] = self.b_f
        self.params["weightsXf_"+layerName] = self.W_f
        self.params["weightsHf_"+layerName] = self.U_f

        self.params["biasC_"+layerName] = self.b_c
        self.params["weightsXc_"+layerName] = self.W_c
        self.params["weightsHc_"+layerName] = self.U_c

        self.params["biasO_"+layerName] = self.b_o
        self.params["weightsXo_"+layerName] = self.W_o
        self.params["weightsHo_"+layerName] = self.U_o

        self.params["biasCat_"+layerName] = self.b_cat
        self.params["weightsCat_"+layerName] = self.W_cat


    def appendParams(self, newParams):
        """
        Append to the params dict for current layer with new set of params from previous layer
        :param newParams:
        :return:
        """
        self.params.update(newParams)


    def printLayerParams(self):
        """
        Print params in current layer
        :return:
        """
        print "Current parameter values for %s" %self.layerName
        print "-" * 50
        for pName, pValue in self.params.iteritems():
            print pName + " : " + str(np.asarray(pValue.eval()))

        print "-" * 50


    def getPremiseGrads(self):
        """
        Return a list of pairs of form (paramName, gradValue) to update premise layer
        :return:
        """
        premiseGrads = []
        for paramName, value in self.params.iteritems():
            descrip, suffix = paramName.split("_")
            if suffix == "premiseLayer":
                premiseGrads.append((paramName, value))

        return premiseGrads


    def updateParams(self, paramUpdates):
        """
        Update layer params with new values of params
        :param paramUpdates: New values of params
        """
        for paramName, newValue in paramUpdates:
            self.params[paramName] = newValue


    def setInitialLayerParams(self, hiddenInit, candidateValInit):
        """
        Sets the initialization hidden value and candidate value parameters
        :param hiddenValInit:
        :param candidateValInit:
        """
        self.hiddenInit = hiddenInit
        self.candidateValInit = candidateValInit


    def _step(self, input, prevHiddenState, prevCellState):
        """
        Function used for executing computation of one
        time step in hidden state
        :param input: Input vec at current time step.
        :param prevHiddenState: Vec of hidden state at previous time step.
        :param prevCellState: Vec of cell state at previous time step.
        """
        # First project from dimEmbedding to dimInput
        #print "W to input: ", np.asarray(self.W_toInput.eval()), " ", np.asarray(self.b_toInput.eval())
        input = T.dot(input, self.W_toInput.T) + self.b_toInput

        #print "Input: ", np.asarray(input.eval())

        forgetGate = T.nnet.sigmoid(T.dot(input, self.W_f.T) + T.dot(prevHiddenState, self.U_f.T)
                                    + self.b_f) # Should be (numSamples, dimHidden)
        inputGate = T.nnet.sigmoid(T.dot(input, self.W_i.T) + T.dot(prevHiddenState, self.U_i.T)
                                    + self.b_i) # Ditto
        candidateVals = T.tanh(T.dot(input, self.W_c.T) + T.dot(prevHiddenState, self.U_c.T)
                                + self.b_c) # Ditto
        candidateVals = forgetGate * prevCellState + inputGate * candidateVals # Ditto
        output = T.nnet.sigmoid(T.dot(input, self.W_o.T) + T.dot(prevHiddenState, self.U_o.T)
                                 + self.b_o) # Ditto
        hiddenState = output * T.tanh(candidateVals) # Ditto

        return hiddenState, candidateVals


    def forwardRun(self, inputMat, timeSteps):
        """
        Executes forward computation for designated number of time steps.
        Returns output vectors for all timesteps.
        :param inputMat: Input matrix of dimension (numTimesteps, numSamples, dimProj)
        :param timeSteps: Number of timesteps to use for unraveling each of 'numSamples'
        :param numSamples:  Number of samples to do forward computation for this batch
        """
        # Outputs of premise layer passed as input to hypothesis layer
        if self.hiddenInit is None and self.candidateValInit is None:
            hiddenInit = T.unbroadcast(T.alloc(np.cast[theano.config.floatX](1.), inputMat.shape[1], self.dimHidden),0)
            candidateValsInit = T.unbroadcast(T.alloc(np.cast[theano.config.floatX](1.), inputMat.shape[1], self.dimHidden), 0)
        else:
            hiddenInit = self.hiddenInit # Not sure if this is right...
            candidateValsInit = self.candidateValInit

        modelOut, updates = theano.scan(self._step,
                                sequences=[inputMat],
                                outputs_info=[hiddenInit, candidateValsInit], # Running a batch of samples at a time
                                name="layers",
                                n_steps=timeSteps)


        self.finalCandidateVal = modelOut[1][-1]
        self.finalHiddenVal = modelOut[0][-1]

        return modelOut, updates


    def projectToCategories(self):
        """
        Takes the final output of the forward run of an LSTM layer and projects
         to a vector of dim equal to number of categories we are classifying over.
        """
        catOutput = T.dot(self.finalHiddenVal, self.W_cat) + self.b_cat
        return catOutput


    def applySoftmax(self, catOutput):
        """
        Apply softmax to final vector of outputs
        :return:
        """
        softmaxOut = T.nnet.softmax(catOutput)
        return softmaxOut


    def computeCrossEntropyCost(self, yPred, yTarget):
        """
        Given predictions returned through softmax projection, compute
        cross entropy cost
        :param yPred: Output from LSTM with softmax applied
        :return: Loss for given predictions and targets
        """
        return T.nnet.categorical_crossentropy(yPred, yTarget).mean()


    def computeAccuracy(self, yPred, yTarget):
        """
        Computes accuracy for target and predicted values
        :param yPred:
        :param yTarget:
        :return:
        """
        return T.mean(T.eq(T.argmax(yPred, axis=-1), T.argmax(yTarget, axis=-1)))


    def costFunc(self, inputPremise, inputHypothesis, yTarget, layer, numTimesteps=1):
        """
        Compute end-to-end cost function for a collection of input data.
        :param layer: whether we are doing a forward computation in the
                        premise or hypothesis layer
        :return: Symbolic expression for cost function as well as theano function
                 for computing cost expression.
        """
        if layer == "premise":
            _ = self.forwardRun(inputPremise, numTimesteps)
        elif layer == "hypothesis":
            _ = self.forwardRun(inputHypothesis, numTimesteps)
        catOutput = self.projectToCategories()
        softmaxOut = self.applySoftmax(catOutput)
        cost = self.computeCrossEntropyCost(softmaxOut, yTarget)
        #theano.printing.pydotprint(cost, outfile="costGraph")
        return cost, theano.function([inputPremise, inputHypothesis, yTarget],
                                     cost, name='LSTM_cost_function')


    def computeGrads(self, inputPremise, inputHypothesis, yTarget, cost):
        """
        Computes gradients for cost function with respect to all parameters.
        :param costFunc:
        :return:
        """
        grads = T.grad(cost, wrt=self.params.values())
        # Clip grads to specific range to avoid parameter explosion
        gradsClipped= [T.clip(g, -3., 3.) for g in grads]

        gradsFn = theano.function([inputPremise, inputHypothesis, yTarget],
                                   gradsClipped, name='gradsFn')
        return grads, gradsFn


    def sgd(self, grads, learnRate):
        """
        Return SGD updates to parameters of model.
        :return: Function that can be used to compute sgd updates of parameters
        """
        paramUpdate = [(param, param - learnRate*grad) for param, grad in
                       zip(self.params.values(), grads)]

        return paramUpdate #theano.function([learnRate], [], updates=paramUpdate, name="sgdParamUpdate")


    def rmsprop(self, grads, learnRate, inputPremise, inputHypothesis, yTarget, cost):
        """
        Return RMSprop updates for parameters of model.
        :param grads:
        :param learnRate:
        :return:
        """
        zippedGrads = []
        runningGrads2 = []
        updir = []
        for k, p in self.params.iteritems():
            paramPrefix, layerName = k.split("_")

            # Super hacky... to get the broadcasting to be compatible
            if paramPrefix[0:4] == "bias":
                zippedGrads.append(theano.shared(p.get_value() * np.asarray(0.),
                        name="%s_grad" %k, broadcastable=(True, False)))
                runningGrads2.append(theano.shared(p.get_value() * np.asarray(0.),
                        name="%s_rgrad2" %k, broadcastable=(True, False)))
                updir.append(theano.shared(p.get_value() * np.asarray(0.),
                        name="%s_updir" %k, broadcastable=(True, False)))
            else:
                zippedGrads.append(theano.shared(p.get_value() * np.asarray(0.),
                        name="%s_grad" %k))
                runningGrads2.append(theano.shared(p.get_value() * np.asarray(0.),
                        name="%s_rgrad2" %k))
                updir.append(theano.shared(p.get_value() * np.asarray(0.),
                        name="%s_updir" %k))


        zgUpdate = [(zg, g) for zg, g in zip(zippedGrads, grads)]
        rg2Update = [(rg2, 0.95 * rg2 + 0.05 * (g ** 2))
             for rg2, g in zip(runningGrads2, grads)]

        # Computes cost but does not update params
        fGradShared = theano.function([inputPremise, inputHypothesis, yTarget], cost,
                                    updates=zgUpdate + rg2Update,
                                    name='rmspropFGradShared')

        updirNew = [(ud, zg / T.sqrt(rg2 + 1e-4))
                for ud, zg, rg2 in zip(updir, zippedGrads, runningGrads2)]

        paramUpdate = [(p, p - learnRate * udn[1])
                for p, udn in zip(self.params.values(), updirNew)]

        fUpdate = theano.function([learnRate], [], updates=updirNew + paramUpdate,
                               on_unused_input='ignore',
                               name='rmsprop_f_update')

        return fGradShared, fUpdate