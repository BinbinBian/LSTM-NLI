""" Defines the hidden layers used by the model.
"""
import numpy as np
import theano
import theano.tensor as T


# Set random seed for deterministic runs
SEED = 100
np.random.seed(SEED)

class HiddenLayer(object):
    def __init__(self, dimInput, dimHiddenState, layerName, numCategories=3):
        """
        :param inputMat: Matrix of input vectors to use for unraveling
                         hidden layer.
        :param dimInput: Dimension of vector of input to hidden cell.
        :param dimState: Dimension of hidden HiddenState.
        :param layerName: Name of current LSTM layer ('premise', 'hypothesis')
        """
        # Dictionary of model parameters.
        self.params = {}
        self.layerName = layerName
        self.inputDim = dimInput
        self.dimHidden = dimHiddenState

        # Represents number of categories used for classification
        self.numLabels = numCategories

        # Initialization values for hidden layer compute unit parameters
        self.hiddenInit = None
        self.candidateValInit = None

        # TODO: what to use for initializing parameters (random?)

        # TODO: include parameters for projecting from input word vector dimensions to dimInput

        # Parameters for forget gate
        self.b_f = theano.shared(np.random.randn(1, dimHiddenState), name="biasF_"+layerName, broadcastable=(True, False))
        self.W_f = theano.shared(np.random.randn(dimHiddenState, dimInput),
                                      name="weightsXf_"+layerName)
        self.U_f = theano.shared(np.random.randn(dimHiddenState, dimHiddenState),
                                 name="weightsHf_"+layerName)

        # Parameters for input gate
        self.b_i = theano.shared(np.random.randn(1, dimHiddenState), name="biasI_"+layerName, broadcastable=(True, False))
        self.W_i = theano.shared(np.random.randn(dimHiddenState, dimInput),
                                     name="weightsXi_"+layerName)
        self.U_i = theano.shared(np.random.randn(dimHiddenState, dimHiddenState),
                                 name="weightsHi_"+layerName)


        # Parameters for candidate values
        self.b_c = theano.shared(np.random.randn(1, dimHiddenState),
                                      name="biasC_"+layerName, broadcastable=(True, False))
        self.W_c = theano.shared(np.random.randn(dimHiddenState, dimInput),
                                         name="weightsXc_"+layerName)
        self.U_c = theano.shared(np.random.randn(dimHiddenState, dimHiddenState),
                                 name="weightsHc_"+layerName)

        # Parameters for final output vector transform (for final
        # classification)
        self.b_o = theano.shared(np.random.randn(1, dimHiddenState),
                                            name="biasO_"+layerName, broadcastable=(True, False))
        self.W_o = theano.shared(np.random.randn(dimHiddenState, dimInput),
                                               name="weightsXo_"+layerName)
        self.U_o = theano.shared(np.random.randn(dimHiddenState, dimHiddenState),
                                 name="weightsHo_"+layerName)


        # Parameters for linear projection from output of forward pass to a
        # vector with dimension equal to number of categories being classified
        # via one more softmax
        self.b_cat = theano.shared(np.random.randn(1, self.numLabels),
                                            name="biasCat_"+layerName, broadcastable=(True, False))
        self.W_cat = theano.shared(np.random.randn(dimHiddenState, self.numLabels),
                                               name="weightsCat_"+layerName)


        self.finalCandidateVal = None # Stores final cell state from scan in forwardPass
        self.finalHiddenVal = None  # Stores final hidden state from scan in forwardPass

        # Add shared vars to params dict
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


    def printParams(self):
        """
        Prints current parameters of model for debugging purposes.
        """
        print "Current parameter values for %s" %(self.layerName)
        print "-" * 50
        for pName, pValue in self.params.iteritems():
            print pName, " : ", pValue.eval()

        print "-" * 50


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
        #print "Input mat shape: ", inputMat.shape.eval()

        # Outputs of premise layer passed as input to hypothesis layer
        if self.hiddenInit is None and self.candidateValInit is None:
            hiddenInit = T.unbroadcast(T.alloc(np.cast[theano.config.floatX](1.), inputMat.shape[1], self.dimHidden),0)
            candidateValsInit = T.unbroadcast(T.alloc(np.cast[theano.config.floatX](1.), inputMat.shape[1], self.dimHidden), 0)
        else:
            hiddenInit = self.hiddenInit
            candidateValsInit = self.candidateValInit

        #print hiddenInit.eval()
        #print candidateValsInit.eval()
        modelOut, updates = theano.scan(self._step,
                                sequences=[inputMat],
                                outputs_info=[hiddenInit, candidateValsInit], # Running a batch of samples at a time
                                name="layers",
                                n_steps=timeSteps)

        self.finalCandidateVal = modelOut[1][-1] #[0][-1]
        self.finalHiddenVal = modelOut[0][-1] # [1][-1]

        return modelOut, updates


    def projectToCategories(self):
        """
        Takes the final output of the forward run of an LSTM layer and projects
         to a vector of dim equal to number of categories we are classifying over.
        """
        #print "Final Hidden: ", self.finalHiddenVal.eval()
        #print "Final W cat: ", self.W_cat.eval()
        #print "Final b cat: ", self.b_cat.eval()
        catOutput = T.dot(self.finalHiddenVal, self.W_cat) + self.b_cat
        return catOutput


    def getPredictions(self, catOutput):
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
        #print "Y pred: ", yPred.eval()
        #print "Y target: ", yTarget.eval()
        return T.nnet.categorical_crossentropy(yPred, yTarget).mean()


    def computeAccuracy(self, yPred, yTarget):
        """
        Computes accuracy for target and predicted values
        :param yPred:
        :param yTarget:
        :return:
        """
        return T.mean(T.eq(T.argmax(yPred, axis=-1), T.argmax(yTarget, axis=-1)))


    def costFunc(self, x, yTarget, numTimesteps=1):
        """
        Compute end-to-end cost function for a collection of input data.
        :return: Symbolic expression for cost function as well as theano function
                 for computing cost expression.
        """
        _ = self.forwardRun(x, numTimesteps) # Last parameter is num Samples -- may want to remove that
        catOutput = self.projectToCategories()
        softmaxOut = self.getPredictions(catOutput)
        cost = self.computeCrossEntropyCost(softmaxOut, yTarget)
        theano.printing.pydotprint(cost, outfile="costGraph")
        return cost #, theano.function([x, yTarget], cost, name='LSTM_cost_function')


    def computeGrads(self, x, yTarget, cost):
        """
        Computes gradients for cost function with respect to all parameters.
        :param costFunc:
        :return:
        """
        theano.pp(cost)
        grads = T.grad(cost, wrt=self.params.values())
        #costGrad = theano.function([x, yTarget], grads, name='costGradients')
        return grads #, costGrad


    def sgd(self, grads, learnRate):
        """
        Apply SGD updates to parameters of model.
        :return: Function that can be used to compute sgd updates of parameters
        """
        paramUpdate = [(param, param - learnRate*grad) for param, grad in
                       zip(self.params.values(), grads)]

        return paramUpdate #theano.function([learnRate], [], updates=paramUpdate, name="sgdParamUpdate")





