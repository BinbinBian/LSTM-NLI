""" Defines the hidden layers used by the model.
"""
import numpy as np
import theano
import theano.tensor as T

from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from util.afs_safe_logger import Logger
from util.utils import HeKaimingInitializer, GaussianDefaultInitializer, computeParamNorms

# Set random seed for deterministic runs
SEED = 100
np.random.seed(SEED)
rng = RandomStreams(SEED)

logger = Logger(log_path="/Users/mihaileric/Documents/Research/LSTM-NLI/log/"
                         "experimentLog.txt")

# TODO: Refactor so that initialization of params is provided as an option

class LSTMLayer(object):
    def __init__(self, dimInput, dimHiddenState, dimEmbedding, layerName, dropoutMode, initializer,
                 numCategories=3):
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
        self.dropoutMode = dropoutMode

        # Represents number of categories used for classification
        self.numLabels = numCategories

        # Initialization values for hidden layer compute unit parameters
        self.outputInit = None
        self.cellStateInit = None

        # Parameters for projecting from embedding size to input dimension
        self.b_toInput = theano.shared(np.zeros((1, dimInput), dtype=np.float32), name="biasToInput_"+layerName,
                                        broadcastable=(True, False))
        self.W_toInput = theano.shared(initializer((dimInput, dimEmbedding)),
                                            name="weightsXtoInput_"+layerName)

        # Parameters for forget gate
        self.b_f = theano.shared(np.zeros((1, dimHiddenState), dtype=np.float32), name="biasF_"+layerName,
                                 broadcastable=(True, False))
        self.W_f = theano.shared(initializer((dimHiddenState, dimInput)),
                                                name="weightsXf_"+layerName)
        self.U_f = theano.shared(initializer((dimHiddenState,
                                        dimHiddenState)),
                                        name="weightsHf_"+layerName)

        # Parameters for input gate
        self.b_i = theano.shared(np.zeros((1, dimHiddenState), dtype=np.float32), name="biasI_"+layerName, broadcastable=(True, False))
        self.W_i = theano.shared(initializer((dimHiddenState, dimInput)),
                                                name="weightsXi_"+layerName)
        self.U_i = theano.shared(initializer((dimHiddenState,
                                        dimHiddenState)),
                                        name="weightsHi_"+layerName)


        # Parameters for candidate values
        self.b_c = theano.shared(np.zeros((1, dimHiddenState), dtype=np.float32), name="biasC_"+layerName, broadcastable=(True, False))
        self.W_c = theano.shared(initializer((dimHiddenState,
                                               dimInput)),
                                               name="weightsXc_"+layerName)
        self.U_c = theano.shared(initializer((dimHiddenState,
                                         dimHiddenState)),
                                         name="weightsHc_"+layerName)

        # Parameters for final output vector transform (for final
        # classification)
        self.b_o = theano.shared(np.zeros((1, dimHiddenState), dtype=np.float32), name="biasO_"+layerName, broadcastable=(True, False))
        self.W_o = theano.shared(initializer((dimHiddenState,
                                               dimInput)),
                                               name="weightsXo_"+layerName)
        self.U_o = theano.shared(initializer((dimHiddenState,
                                         dimHiddenState)),
                                         name="weightsHo_"+layerName)


        # Parameters for linear projection from output of forward pass to a
        # vector with dimension equal to number of categories being classified
        # via one more softmax
        self.b_cat = theano.shared(np.zeros((1, self.numLabels), dtype=np.float32), name="biasCat_"+layerName, broadcastable=(True, False))
        self.W_cat = theano.shared(initializer((dimHiddenState,
                                            self.numLabels)),
                                            name="weightsCat_"+layerName)


        self.finalCellState = None # Stores final cell state from scan in forwardRun
        self.finalOutputVal = None  # Stores final hidden state from scan in forwardRun

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

        # Keep track of names of parameters specific to LSTM cell
        self.LSTMcellParams = ["weightsXi_"+layerName, "weightsHi_"+layerName,
                               "weightsXf_"+layerName, "weightsHf_"+layerName,
                               "weightsXc_"+layerName, "weightsHc_"+layerName,
                               "weightsXo_"+layerName, "weightsHo_"+layerName]


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


    def setInitialLayerParams(self, outputInit, cellStateInit):
        """
        Sets the initialization hidden value and candidate value parameters
        :param hiddenValInit:
        :param candidateValInit:
        """
        self.outputInit = outputInit
        self.cellStateInit = cellStateInit


    def _step(self, input, prevHiddenState, prevCellState):
        """
        Function used for executing computation of one
        time step in hidden state
        :param input: Input vec at current time step.
        :param prevHiddenState: Vec of hidden state at previous time step.
        :param prevCellState: Vec of cell state at previous time step.
        """
        # First project from dimEmbedding to dimInput
        input = T.dot(input, self.W_toInput.T) + self.b_toInput

        forgetGate = T.nnet.sigmoid(T.dot(input, self.W_f.T) + T.dot(prevHiddenState, self.U_f.T)
                                    + self.b_f) # Should be (numSamples, dimHidden)
        inputGate = T.nnet.sigmoid(T.dot(input, self.W_i.T) + T.dot(prevHiddenState, self.U_i.T)
                                    + self.b_i) # Ditto
        candidateVals = T.tanh(T.dot(input, self.W_c.T) + T.dot(prevHiddenState, self.U_c.T)
                                + self.b_c) # Ditto
        cellState = forgetGate * prevCellState + inputGate * candidateVals # Ditto
        output = T.nnet.sigmoid(T.dot(input, self.W_o.T) + T.dot(prevHiddenState, self.U_o.T)
                                 + self.b_o) # Ditto
        hiddenState = output * T.tanh(cellState) # Ditto

        return hiddenState, cellState


    def forwardRun(self, inputMat, timeSteps):
        """
        Executes forward computation for designated number of time steps.
        Returns output vectors for all timesteps.
        :param inputMat: Input matrix of dimension (numTimesteps, numSamples, dimProj)
        :param timeSteps: Number of timesteps to use for unraveling each of 'numSamples'
        :param numSamples:  Number of samples to do forward computation for this batch
        """
        # Outputs of premise layer passed as input to hypothesis layer
        if self.outputInit is None and self.cellStateInit is None:
            outputInit = T.unbroadcast(T.alloc(np.cast[theano.config.floatX](0.), inputMat.shape[1], self.dimHidden),0)
            cellStateInit = T.unbroadcast(T.alloc(np.cast[theano.config.floatX](0.), inputMat.shape[1], self.dimHidden), 0)
        else:
            outputInit = self.outputInit
            cellStateInit = self.cellStateInit

            assert outputInit is not None
            assert cellStateInit is not None

        timestepOut, updates = theano.scan(self._step,
                                sequences=[inputMat],
                                outputs_info=[outputInit, cellStateInit], # Running a batch of samples at a time
                                name="layers",
                                n_steps=timeSteps)

        # Store the outputs for all timesteps as attribute
        self.allOutputs = timestepOut[0]

        self.finalCellState = timestepOut[1][-1]
        self.finalOutputVal = timestepOut[0][-1]

        return timestepOut, updates


    def projectToCategories(self):
        """
        Takes the final output of the forward run of an LSTM layer and projects
         to a vector of dim equal to number of categories we are classifying over.
        """
        catOutput = T.dot(self.finalOutputVal, self.W_cat) + self.b_cat
        return catOutput


    def computeCrossEntropyCost(self, catOutput, yTarget):
        """
        Given predictions returned through softmax projection, compute
        cross entropy cost
        :param yPred: Output from LSTM with softmax applied
        :return: Loss for given predictions and targets
        """
        yPred = T.nnet.softmax(catOutput)
        return T.nnet.categorical_crossentropy(yPred, yTarget).mean()


    def computeAccuracyLayer(self, yPred, yTarget):
        """
        Computes accuracy for target and predicted values
        :param yPred:
        :param yTarget:
        :return:
        """
        return T.mean(T.eq(T.argmax(yPred, axis=-1), T.argmax(yTarget, axis=-1)))


    # TODO: Make this into static method in utility module
    def applyDropout(self, tensor, mode, dropoutRate):
        """
        Apply dropout with given rate to given tensor, according to mode
        (which indicates whether you are training or testing). Return transformed
        tensor.
        :return:
        """
        # Explicit cast to float32 so that we don't accidentally get float64 tensor variables
        dropoutRate = theano.shared(np.array(dropoutRate).astype(np.float32))
        transformed = T.switch(mode,
            (tensor * rng.binomial(tensor.shape, p=dropoutRate, n=1, dtype=theano.config.floatX)),
            tensor * dropoutRate) # TODO: Make sure this refers to keep rate
        
        #print "tensor dtype: ", tensor.dtype
        #print "dropout rate dtype: ", dropoutRate.dtype
        #print "dropout rate: ", np.asarray(dropoutRate.eval())
        #print "transformed dtype: ", transformed.dtype
        return transformed


    def initSentAttnParams(self):
        """
        Initializes sentence attention parameters if sentence level
        attention is part of the model
        :return:
        """
        self.W_y = theano.shared(normal((self.dimHidden,
                                               self.dimHidden)),
                                               name="weightsWy_"+self.layerName)
        self.W_h = theano.shared(normal((self.dimHidden,
                                               self.dimHidden)),
                                               name="weightsWh_"+self.layerName)
        self.W_x = theano.shared(normal((self.dimHidden,
                                               self.dimHidden)),
                                               name="weightsWx_"+self.layerName)
        self.W_p = theano.shared(normal((self.dimHidden,
                                               self.dimHidden)),
                                               name="weightsWp_"+self.layerName)
        self.w = theano.shared(normal((self.dimHidden, 1)),
                               name="weightsAlphasoftmax_"+self.layerName)

        self.params["weightsWy_"+self.layerName] = self.W_y
        self.params["weightsWh_"+self.layerName] = self.W_h
        self.params["weightsWx_"+self.layerName] = self.W_x
        self.params["weightsWp_"+self.layerName] = self.W_p
        self.params["weightsAlphasoftmax_"+self.layerName] = self.w


        # TODO: May want to add params to self.LSTMparams for L2 regularization
    def applySentenceAttention(self, premiseOutputs, finalHypothesisOutput, numTimestepsPremise):
        """
        Apply sentence level attention by attending over all premise outputs
        once with the final hypothesis output. Note this is different from
        word-by-word attention over the premise.
        :param premiseOutputs:
        :param finalHypothesisOutput:
        :return:
        """
        # Note: Notation follows that in Rocktaschel's attention mechanism explanation:
        # http://arxiv.org/pdf/1509.06664v2.pdf
        timestep, numSamp, dimHidden = premiseOutputs.shape
        Y = premiseOutputs.reshape((numSamp, timestep, dimHidden))
        WyY = T.dot(Y, self.W_y) # Computing (WyY).T

        transformedHn = (T.dot(self.W_h, finalHypothesisOutput.T)).T
        repeatedHn = [transformedHn] * numTimestepsPremise
        # TODO: Condense this later if it works
        repeatedHn = T.stacklists(repeatedHn)
        repeatedHn = repeatedHn.dimshuffle(1, 0, 2) # (numSample, timestep, dimHidden)

        M = T.tanh(WyY + repeatedHn)
        alpha = T.nnet.softmax(T.dot(M, self.w).flatten(2)) # Hackery to make into 2d tensor of (numSamp, timestep)
        Y = Y.dimshuffle(0, 2, 1)
        rOut, updates = theano.scan(fn=lambda Yt, alphat: T.dot(Yt, alphat),
                                    outputs_info=None, sequences=[Y, alpha],
                                    non_sequences=None)
        WxHn = T.dot(finalHypothesisOutput, self.W_x)
        WpR = T.dot(rOut, self.W_p)
        hstar = T.tanh(WxHn + WpR)

        return hstar


    def initWordwiseAttnParams(self):
        """
        Initializes parameters for wordwise attention if specified as part
        of model.
        :return:
        """
        self.initSentAttnParams()
        self.W_r = theano.shared(normal((self.dimHidden,
                                               self.dimHidden)),
                                               name="weightsWr_"+self.layerName)
        self.W_t = theano.shared(normal((self.dimHidden,
                                               self.dimHidden)),
                                               name="weightsWt_"+self.layerName)


    # TODO: Get rid of print statements after testing on entire corpus and getting reasonable results
    def applyWordwiseAttention(self, premiseOutputs, hypothesisOutputs,
                               finalHypothesisOutput, batchSize,
                               numTimestepsPremise, numTimestepsHypothesis):
        """
        Apply word-by-word attention as described in 2.4 of Rocktaschel paper
        :param premiseOutputs:
        :param hypothesisOutputs:
        :param finalHypothesisOutput:
        :param numTimestepsPremise:
        :return:
        """
        timestep, numSamp, dimHidden = premiseOutputs.shape
        Y = premiseOutputs.reshape((numSamp, timestep, dimHidden))

        #print "Y shape beginning: ", Y.shape.eval()

        WyY = T.dot(Y, self.W_y) # Computing (WyY).T

        # TODO: How to initialize r_{t-1} -- probably not right
        r_t = theano.shared(normal((self.dimHidden, batchSize)), name="rt_"+self.layerName)

        #print "WyY: ", WyY.eval()
        # Iterate over hypothesis vector for every timestep
        for t in range(numTimestepsHypothesis):
            #print "-" * 100
            #print "Iter: ", t
            ht = hypothesisOutputs[t]
            #print "ht: ", ht.eval()
            #print "ht shape: ", ht.shape.eval()

            transformedHt = (T.dot(self.W_h, ht.T)).T # Modify here (11)

            #print "tranformed Ht: ", transformedHt.eval()
            #print "transformed ht shape: ", transformedHt.shape.eval()

            #print "Wr shape: ", self.W_r.shape.eval()
            #print "Wr: ", np.asarray(self.W_r.eval())

            #print "R_t shape: ", r_t.shape.eval()
            #print "R_t: ", np.asarray(r_t.eval())

            WrRt = (T.dot(self.W_r, r_t)).T # r_t.T?

            #print "WrRt: ", WrRt.eval()
            #print "WrRt shape: ", WrRt.shape.eval()
            #print "WrRt: ", WrRt.eval()

            transformedHtRt = transformedHt + WrRt

            #print "tranformed HtRt: ", transformedHtRt.eval()
            #print "transformed HtRt shape: ", transformedHtRt.shape.eval()
            #print "transformed HtRt : ", transformedHtRt.eval()

            premiseWeights = [transformedHtRt] * numTimestepsPremise
            # TODO: Condense this later if it works
            premiseWeights = T.stacklists(premiseWeights)
            #print "Premise weights: ", premiseWeights.eval()
            #print "premise weights shape: ", premiseWeights.shape.eval()
            #print "premise weights: ", premiseWeights.eval()

            premiseWeights = premiseWeights.dimshuffle(1, 0, 2) # (numSample, timestep, dimHidden)

            Mt = T.tanh(WyY + premiseWeights) # Modify here (11)

            #print "Mt: ", Mt.eval()
            #print "Mt shape: ", Mt.shape.eval()
            #print "Mt: ", Mt.eval()

            #print "w shape: ", self.w.shape.eval()
            #print "w: ", np.asarray(self.w.eval())

            #print "Mtw dotted shape: ", T.dot(Mt, self.w).shape.eval()

            #print "Mtw dotted/flattened shape: ", T.dot(Mt, self.w).flatten(2).shape.eval()
            alphat = T.nnet.softmax(T.dot(Mt, self.w).flatten(2)) # Hackery to make into 2d tensor of (numSamp, timestep)

            #print "Alpha: ", alphat.eval()
            #print "Alpha shape: ", alphat.shape.eval()

            Y = Y.dimshuffle(0, 2, 1)

            #print "Y shape: ", Y.shape.eval()

            rtOut, updates = theano.scan(fn=lambda Yt, alphat: T.dot(Yt, alphat),
                                    outputs_info=None, sequences=[Y, alphat],
                                    non_sequences=None)

            rtOut = rtOut.T
            #print "Rtout: ", rtOut.eval()
            #print "Rtout shape: ", rtOut.shape.eval()

            #print "Dot shape: ", T.dot(self.W_t, r_t).shape.eval()
            #print "Dot : ", T.dot(self.W_t, r_t).eval()

            r_t = rtOut + T.tanh(T.dot(self.W_t, r_t))

            #print "r_t shape: ", r_t.shape.eval()
            #print "R_t: ", np.asarray(r_t.eval())
            #r_t = r_t.T

            # Shuffle back to original orientation so next iteration doesn't die
            Y = Y.dimshuffle(0, 2, 1)

        WxHn = T.dot(finalHypothesisOutput, self.W_x)

        #print "WxHn: ", WxHn.eval()
        #print "WxHn shape : ", WxHn.shape.eval()

        WpR = T.dot(self.W_p, r_t).T

        #print "WpR: ", WpR.eval()
        #print "WpR shape : ", WpR.shape.eval()

        hstar = T.tanh(WxHn + WpR)

        return hstar


    def costFunc(self, inputPremise, inputHypothesis, yTarget, layer, L2regularization,
                 dropoutRate, premiseOutputs, batchSize, sentenceAttention=False, wordwiseAttention=False,
                 numTimestepsHypothesis=1, numTimestepsPremise=1):
        """
        Compute end-to-end cost function for a collection of input data.
        :param layer: whether we are doing a forward computation in the
                        premise or hypothesis layer
        :return: Symbolic expression for cost function as well as theano function
                 for computing cost expression.
        """
        if layer == "premise":
            _ = self.forwardRun(inputPremise, numTimestepsPremise)
        elif layer == "hypothesis":
            timestepOut, _ = self.forwardRun(inputHypothesis, numTimestepsHypothesis)

        # Apply sentence level attention -- notation consistent with paper
        if sentenceAttention:
            hstar = self.applySentenceAttention(premiseOutputs, self.finalOutputVal,
                                                numTimestepsPremise)
            self.finalOutputVal = hstar

        # Apply word by word attention
        if wordwiseAttention:
            hstar = self.applyWordwiseAttention(premiseOutputs, timestepOut[0],
                                                self.finalOutputVal, batchSize,
                                                numTimestepsPremise, numTimestepsHypothesis)
            self.finalOutputVal = hstar

        # Apply dropout here before projecting to categories? apply to finalOutputVal
        self.finalOutputVal = self.applyDropout(self.finalOutputVal, self.dropoutMode,
                                                dropoutRate)
        catOutput = self.projectToCategories()
        cost = self.computeCrossEntropyCost(catOutput, yTarget)

        # Get params specific to cell and add L2 regularization to cost
        LSTMparams = [self.params[cParam] for cParam in self.LSTMcellParams]
        cost = cost + computeParamNorms(LSTMparams, L2regularization)
        return cost, theano.function([inputPremise, inputHypothesis, yTarget],
                                     cost, name='LSTM_cost_function', on_unused_input="warn")


    # TODO: replace this with implementation in 'trainingUtils'
    def computeGrads(self, inputPremise, inputHypothesis, yTarget, cost, gradMax):
        """
        Computes gradients for cost function with respect to all parameters.
        :param costFunc:
        :param gradMax: maximum gradient magnitude to use for clipping
        :return:
        """
        grads = T.grad(cost, wrt=self.params.values())
        # Clip grads to specific range to avoid parameter explosion
        gradsClipped = [T.clip(g, -gradMax, gradMax) for g in grads]

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


    # TODO: replace this with implementation in 'trainingUtils'
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
                               on_unused_input='warn',
                               name='rmsprop_f_update')

        return fGradShared, fUpdate
