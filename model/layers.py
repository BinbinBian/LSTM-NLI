""" Defines the hidden layers used by the model.
"""
import numpy as np
import theano
import theano.tensor as T


# Set random seed for deterministic runs
SEED = 100
np.random.seed(SEED)

class HiddenLayer(object):
    def __init__(self, dimInput, dimHiddenState, layerName):
        """
        :param inputMat: Matrix of input vectors to use for unraveling
                         hidden layer.
        :param dimInput: Dimension of vector of input to hidden cell.
        :param dimState: Dimension of hidden HiddenState.
        :param layerName: Name of current LSTM layer ('premise', 'hypothesis')
        """
        # Dictionary of model parameters.
        self.params = {}

        self.inputDim = dimInput
        self.dimHidden = dimHiddenState

        self.outputs = None

        # TODO: what to use for initializing parameters (all zeros or random?)

        # Parameters for forget gate
        self.b_f= theano.shared(np.random.randn(1, dimHiddenState), name="biasForget_"+layerName, broadcastable=(True, False))
        self.W_f= theano.shared(np.random.randn(dimHiddenState, (dimInput
                                                                + dimHiddenState)),
                                      name="weightsForget_"+layerName)

        # Parameters for input gate
        self.b_i= theano.shared(np.random.randn(1, dimHiddenState), name="biasInput_"+layerName, broadcastable=(True, False))
        self.W_i= theano.shared(np.random.randn(dimHiddenState, (dimInput
                                                                + dimHiddenState)),
                                     name="weightsInput_"+layerName)

        # Parameters for candidate values
        self.b_c= theano.shared(np.random.randn(1, dimHiddenState),
                                      name="biasCandidate_"+layerName, broadcastable=(True, False))
        self.W_c= theano.shared(np.random.randn(dimHiddenState, (dimInput
                                                                + dimHiddenState)),
                                         name="weightsCandidate_"+layerName)

        # Parameters for final output vector transform (for final
        # classification)
        self.b_o= theano.shared(np.random.randn(1, dimHiddenState),
                                            name="biasOutputTransform_"+layerName, broadcastable=(True, False))
        self.W_o= theano.shared(np.random.randn(dimHiddenState, (dimInput
                                                                + dimHiddenState)),
                                               name="weightsOutputTransform_"+layerName)

        # Add shared vars to params dict
        self.params["biasInput_"+layerName] = self.b_i
        self.params["weightsInput_"+layerName] = self.W_i

        self.params["biasForget_"+layerName] = self.b_f
        self.params["weightsForget_"+layerName] = self.W_f

        self.params["biasCandidate_"+layerName] = self.b_c
        self.params["weightsCandidate_"+layerName] = self.W_c

        self.params["biasOutputTransform_"+layerName] = self.b_o
        self.params["weightsOutputTransform_"+layerName] = self.W_o


    def _step(self, input, prevHiddenState, prevCellState):
        """
        Function used for executing computation of one
        time step in hidden state
        :param input: Input vec at current time step.
        :param prevHiddenState: Vec of hidden state at previous time step.
        :param prevCellState: Vec of cell state at previous time step.
        """
        combinedState = T.concatenate([prevHiddenState, input], axis=1) # Should be (numSamples, dimHidden + dimInput)
        forgetGate = T.nnet.sigmoid(T.dot(combinedState, self.W_f.T)
                                    + self.b_f) # Should be (numSamples, dimHidden)
        inputGate = T.nnet.sigmoid(T.dot(combinedState, self.W_i.T) +
                                    self.b_i) # Ditto
        candidateVals = T.tanh(T.dot(combinedState, self.W_c.T) +
                                self.b_c) # Ditto
        candidateVals = forgetGate * prevCellState + inputGate * candidateVals # Ditto
        output = T.nnet.sigmoid(T.dot(combinedState, self.W_o.T) +
                                 self.b_o) # Ditto
        hiddenState = output * T.tanh(candidateVals) # Ditto

        return hiddenState, candidateVals

    def forwardRun(self, inputMat, timeSteps, numSamples):
        """
        Executes forward computation for designated number of time steps.
        Returns output vectors for all timesteps.
        :param inputMat: Input matrix of dimension (numSamples, dimProj)
        :param timeSteps: Number of timesteps to use for unraveling each of 'numSamples'
        :param numSamples:  Number of samples to do forward computation for this batch
        """
        # modelOut, updates = theano.scan(HiddenLayer._step,
        #                         sequences=[inputMat],
        #                         outputs_info=[T.alloc(np.array(0.),
        #                                                   numSamples,
        #                                                   self.dimHidden),
        #                                      T.alloc(np.array(0.),
        #                                                   numSamples,
        #                                                   self.dimHidden)], # Running a batch of samples at a time
        #                         name="layers",
        #                         n_steps=timeSteps)

        modelOut, updates = theano.scan(HiddenLayer._step,
                                sequences=[inputMat],
                                outputs_info=[T.alloc(np.array(0.),
                                                          numSamples,
                                                          self.dimHidden)], # Running a batch of samples at a time
                                name="layers",
                                n_steps=timeSteps)

        self.outputs = modelOut[-1]
        return modelOut, updates

    # TODO: Must work out cost before I can do optimization via SGD, etc.