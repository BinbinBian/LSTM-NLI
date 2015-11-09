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
        self.b_f= T.shared(np.randn(dimHiddenState), name="biasForget_"+layerName)
        self.W_f= T.shared(np.randn(dimHiddenState, (dimInput
                                                                + dimHiddenState)),
                                      name="weightsForget_"+layerName)

        # Parameters for input gate
        self.b_i= T.shared(np.randn(dimHiddenState), name="biasInput_"+layerName)
        self.W_i= T.shared(np.randn(dimHiddenState, (dimInput
                                                                + dimHiddenState)),
                                     name="weightsInput_"+layerName)

        # Parameters for candidate values
        self.b_c= T.shared(np.randn(dimHiddenState),
                                      name="biasCandidate_"+layerName)
        self.W_c= T.shared(np.randn(dimHiddenState, (dimInput
                                                                + dimHiddenState)),
                                         name="weightsCandidate_"+layerName)

        # Parameters for final output vector transform (for final
        # classification)
        self.b_o= T.shared(np.randn(dimHiddenState),
                                            name="biasOutputTransform_"+layerName)
        self.W_o= T.shared(np.randn(dimHiddenState, (dimInput
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

        combinedState = T.stack([prevHiddenState, input]) # Check that this performs same operation as np.concatenate
        forgetGate = T.nnet.sigmoid(T.dot(self.W_f, combinedState)
                                    + self.b_f)
        inputGate = T.nnet.sigmoid(T.dot(self.W_i, combinedState) +
                                   self.b_i)
        candidateVals = T.tanh(T.dot(self.W_c, combinedState) +
                               self.b_c)
        candidateVals = forgetGate * prevCellState + inputGate * candidateVals
        output = T.nnet.sigmoid(T.dot(self.W_o, combinedState) +
                                self.b_o)
        hiddenState = output * T.tanh(candidateVals)

        return hiddenState, candidateVals


    def forwardRun(self, inputMat, timeSteps, numSamples):
        """
        Executes forward computation for designated number of time steps.
        Returns output vectors for all timesteps.
        :param inputMat: Input matrix of dimension (numSamples, dimProj)
        :param timeSteps: Number of timesteps to use for unraveling each of 'numSamples'
        :param numSamples:  Number of samples to do forward computation for this batch
        """
        modelOut, updates = theano.scan(HiddenLayer._step,
                                sequences=[inputMat],
                                outputs_info=[T.alloc(np.array(0.),
                                                          numSamples,
                                                          self.dimHidden),
                                             T.alloc(np.array(0.),
                                                          numSamples,
                                                          self.dimHidden)], # Running a batch of samples at a time
                                name="layers",
                                n_steps=timeSteps)

        self.outputs = modelOut[-1] # TODO: Maybe only want the first (or last?) element of this list
        return modelOut, updates

    # TODO: Must work out cost before I can do optimization via SGD, etc.