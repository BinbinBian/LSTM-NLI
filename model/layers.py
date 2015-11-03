""" Defines the hidden layers used by the model.
"""
import numpy as np
import theano
import theano.tensor as T


# Set random seed for deterministic runs
SEED = 100
np.random.seed(SEED)

class HiddenLayer(object):
    def __init__(self, dimInput, dimHiddenState):
        """
        :param inputMat: Matrix of input vectors to use for unraveling
                         hidden layer.
        :param dimInput: Dimension of vector of input to hidden cell.
        :param dimState: Dimension of hidden HiddenState.
        """
        # Dictionary of model parameters.
        self.params = {}

        self.inputDim = dimInput
        self.dimHidden = dimHiddenState

        self.outputs = None

        # TODO: what to use for initializing parameters (all zeros or random?)

        # Parameters for forget gate
        self.biasForget = T.shared(np.randn(dimHiddenState), name="biasForget")
        self.weightsForget = T.shared(np.randn(dimHiddenState, (dimInput
                                                                + dimHiddenState)),
                                      name="weightsForget")

        # Parameters for input gate
        self.biasInput = T.shared(np.randn(dimHiddenState), name="biasInput")
        self.weightsInput = T.shared(np.randn(dimHiddenState, (dimInput
                                                                + dimHiddenState)),
                                     name="weightsInput")

        # Parameters for candidate values
        self.biasCandidate = T.shared(np.randn(dimHiddenState),
                                      name="biasCandidate")
        self.weightsCandidate = T.shared(np.randn(dimHiddenState, (dimInput
                                                                + dimHiddenState)),
                                         name="weightsCandidate")

        # Parameters for final output vector transform (for final
        # classification)
        self.biasOutputTransform = T.shared(np.randn(dimHiddenState),
                                            name="biasOutputTransform")
        self.weightsOutputTransform = T.shared(np.randn(dimHiddenState, (dimInput
                                                                + dimHiddenState)),
                                               name="weightsOutputTransform")

        # Add shared vars to params dict
        self.params["biasInput"] = self.biasInput
        self.params["weightsInput"] = self.weightsInput

        self.params["biasForget"] = self.biasForget
        self.params["weightsForget"] = self.weightsForget

        self.params["biasCandidate"] = self.biasCandidate
        self.params["weightsCandidate"] = self.weightsCandidate

        self.params["biasOutputTransform"] = self.biasOutputTransform
        self.params["weightsOutputTransform"] = self.weightsOutputTransform


    def _step(self, input, prevHiddenState, prevCellState):
        """
        Function used for executing computation of one
        time step in hidden state
        :param input: Input vec at current time step.
        :param prevHiddenState: Vec of hidden state at previous time step.
        :param prevCellState: Vec of cell state at previous time step.
        """

        combinedState = T.stack([prevHiddenState, input]) # Check that this performs same operation as np.concatenate
        forgetGate = T.nnet.sigmoid(T.dot(self.weightsForget, combinedState)
                                    + self.biasForget)
        inputGate = T.nnet.sigmoid(T.dot(self.weightsInput, combinedState) +
                                   self.biasInput)
        candidateVals = T.tanh(T.dot(self.weightsCandidate, combinedState) +
                               self.biasCandidate)
        candidateVals = forgetGate * prevCellState + inputGate * candidateVals
        output = T.nnet.sigmoid(T.dot(self.weightsOutputTransform, combinedState) +
                                self.biasOutputTransform)
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