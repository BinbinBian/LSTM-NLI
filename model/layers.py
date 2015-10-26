""" Defines the hidden layers used by the model.
"""
import numpy as np
import theano
import theano.tensor as T


# Set random seed for determinism
SEED = 100
np.random.seed(SEED)

class HiddenLayer(object):
    def __init__(self, inputMat, dimInput, dimHiddenState):
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
        self.inputs = inputMat

        self.timeSteps = inputMat.shape[0]

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


    def _step(self):
        """
        Function used for executing computation of one
        time step in hidden state
        """

    def forwardRun(self):
        """
        Executes forward computation for designated number of time steps.
        Returns output vectors for all timesteps.
        """
        # Will make a call to theano scan function for stepping

