""" Defines the hidden layers used by the model.
"""
import numpy as np
import theano
import theano.tensor as T


# Set random seed for determinism
SEED = 100
np.random.seed(SEED)

class HiddenLayer(object):
    def __init__(self, dimInput, dimHiddenState):
        """

        :param dimInput: Dimension of vector of input to hidden cell.
        :param dimState: Dimension of hidden HiddenState.
        """
        # Dictionary of model parameters.
        self.params = {}

        self.outputVector = None
        self.inputVector = None

        # TODO: what to use for initializing parameters (all zeros or random?)

        # Parameters for forget gate
        self.biasForget = T.shared(np.randn(dimHiddenState))
        self.weightsForget = T.shared(np.randn(dimHiddenState, (dimInput
                                                                + dimHiddenState)))

        # Parameters for input gate
        self.biasInput = T.shared(np.randn(dimHiddenState))
        self.weightsInput = T.shared(np.randn(dimHiddenState, (dimInput
                                                                + dimHiddenState)))

        # Parameters for candidate values
        self.biasCandidate = T.shared(np.randn(dimHiddenState))
        self.weightsCandidate = T.shared(np.randn(dimHiddenState, (dimInput
                                                                + dimHiddenState)))

        # Parameters for final output vector transform (for final
        # classification)
        self.biasOutputTransform = T.shared(np.randn(dimHiddenState))
        self.weightsOutputTransform = T.shared(np.randn(dimHiddenState, (dimInput
                                                                + dimHiddenState)))

        self.params["biasInput"] = self.biasInput
        self.params["weightsInput"] = self.weightsInput

        self.params["biasForget"] = self.biasForget
        self.params["weightsForget"] = self.weightsForget

        self.params["biasCandidate"] = self.biasCandidate
        self.params["weightsCandidate"] = self.weightsCandidate

        self.params["biasOutputTransform"] = self.biasOutputTransform
        self.params["weightsOutputTransform"] = self.weightsOutputTransform

