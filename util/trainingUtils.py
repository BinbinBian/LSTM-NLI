"""
A bunch of utility functions used across different training schemes
"""
import numpy as np
import theano
import theano.tensor as T


def computeGrads(inputPremise, inputHypothesis, yTarget, cost, gradMax, params):
        """
        Computes gradients for cost function with respect to all parameters.
        :param gradMax: maximum gradient magnitude to use for clipping
        :param params: list of params to derive with respect to
        :return:
        """
        grads = T.grad(cost, wrt=params)
        # Clip grads to specific range to avoid parameter explosion
        gradsClipped = [T.clip(g, -gradMax, gradMax) for g in grads]

        gradsFn = theano.function([inputPremise, inputHypothesis, yTarget],
                                   gradsClipped, name='gradsFn')
        return grads, gradsFn

# TODO: probably include different optimization techniques here

def rmsprop(grads, learnRate, inputPremise, inputHypothesis, yTarget, cost, params):
        """
        Return RMSprop updates for parameters of model.
        :param grads:
        :param learnRate:
        :param params: dict of params to derive with respect to
        :return:
        """
        zippedGrads = []
        runningGrads2 = []
        updir = []
        for k, p in params.iteritems():
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
                for p, udn in zip(params.values(), updirNew)]

        fUpdate = theano.function([learnRate], [], updates=updirNew + paramUpdate,
                               on_unused_input='warn',
                               name='rmsprop_f_update')

        return fGradShared, fUpdate