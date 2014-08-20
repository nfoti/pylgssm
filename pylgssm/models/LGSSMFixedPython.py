from __future__ import division

from .LGSSMBase import LGSSMBase

import numpy as np

class LGSSMFixedPython(LGSSMBase):
    """ Python implementation with fixed parameters.

        Model parameters A, C, Q, R, mu_0, and S_0 are assumed fixed.
    """

    def __init__(self, A, C, Q=None, R=None, mu_0=None, S_0=None):
        """
            A    : K x K latent transition matrix
            C    : p x K latent to observation matrix
            Q    : K x K latent covariance matrix
            R    : p x p observation noise covariance
            mu_0 : K vector, mean of starting latent state (no associated
                   observation)
            S_0  : K x K covariance matrix for initiaion latent state
        """
        super(LGSSMFixedPython, self).__init__()
        # Put some error checking in here
        self._A = A
        self._C = C
        p, K = C.shape
        self._p = p
        self._K = K
        self._Q = Q if Q is not None else np.eye(K)
        self._R = R if R is not None else np.eye(p)
        self._mu_0 = mu_0 if mu_0 is not None else np.zeros(K)
        self._S_0 = S_0 if S_0 is not None else np.eye(K)

    def sample(self, T):
        A = self._A
        C = self._C
        Q = self._Q
        R = self._R
        mu_0 = self._mu_0
        S_0 = self._S_0
        p, K = C.shape

        V = np.random.multivariate_normal(np.zeros(K), Q, size=T)
        W = np.random.multivariate_normal(np.zeros(p), R, size=T)

        X = np.empty((T, K))
        Y = np.empty((T, p))

        # Initial state with no observation
        X0 = mu_0 + np.random.multivariate_normal(np.zeros(K), S_0)
        
        X[0,:] = np.dot(A, X0) + V[0,:]
        Y[0,:] = np.dot(C, X[0,:]) + W[0,:]
        for t in xrange(1,T):
            X[t,:] = np.dot(A, X[t-1,:]) + V[t,:]
            Y[t,:] = np.dot(C, X[t,:]) + W[t,:]

        return X, Y
