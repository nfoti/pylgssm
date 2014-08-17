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


class LGSSMFixedSarkkaPython(LGSSMFixedPython):
    """ Implement RTS Smoother as in "Bayesian Filtering and Smooting" book.
        This coressponds to the sequential version in Beal's thesis.
    """

    def kalman_filter(self, didx=None):
        """ Follows algorithm on pg. 57 of "Bayesian Filtering and Smoothing.

            didx: int or iterable of ints, subset of observations sequences to
                  consider.
            
            Updates `_state_filter_dists` member variable.

            TODO: Return the filtering distributions.
        """
        C = self._C
        A = self._A
        Q = self._Q
        R = self._R
        p, K = C.shape
        mu_0 = self._mu_0
        S_0 = self._S_0

        if didx is  None:
            datas = self._datas
        elif type(didx) is int:
            datas = self._data[didx]
        else:
            # This will barf if didx is not an iterable of ints
            datas = [self._data[idx] for idx in didx]

        Ts = [d.shape[0] for d in datas]

        filter_dists = list()

        for i, (data, T) in enumerate(zip(datas, Ts)):
            kal_m = np.empty((T, K))
            kal_P = np.empty((T, K, K))

            # Initial prediction step
            mbar = np.dot(A, mu_0)
            Pbar = np.dot(A, np.dot(S_0, A.T)) + Q

            for t in xrange(T):

                # Update step (4.21)
                y_t = np.atleast_1d(data[t,:])
                v_t = y_t - np.dot(C, mbar)
                S_t = np.dot(C, np.dot(Pbar, C.T)) + R

                K_t = np.dot(Pbar, np.dot(C.T, np.linalg.inv(S_t)))

                kal_m[t,:] = mbar + np.dot(K_t, v_t)
                kal_P[t,:,:] = Pbar - np.dot(K_t, np.dot(S_t, K_t.T))

                # Prediction step for next time step depends on t
                mbar = np.dot(A, kal_m[t,:])
                Pbar = np.dot(A, np.dot(kal_P[t], A.T)) + Q

            filter_dists.append((kal_m, kal_P))

        return filter_dists

    def rts_smoother(self):
        """ Follows algorithm on pg. 57 of "Bayesian Filtering and Smoothing.

            didx: int or iterable of ints, subset of observations sequences to
                  consider.

            Returns means and covariances for smoothing distributions.
        """

        return None



class LGSSMFixedBealPython(LGSSMFixedPython):
    """ Implements RTS smoother using a forward messages, \alpha(x_t) and
        backward messages, \beta(x_t), so that
        p(x_t | y_1:T) \propto \alpha(x_t)*\beta(x_t).  This is what the rest
        of our algorithms will look like, I just wanted two implementations to
        compare against.
    """
    pass
