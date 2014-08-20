from __future__ import division

from .LGSSMFixedPython import LGSSMFixedPython

import numpy as np


class LGSSMFixedSarkkaPython(LGSSMFixedPython):
    """ Implement RTS Smoother as in "Bayesian Filtering and Smooting" book.
        This coressponds to the sequential version in Beal's thesis.
    """

    def kalman_filter(self, didx=None):
        """ Follows algorithm on pg. 57 of "Bayesian Filtering and Smoothing.

            didx: int or iterable of ints, subset of observations sequences to
                  consider.
            
            Returns:
                filtering_dists : list of tuples, one for each observation
                                  sequence, containing the means and covariances
                                  of the filtering distribution.
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
            Mf = np.empty((T, K))
            Pf = np.empty((T, K, K))

            # Initial prediction step
            mbar = np.dot(A, mu_0)
            Pbar = np.dot(A, np.dot(S_0, A.T)) + Q

            for t in xrange(T):

                # Update step (4.21)
                y_t = np.atleast_1d(data[t,:])
                v_t = y_t - np.dot(C, mbar)
                S_t = np.dot(C, np.dot(Pbar, C.T)) + R

                K_t = np.dot(Pbar, np.dot(C.T, np.linalg.inv(S_t)))

                Mf[t,:] = mbar + np.dot(K_t, v_t)
                Pf[t,:,:] = Pbar - np.dot(K_t, np.dot(S_t, K_t.T))

                # Prediction step for next time step depends on t
                mbar = np.dot(A, Mf[t,:])
                Pbar = np.dot(A, np.dot(Pf[t], A.T)) + Q

            filter_dists.append((Mf, Pf))

        return filter_dists

    def rts_smoother(self, didx=None, filter_dists=None):
        """ Follows algorithm on pg. 57 of "Bayesian Filtering and Smoothing.

            didx : int or iterable of ints, subset of observations sequences to
                  consider.
            filter_dists : list of filtering distributions for the observation
                           sequences (as produced by `kalman_filter` above).
                           If not provided `kalman_filter` is called.

            Returns:
            
            smoothing_dists : list of tuples, one for each observation
                              sequence, containing the means and covariances of
                              the smoothing distributions.
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

        # `kalman_filter` handles didx properly
        if filter_dists is None:
            filter_dists = self.kalman_filter(didx)

        Ts = [d.shape[0] for d in datas]

        smoothing_dists= list()

        for i, (data, T) in enumerate(zip(datas, Ts)):
            Ms = np.empty((T, K))
            Ps = np.empty((T, K, K))
            
            Mf = filter_dists[i][0]
            Pf = filter_dists[i][1]

            Ms[-1,:] = Mf[-1,:]
            Ps[-1,:,:] = Pf[-1,:,:]

            for t in reversed(xrange(T-1)):
                m_t = Mf[t,:]
                P_t = Pf[t,:,:]

                mbar = np.dot(A, m_t)
                Pbar = np.dot(A, np.dot(P_t, A.T)) + Q

                G_t = np.dot(P_t, np.dot(A.T, np.linalg.inv(Pbar)))

                Ms[t,:] = m_t + np.dot(G_t, Ms[t+1,:] - mbar)
                Ps[t,:,:] = P_t + np.dot(G_t, np.dot(Ps[t+1,:,:] - Pbar, G_t.T))

            smoothing_dists.append((Ms, Ps))

        return smoothing_dists
