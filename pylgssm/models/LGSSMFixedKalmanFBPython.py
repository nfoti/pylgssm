from __future__ import division

from .LGSSMFixedPython import LGSSMFixedPython

import numpy as np


class LGSSMFixedKalmanFBPython(LGSSMFixedPython):
    """ Implements RTS smoother using a forward messages, \alpha(x_t) and
        backward messages, \beta(x_t), so that
        p(x_t | y_1:T) \propto \alpha(x_t)*\beta(x_t).  This is what the rest
        of our algorithms will look like, I just wanted two implementations to
        compare against.
    """

    def _forward_msgs(self, didx=None):
        """ Follows Algorithm 3 in Section 2.7 of Emily Fox's PhD thesis.

            didx: int or iterable of ints, subset of observations sequences to
                  consider.
            
            Returns:
                alphas : list of tuples, one for each observation
                                  sequence, containing the means and covariances
                                  of the filtering distribution.


            The messages are stored in information form so that we can combine
            them with the backward messages trivially.
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

        alphas = list()

        Qi = np.linalg.inv(Q)
        Ri = np.linalg.inv(R)
        Ai = np.linalg.inv(A)

        for i, (data, T) in enumerate(zip(datas, Ts)):
            Hs = np.empty((T, K))
            Ls = np.empty((T, K, K))

            Lf = np.linalg.inv(S_0)
            th_f = np.linalg.solve(S_0, mu_0)

            M = np.dot(Ai.T, np.dot(Lf, Ai))
            J = np.dot(M, np.linalg.inv(M + Qi))
            L = np.eye(K) - J

            for t in xrange(T):
                Lpred = np.dot(L, np.dot(M, L.T)) + np.dot(J, np.dot(Qi, J.T))
                th_pred = np.dot(L, np.dot(Ai.T, th_f))

                y = np.squeeze(data[t,:])
                Lf = Lpred + np.dot(C.T, np.dot(Ri, C))
                th_f = th_pred + np.dot(C.T, np.dot(Ri, y))

                Hs[t,:] = th_f
                Ls[t,:,:] = Lf

                M = np.dot(Ai.T, np.dot(Lf, Ai))
                J = np.dot(M, np.linalg.inv(M + Qi))
                L = np.eye(K) - J
            
            alphas.append((Hs, Ls))

        return alphas


    def kalman_filter(self, didx=None, forward_msgs=None):
        """ Follows algorithm in Section 2.7 of Emily Fox's PhD thesis.

            didx : int or iterable of ints, subset of observations sequences to
                  consider.
            forward_msgs : list of tuples, one for each obs. sequence,
                           containing the information form of the forward
                           messages.  This function turns these into means and
                           covariances to pass to the outside world.
            
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

        if forward_msgs is None:
            alphas = self._forward_msgs(didx)

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

            th_f, Lf = alphas[i]

            for t in xrange(T):
                Pf[t,:,:] = np.linalg.inv(Lf[t,:,:])
                Mf[t,:] = np.dot(Pf[t,:,:], th_f[t,:])

            filter_dists.append((Mf, Pf))

        return filter_dists

    def _backward_msgs(self, didx=None):
        """ Compute Beal's backward messages.

            didx: int or iterable of ints, subset of observations sequences to
                  consider.

            Returns:
                list of tuples, one for each obs. sequence, with the means and
                INFORMATION matrices of the backward messages.

            These are NOT the same as BP messages as they include the
            likelihood.
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

        backward_msgs = list()

        Qi = np.linalg.inv(Q)
        Ri = np.linalg.inv(R)
        Ai = np.linalg.inv(A)

        for i, (data, T) in enumerate(zip(datas, Ts)):
            
            Hs = np.empty((T, K))
            Ls = np.empty((T, K, K))

            Hs[-1,:] = np.zeros(K)
            Ls[-1,:,:] = np.zeros((K,K))
            
            Lb = np.zeros((K,K))
            th_b = np.zeros(K)

            AtQi = np.dot(A.T, Qi)
            QiA = np.dot(Qi, A)
            CtRi = np.dot(C.T, Ri)
            CtRiC = np.dot(CtRi, C)

            for t in reversed(xrange(T-1)):
                y = np.squeeze(data[t,:])
                r = np.dot(CtRi, y) + th_b
                tmp = np.linalg.inv(Qi + CtRiC + Lb)

                Lb = np.dot(AtQi, A) - np.dot(AtQi, np.dot(tmp, QiA))
                th_b = np.dot(AtQi, np.dot(tmp, r))

                Hs[t,:] = th_b
                Ls[t,:,:] = Lb

            backward_msgs.append((Hs, Ls))

        return backward_msgs

    def rts_smoother(self, didx=None, forward_msgs=None, backward_msgs=None):
        """ Follows algorithm on pg. 57 of "Bayesian Filtering and Smoothing.

            didx : int or iterable of ints, subset of observations sequences to
                  consider.
            forward_msgs : list of forward messages for the observation
                           sequences (as produced by `_forward_msgs` above).
                           If not provided `kalman_filter` is called.  These
                           messages are stored in information form.
            backward_msgs : list of backward messages for the observations
                            sequences (as produced by `_backward_msgs` above).
                            If not provided `_backward_msgs` is called.  These
                            messages are stored in information form.

            Returns:
            
            smoothing_dists : list of tuples, one for each observation
                              sequence, containing the means and covariances of
                              the smoothing distributions.
        """
        
        p, K = self._C.shape

        if didx is  None:
            datas = self._datas
        elif type(didx) is int:
            datas = self._data[didx]
        else:
            # This will barf if didx is not an iterable of ints
            datas = [self._data[idx] for idx in didx]

        if forward_msgs is None:
            alphas = self._forward_msgs(didx)
        if backward_msgs is None:
            betas = self._backward_msgs(didx)

        Ts = [d.shape[0] for d in datas]

        smoothing_dists = list()

        for i, (data, T) in enumerate(zip(datas, Ts)):
            Ms = np.empty((T, K))
            Ps = np.empty((T, K, K))
            
            Hf, Lf = alphas[i]
            Hb, Lb = betas[i]

            # Multiplying Gaussians in formation form is adding parameters
            Hs = Hf + Hb
            Ls = Lf + Lb

            for t in xrange(T):
                
                Ps[t,:,:] = np.linalg.inv(Ls[t,:,:])
                Ms[t,:] = np.dot(Ps[t,:,:], np.squeeze(Hs[t,:]))

            smoothing_dists.append((Ms, Ps))

        return smoothing_dists
