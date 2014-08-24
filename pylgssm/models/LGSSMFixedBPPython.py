from __future__ import division

from .LGSSMFixedPython import LGSSMFixedPython

import numpy as np


class LGSSMFixedBPPython(LGSSMFixedPython):
    """ Implements Kalman filter and RTS smoother using belief propagation.
        In particular the messages do not include the likelihoods (node
        potentials) which are incorported when computing the desired marginals.
    """

    def _forward_msgs(self, didx=None):
        """ Compute forward messages.
            
            didx: int or iterable of ints, subset of observations sequences to
                  consider.

            Returns:
                fwd_msgs : list of tuples, one for each observation sequence,
                           containing the information parameters of the forward
                           messages.  Each tuple is of the form (Hs, Ls) where
                           Hs is T x p and Ls is T x p x p and where row t
                           contains the message parameter for the message from
                           time t -> t+1.  This means that the last row
                           corresponds to message T-1 -> T.

                           See rts_smoother for how to use these messages to
                           compute the marginals.

            The messages are stored in information form so that we can combine
            them at later states easily (and for numerical stability).
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

        fwd_msgs = list()

        Qi = np.linalg.inv(Q)
        Ri = np.linalg.inv(R)
        Ai = np.linalg.inv(A)

        for i, (data, T) in enumerate(zip(datas, Ts)):
            Hs = np.empty((T, K))
            Ls = np.empty((T, K, K))

            QiA = np.linalg.solve(Q, A)
            AtQiA = np.dot(A.T, QiA)
            CtRi = np.linalg.solve(R.T, C).T # np.dot(C.T, Ri)
            CtRiC = np.dot(CtRi, C)

            Jt = np.linalg.inv(S_0) + AtQiA
            ht = np.linalg.solve(S_0, mu_0)

            Jti = np.linalg.inv(Jt)
            Ls[0,:,:] = -np.dot(QiA, np.dot(Jti, QiA.T))
            Hs[0,:] = -np.dot(QiA, np.dot(Jti, ht))

            for t in xrange(1,T-1):
                # t indexes the messages array, need to transform to data array
                y = np.squeeze(data[t-1,:])
                Jt = Qi + CtRiC + AtQiA
                ht = np.dot(CtRi, y)
                
                Jmi = np.linalg.inv(Jt + Ls[t-1,:,:])
                hm = np.squeeze(Hs[t-1,:])
                Ls[t,:,:] = -np.dot(QiA, np.dot(Jmi, QiA.T))
                Hs[t,:] = -np.dot(QiA, np.dot(Jmi, ht + hm))

            y = np.squeeze(data[-1,:])
            Jt = Qi + CtRiC
            ht = np.dot(CtRi, y)
            Jmi = np.linalg.inv(Jt + Ls[-2,:,:])
            hm = np.squeeze(Hs[-2,:])
            Ls[-1,:,:] = -np.dot(QiA, np.dot(Jmi, QiA.T))
            Hs[-1,:] = -np.dot(QiA, np.dot(Jmi, ht + hm))

            fwd_msgs.append((Hs, Ls))

        return fwd_msgs

    def kalman_filter(self, didx=None, fwd_msgs=None):
        """ Combine forward messages with likelihoods (node potentials) to
            compute the filtering distribution at each time.

            didx : int or iterable of ints, subset of observations sequences to
                  consider.
            fwd_msgs : list of tuples, one for each obs. sequence,
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

        if fwd_msgs is None:
            fwd_msgs = self._forward_msgs(didx)

        if didx is  None:
            datas = self._datas
        elif type(didx) is int:
            datas = self._data[didx]
        else:
            # This will barf if didx is not an iterable of ints
            datas = [self._data[idx] for idx in didx]

        Ts = [d.shape[0] for d in datas]

        filter_dists = list()

        Qi = np.linalg.inv(Q)
        Ri = np.linalg.inv(R)
        Ai = np.linalg.inv(A)

        for i, (data, T) in enumerate(zip(datas, Ts)):
            Mf = np.empty((T, K))
            Pf = np.empty((T, K, K))

            QiA = np.linalg.solve(Q, A)
            AtQiA = np.dot(A.T, QiA)
            CtRi = np.linalg.solve(R.T, C).T # np.dot(C.T, Ri)
            CtRiC = np.dot(CtRi, C)

            Hf, Lf = fwd_msgs[i]

            for t in xrange(T):
                y = np.squeeze(data[t,:])
                J = Qi + CtRiC + AtQiA + Lf[t,:,:]
                h = np.dot(CtRi, y) + np.squeeze(Hf[t,:])

                Pf[t,:,:] = np.linalg.inv(J)
                Mf[t,:] = np.dot(Pf[t,:,:], h)

            filter_dists.append((Mf, Pf))

        return filter_dists

    def _backward_msgs(self, didx=None):
        """ Compute BP backward messages

            didx: int or iterable of ints, subset of observations sequences to
                  consider.

            Returns:
                bak_msgs : list of tuples, one for each obs. sequence, with the
                           information parameters of the backward messages.
                           The tuples are of the form (Hs, Ls) where the rows
                           of Hs contain the potential vectors and the
                           information matrices of the messages respectively.
                           Note that row t contains the message from t+1 -> t
                           so that row T-1 contains the message T -> T-1 and
                           row 0 contains the message 1 -> 0.

                           See rts_smoother for how to use these messages to
                           compute the marginals.

            The messages are stored in information for so that they can be
            combined trivially (and for numerical reasons).
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

        bak_msgs = list()

        Qi = np.linalg.inv(Q)
        Ri = np.linalg.inv(R)
        Ai = np.linalg.inv(A)

        for i, (data, T) in enumerate(zip(datas, Ts)):
            
            Hs = np.empty((T, K))
            Ls = np.empty((T, K, K))

            QiA = np.linalg.solve(Q, A)
            AtQiA = np.dot(A.T, QiA)
            CtRi = np.linalg.solve(R.T, C).T # np.dot(C.T, Ri)
            CtRiC = np.dot(CtRi, C)

            y = np.squeeze(data[-1,:])
            Jt = Qi + CtRiC
            Jm = np.linalg.inv(Jt)
            ht = np.dot(CtRi, y)
            hm = ht

            for t in reversed(xrange(0,T)):

                Ls[t,:,:] = -np.dot(QiA.T, np.dot(Jm, QiA))
                Hs[t,:] = -np.dot(QiA.T, np.dot(Jm, hm))

                # Just processed t, this sets up stuff for next iter so need
                # data[t-1].  Similarly, the current t is the "previous"
                # message in the next iteration.
                y = np.squeeze(data[t-1,:])
                Jt = Qi + CtRiC + AtQiA
                Jm = np.linalg.inv(Jt + Ls[t,:,:])
                ht = np.dot(CtRi, y)
                hm = ht + np.squeeze(Hs[t,:])

            bak_msgs.append((Hs, Ls))

        return bak_msgs

    def rts_smoother(self, didx=None, fwd_msgs=None, bak_msgs=None):
        """ Follows algorithm on pg. 57 of "Bayesian Filtering and Smoothing.

            didx : int or iterable of ints, subset of observations sequences to
                  consider.
            fwd_msgs : list of forward messages for the observation
                           sequences (as produced by `_+1forward_msgs` above).
                           If not provided `kalman_filter` is called.  These
                           messages are stored in information form.
            bak_msgs : list of backward messages for the observations
                            sequences (as produced by `_backward_msgs` above).
                            If not provided `_backward_msgs` is called.  These
                            messages are stored in information form.

            Returns:
            
            smoothing_dists : list of tuples, one for each observation
                              sequence, containing the means and covariances of
                              the smoothing distributions.

                              The tuples look like (mu, S) with the mean and
                              covariance of the smoothing distributions.  Both
                              mu and S have T+1 rows, where row 0 corresponds
                              to the distribution to the auxiliary state x_0.
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

        if fwd_msgs is None:
            fwd_msgs = self._forward_msgs(didx)
        if bak_msgs is None:
            bak_msgs = self._backward_msgs(didx)

        Ts = [d.shape[0] for d in datas]

        smoothing_dists = list()

        Qi = np.linalg.inv(Q)
        Ri = np.linalg.inv(R)
        Ai = np.linalg.inv(A)

        for i, (data, T) in enumerate(zip(datas, Ts)):
            Ms = np.empty((T+1, K))
            Ps = np.empty((T+1, K, K))
            
            Hf, Lf = fwd_msgs[i]
            Hb, Lb = bak_msgs[i]

            QiA = np.linalg.solve(Q, A)
            AtQiA = np.dot(A.T, QiA)
            CtRi = np.linalg.solve(R.T, C).T # np.dot(C.T, Ri)
            CtRiC = np.dot(CtRi, C)

            # x_0 special case
            S0i = np.linalg.inv(S_0)
            J = S0i + AtQiA + Lb[0,:,:]
            h = np.dot(S0i, mu_0) + np.squeeze(Hb[0,:])

            Ps[0,:,:] = np.linalg.inv(J)
            Ms[0,:] = np.dot(Ps[0,:,:], h)

            # x_T only has the forwards message, JT, and the evidence
            y = np.squeeze(data[-1,:])
            J = Qi + CtRiC + Lf[-1,:,:]
            h = np.dot(CtRi, y)# + np.squeeze(Hf[-1,:])

            # x_T special case
            Ps[-1,:,:] = np.linalg.inv(J)
            Ms[-1,:] = np.dot(Ps[-1,:,:], h)
            
            # Everything in between is normal
            for t in xrange(1,T):
                y = np.squeeze(data[t-1,:])
                J = Qi + CtRiC + AtQiA + Lf[t-1,:,:]# + Lb[t,:,:]
                h = np.dot(CtRi, y) + np.squeeze(Hf[t-1,:])# + np.squeeze(Hf[t,:])

                Ps[t,:,:] = np.linalg.inv(J)
                Ms[t,:] = np.dot(Ps[t,:,:], h)
                
            smoothing_dists.append((Ms, Ps))

        return smoothing_dists
