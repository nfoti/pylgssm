from __future__ import division

from .LGSSMFixedPython import LGSSMFixedPython

import numpy as np


class LGSSMFixedKalmanFBPython(LGSSMFixedPython):
    """ Implements Kalman filter and RTS smoother using belief propagation.
        In particular the messages do not include the likelihoods (node
        potentials) which are incorported when computing the desired marginals.
    """

    def_forward_msgs(self, didx=None):
        """ Compute forward messages.
            
            didx: int or iterable of ints, subset of observations sequences to
                  consider.

            Returns:
                fwd_msgs : list of tuples, one for each observation sequence,
                           containing the information parameters of the forward
                           messages.

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


            for t in xrange(T):
                #TODO
                pass
            
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

        for i, (data, T) in enumerate(zip(datas, Ts)):
            Mf = np.empty((T, K))
            Pf = np.empty((T, K, K))

            th_f, Lf = fwd_msgs[i]

            for t in xrange(T):
                #TODO: This isn't done yet.  Need likelihood in here somewhere
                # too.
                Pf[t,:,:] = np.linalg.inv(Lf[t,:,:])
                Mf[t,:] = np.dot(Pf[t,:,:], th_f[t,:])

            filter_dists.append((Mf, Pf))

        return filter_dists

    def _backward_msgs(self, didx=None):
        """ Compute BP backward messages

            didx: int or iterable of ints, subset of observations sequences to
                  consider.

            Returns:
                list of tuples, one for each obs. sequence, with the
                information parameters of the backward messages.

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

            Hs[-1,:] = np.zeros(K)
            Ls[-1,:,:] = np.zeros((K,K))
            
            for t in reversed(xrange(T-1)):
                #TODO:
                pass

            bak_msgs.append((Hs, Ls))

        return bak_msgs

    def rts_smoother(self, didx=None, fwd_msgs=None, bak_msgs=None):
        """ Follows algorithm on pg. 57 of "Bayesian Filtering and Smoothing.

            didx : int or iterable of ints, subset of observations sequences to
                  consider.
            fwd_msgs : list of forward messages for the observation
                           sequences (as produced by `_forward_msgs` above).
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
        """
        
        p, K = self._C.shape

        if didx is  None:
            datas = self._datas
        elif type(didx) is int:
            datas = self._data[didx]
        else:
            # This will barf if didx is not an iterable of ints
            datas = [self._data[idx] for idx in didx]

        if fwd_msgs is None:
            fwd_msgs = self._forward_msgs(didx)
        if backward_msgs is None:
            bak_msgs = self._backward_msgs(didx)

        Ts = [d.shape[0] for d in datas]

        smoothing_dists = list()

        for i, (data, T) in enumerate(zip(datas, Ts)):
            Ms = np.empty((T, K))
            Ps = np.empty((T, K, K))
            

            #TODO:
            for t in xrange(T):
                pass
                
            smoothing_dists.append((Ms, Ps))

        return smoothing_dists
