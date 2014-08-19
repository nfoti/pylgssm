from __future__ import division

import numpy as np
import numpy.testing as npt

import matplotlib.pyplot as plt

from nose.tools import nottest as nottest

from ..models.LGSSMFixedPython import LGSSMFixedKalmanFBPython as LGSSMFPy

class TestLGSSMFixedPython1d(object):

    def setUp(self):
        nobs = 2
        T = 100
        A = np.atleast_2d(1.)
        C = np.atleast_2d(1.)
        Q = np.atleast_2d(1.)
        R = np.atleast_2d(1.)
        mu_0 = np.atleast_1d(0.)
        S_0 = np.atleast_2d(1.)

        self.nobs = nobs
        self.T = T
        self.m = LGSSMFPy(A, C, Q, R, mu_0, S_0)
        m = self.m

        Xs = list()
        for i in xrange(nobs):
            X, Y = m.sample(T)
            Xs.append(X)
            m.add_data(Y)

        self.Xs = Xs

    def tearDown(self):
        pass

    # Comment out nottest to enable this test.
    @nottest
    def test_kalman_filter(self):
        m = self.m
        filter_dists = m.kalman_filter()

        # Plot results
        for i in xrange(self.nobs):
            t = np.arange(self.T)
            Y = m._datas[i]
            X = self.Xs[i]
            kal_m, kal_P = filter_dists[i]
            kal_m = np.squeeze(kal_m)
            q95 = 1.96*np.sqrt(np.squeeze(kal_P))
            plt.figure()
            plt.plot(t, Y, '.')
            plt.hold(True)
            plt.plot(t, X)
            plt.plot(t, kal_m)
            plt.plot(t, kal_m + q95, '--', color='#4682b4')
            plt.plot(t, kal_m - q95, '--', color='#4682b4')
            plt.legend(['Data', 'Latent state',
                        'Filtered estimate (+-2*std.  dev)'])
            plt.title("LGSSMFixedKalmanFBPython: Data set %d" % i)
            plt.show()

    # Comment out nottest to enable this test.
    @nottest
    def test_rts_smoother(self):
        m = self.m
        filter_dists = m.kalman_filter()
        smoothing_dists = m.rts_smoother()

        # Plot results
        for i in xrange(self.nobs):
            t = np.arange(self.T)
            Y = m._datas[i]
            X = self.Xs[i]
            Mf, Pf = filter_dists[i]
            Mf = np.squeeze(Mf)
            Ms, Ps = smoothing_dists[i]
            Ms = np.squeeze(Ms)
            q95 = 1.96*np.sqrt(np.squeeze(Ps))
            plt.figure()
            plt.plot(t, Y, '.')
            plt.hold(True)
            plt.plot(t, X)
            plt.plot(t, Mf)
            plt.plot(t, Ms)
            plt.plot(t, Ms + q95, '--', color='#4682b4')
            plt.plot(t, Ms - q95, '--', color='#4682b4')
            plt.legend(['Data', 'Latent state',
                        'Filtered estimate', 'Smoothed estimate (+- 2sd)'])
            plt.title("LGSSMFixedKalmanFBPython: Data set %d" % i)
            plt.show()
