from __future__ import division

import numpy as np
import numpy.testing as npt

import matplotlib.pyplot as plt

from nose.tools import nottest as nottest

from ..models.LGSSMFixedPython import LGSSMFixedSarkkaPython as LGSSMFPy
from ..util.plot import plot_gaussian_2D

class TestLGSSMFixedPython2d(object):

    def setUp(self):
        """ The parameters are from Example 4.3 in "Bayesian Filtering and
            Smoothing" dealing with car tracking.

            x = [x_1, x_2, x_3, x_4].T where (x_1,x_2) are position and
            (x_3,x_4) are velocity.
            y = [y_1, y_2].T are the measured position
        """
        T = 50
        nobs = 1
        K = 4
        p = 2
        dt = .1
        q1c = 1.
        q2c = 1.
        sig1 = 0.5
        sig2 = 0.5

        A = np.eye(K)
        A[0,2] = A[1,3] = dt
        Q = np.zeros((K,K))
        Q[0,0] = q1c * (dt**3) / 3.
        Q[0,2] = q1c * (dt**2) / 2.
        Q[1,1] = q2c * (dt**3) / 3.
        Q[1,3] = q2c * (dt**2) / 2.
        Q[2,0] = q1c * (dt**2) / 2.
        Q[2,2] = q1c * dt
        Q[3,1] = q2c * (dt**2) / 2.
        Q[3,3] = q2c * dt

        C = np.array([[1., 0., 0., 0.],
                      [0., 1., 0., 0.]])
        R = np.diag([sig1**2, sig2**2])

        # From Sarkko's website
        mu_0 = np.array([0., 0., 1., -1.]) 
        S_0 = np.eye(K)

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
            plt.figure()
            plt.plot(Y[:,0], Y[:,1], '.')
            plt.hold(True)
            plt.plot(X[:,0], X[:,1])
            plt.plot(kal_m[:,0], kal_m[:,1])
            for k in xrange(0, self.T, 5):
                mu = np.squeeze(kal_m[k,:2])
                Lmbda = np.squeeze(kal_P[k,:2,:2])
                plot_gaussian_2D(mu, Lmbda, color='#4682b4', alpha=0.7,
                        centermarker=False)
            #plt.legend(['Data', 'Latent state',
            #            'Filtered estimate'])
            plt.title("LGSSMFixedSarkkaPython: Data set %d" % i)
            plt.show()

    # Comment out nottest to enable this test.
    @nottest
    def test_rts_smoother(self):
        m = self.m
        filter_dists = m.kalman_filter()
        smoothing_dists = m.rts_smoother(filter_dists=filter_dists)

        # Plot results
        for i in xrange(self.nobs):
            t = np.arange(self.T)
            Y = m._datas[i]
            X = self.Xs[i]
            Mf, Pf = filter_dists[i]
            Ms, Ps = smoothing_dists[i]
            plt.figure()
            plt.plot(Y[:,0], Y[:,1], '.')
            plt.hold(True)
            plt.plot(X[:,0], X[:,1])
            plt.plot(Mf[:,0], Mf[:,1])
            plt.plot(Ms[:,0], Ms[:,1])
            plt.legend(['Data', 'Latent state',
                        'Filtered estimate', 'Smoothed estimate'])
            for k in xrange(0, self.T, 5):
                #mu = np.squeeze(Mf[k,:2])
                #Lmbda = np.squeeze(Mf[k,:2,:2])
                mu = np.squeeze(Ms[k,:2])
                Lmbda = np.squeeze(Ps[k,:2,:2])
                plot_gaussian_2D(mu, Lmbda, color='#4682b4', alpha=0.7,
                        centermarker=False)
            plt.title("LGSSMFixedSarkkaPython: Data set %d" % i)
            plt.show()
