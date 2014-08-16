

class LGSSMBase(object):
    """ Linear Gaussian state space model base class.

        x_t = A*x_{t-1} + v_t, v_t \sim N(0, Q)
        y_t = C*x_t     + w_t, w_t \sim N(0, R)

        x_0 \sim N(\mu_0, S_0)

        x_t \in \mathbb{R}^K
        y_t \in \mathbb{R}^p
    """

    def __init__():
        self._datas = list()
        self._state_dists = list()
        self._state_seqs = list()

    def add_data(self, data):
        self._datas.append(data)
        self._state_dists.append(None)
        self._state_seqs.append(None)

    @staticmethod
    def sample(self, T): 
        pass


class LGSSMFixedPython(LGSSMBase):
    """ Python implementation with fixed parameters.

        Model parameters A, C, Q, R, mu_0, and S_0 are assumed fixed.
    """

    def __init__(A, C, Q=None, R=None, mu_0=None, S_0=None):
        """
            A    : K x K latent transition matrix
            C    : p x K latent to observation matrix
            Q    : K x K latent covariance matrix
            R    : p x p observation noise covariance
            mu_0 : K vector, mean of starting latent state (no associated
                   observation)
            S_0  : K x K covariance matrix for initiaion latent state
        """
        # Put some error checking in here
        self._A = A
        self._C = C
        p, K = C.shape
        self._p = p
        self._K = shape
        self._Q = Q if Q is not None else np.eye(K)
        self._R = R if R is not None else np.eye(p)
        self._mu_0 = mu_0 if mu_0 is not None else np.zeros(K)
        self._S_0 = S_0 if S_0 is not None else np.eye(K)

    def sample(self, T):
        A = self._A
        C = self._C
        Q = self._Q
        R = self._R
        p, K = C.shape

        V = np.random.multivariate_normal(np.zeros(K), Q, size=T)
        W = np.random.multivariate_normal(np.zeros(p), R, size=T)

        X = np.empty(T, K)
        Y = np.empty(T, p)

        # Initial state with no observation
        X0 = mu_0 + np.random.multivariate_normal(np.zeros(K), S_0)
        
        X[0,:] = np.dot(A, X0) + V[0,:]
        Y[0,:] = np.dot(C, X[0,:]) + W[0,:]
        for t in xrange(1,T):
            X[t,:] = np.dot(A, X[t-1,:]) + V[t,:]
            Y[t,:] = np.dot(C, X[t,:]) + W[t,:]

        return X, Y

    def kalman_filter():
        """ Follows algorithm of pg. 57 of "Bayesian Filtering and Smoothing.
        """
        p = self.p
        K = self.K
        C = self._C
        A = self._A
        Q = self._Q
        R = self._R

        datas = self._datas
        Ts = [d.shape[0] for d in datas]

        kal_ms = list()
        kal_Ps = list()
        
        for data, T in enumerate(zip(datas, Ts)):
            kal_m = np.empty(T, K)
            kal_P = np.empty(T, K, K)

            kal_m[0,:] = self._mu_0
            kal_P[0,:,:] = self._S_0

            for t in xrange(1,T):
                # Prediction step
                mbar = np.dot(A, kal_m[t-1,:])
                Pbar = np.dot(A, np.dot(kal_P[t-1], A.T)) + Q

                # Update step (4.21)
                y_t = np.atleast_1d(data[t,:])
                v_t = y_t - np.dot(C, mbar)
                S_t = np.dot(C, np.dot(Pbar, C.T)) + R

                K_t = np.dot(Pbar, np.dot(C.T, np.linalg.inv(S_t)))

                kal_m[t,:] = mbar + np.dot(K_t, v_t)
                kal_P[t,:,:] = Pbar + np.dot(K_t, np.dot(S_t, K_t.T))

            kal_ms.append(kal_m)
            kal_Ps.append(kal_P)

    return kal_ms, kal_Ps
