

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

    def add_data(data):
        self._datas.append(data)
        self._state_dists.append(None)
        self._state_seqs.append(None)


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

    def kalman_filter():
        """ Follows algorithm of pg. 57 of "Bayesian Filtering and Smoothing.
        """
        p = self.p
        K = self.K

        datas = self._datas
        Ts = [d.shape[0] for d in datas]
        
        for data, T in enumerate(zip(datas, Ts)):
            kal_m = np.empty(T, K)
            kal_P = np.empty(T, K, K)

            kal_m[0,:] = self._mu_0
            kal_P[0,:,:] = self._S_0

            for t in xrange(T):
                # Prediction step (4.20)

                # Update step (4.21)

                # Use atleast_1d to make sure will work for scalar and vector
                # data/latent variables.
                pass

