from __future__ import division

import numpy as np


class LGSSMBase(object):
    """ Linear Gaussian state space model base class.

        x_t = A*x_{t-1} + v_t, v_t \sim N(0, Q)
        y_t = C*x_t     + w_t, w_t \sim N(0, R)

        x_0 \sim N(\mu_0, S_0)

        x_t \in \mathbb{R}^K
        y_t \in \mathbb{R}^p
    """

    def __init__(self):
        self._datas = list()
        self._state_seqs = list()   # These are for sampling realizations

    def add_data(self, data):
        self._datas.append(data)
        self._state_seqs.append(None)

    @staticmethod
    def sample(self, T): 
        pass
