
import numpy as np
import matplotlib.pyplot as plt


def plot_gaussian_2D(mu, lmbda, color='b', centermarker=True,label='',alpha=1.):
    ''' Plots mean and cov ellipsoid into current axes.
        
        mu : 2-vector, mean of Gaussian
        lmbda : 2x2 covariance matris    

        Grabbed from Matt Johnson's `pybasicbayes` project.
    '''
    assert len(mu) == 2
    assert lmbda.ndim == 2 and lmbda.shape[0] == 2 and lmbda.shape[1] == 2

    t = np.hstack([np.arange(0,2*np.pi,0.01),0])
    circle = np.vstack([np.sin(t),np.cos(t)])
    ellipse = np.dot(np.linalg.cholesky(lmbda),circle)

    if centermarker:
        plt.plot([mu[0]],[mu[1]],marker='D',color=color,markersize=4,alpha=alpha)

    plt.plot(ellipse[0,:] + mu[0], ellipse[1,:] + mu[1],linestyle='-',
            linewidth=2,color=color,label=label,alpha=alpha)
