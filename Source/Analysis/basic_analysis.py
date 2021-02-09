#
import numpy as np
import scipy.io
import h5py
from sklearn.neighbors.kde import KernelDensity
from zecording import Zecording, zanalysis
import zecording as zec

@zanalysis()
def mean_rate(z: Zecording):
    """
    Computes the mean rate of the neurons over time
    """
    mean_rate = np.mean(z['df'],1) # average over time, mean rate per neuron

    return mean_rate


def weighted_density(coords,gridcoords,weights,bandwidth=0.004,atol=0.01):
    """
    Compute a weighted density estimate
    :param coords: NP-array (N X 3) of coordinates
    :param gridcoords: 3D matrix of coordinates on which the kde will be evaluated
    :param weights: NP-array (N x 1) of values for each point indicated by the coordinates
    :param bandwidth: bandwidth of the Gaussian kernel to be used
    :return: density estimate
    """
    kde = KernelDensity(kernel='gaussian', bandwidth=bandwidth,atol=atol).fit(coords,sample_weight=weights)
    density = kde.score_samples(gridcoords)  # return log scores
    return density


def prepare_gridcoords(x_bounds=[0,0.5],y_bounds = [0,1],z_bounds = [0,0.3],res = 0.01):
    """
    Compute the overall set of grid coordinates
    :param x_bounds: 2-elem list with lower and upper bounds in x
    :param y_bounds: 2-elem list with lower and upper bounds in y
    :param z_bounds: 2-elem list with lower and upper bounds in z
    :param res: spatial resolution of the grid
    :return: gridcoords
    :return: x_steps
    :return: y_steps
    :return: z_steps
    """
    x_steps = np.arange(x_bounds[0],x_bounds[1],res)
    y_steps = np.arange(y_bounds[0],y_bounds[1],res)
    z_steps = np.arange(z_bounds[0],z_bounds[1],res)
    gridcoords = np.zeros([len(x_steps) * len(y_steps) * len(z_steps), 3])
    gridcoords_inds = np.zeros([len(x_steps) * len(y_steps) * len(z_steps), 3],dtype=np.int)

    k=0
    for iX,cX in enumerate(x_steps):
        for iY,cY in enumerate(y_steps):
            for iZ,cZ in enumerate(z_steps):
                gridcoords[k,:] = [cX,cY,cZ]
                gridcoords_inds[k,:] = [iX,iY,iZ]
                k=k+1
    return gridcoords, gridcoords_inds, x_steps, y_steps, z_steps

def reshape_density(density,gridcoords,x_steps,y_steps,z_steps):
    """
    Reshape the density to be as a 3d matrix for convenient indexing
    :param density:
    :param x_steps:
    :param y_steps:
    :param z_steps:
    :return: density_3d:
    """

    density_3d = np.zeros((len(x_steps),len(y_steps),len(z_steps)))
    for ik in range(gridcoords.shape[0]):
        density_3d[gridcoords[ik,0],gridcoords[ik,1],gridcoords[ik,2]] = density[ik]

    return density_3d

def save_matlab(file,**kwargs):
    scipy.io.savemat(file,kwargs)