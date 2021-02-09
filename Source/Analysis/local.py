import numpy as np
import h5py
from scipy.spatial import cKDTree
from datetime import datetime
from sklearn.linear_model import Lasso, BayesianRidge
from os.path import expanduser
import pandas as pd
from typing import Tuple
import pywt


def open_data(path: str) -> (np.ndarray, np.ndarray, np.ndarray):
    """
    Open a HDF5 data file and get the calcium signal, neurons coordinates as well as the time base

    Parameters
    ----------
    path: str

    Returns
    -------
    calcium_signal: Numpy array
        Calcium signal in an array of shape nb_neurons x number_time_points
    coordinates: Numpy array
        nb_neurons x 3
    time_points: Numpy array
        1 x nb_time_points
    """
    with h5py.File(path, 'r') as F:
        calcium_signal = np.transpose(F['Data']['Values'].value, (1, 0))  # NNeurons x NTimeSteps
        coordinates = np.transpose(F['Data']['Coordinates'].value, (1, 0))  # NNeurons x 3
        time_points = np.transpose(F['Data']['Times'].value, (1, 0))  # NTimeSteps

    return calcium_signal, coordinates, time_points


def create_kdtree(coordinates: np.ndarray):
    return cKDTree(coordinates)


def get_neighbors(kd_tree: cKDTree, point_ix: int, radius: float, known=None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get a point's neighbors within a specific radius,
    excluding the ones we already know about from a previous query with a small radius

    Parameters
    ----------
    kd_tree
    point_ix
    radius
    known: set or None

    Returns
    -------
    neigh_ix: list of ints
        Index of new neighboring points
    points: Numpy array
        Coordinates of the new neighboring points
    """
    if known is None:
        known = {point_ix}
    neigh = kd_tree.query_ball_point(kd_tree.data[point_ix, :], radius)
    neigh = np.array(list(set(neigh) - set(known)))

    try:  # normal behavior
        return neigh, kd_tree.data[neigh, :]
    except IndexError:
        # except for the last run, one neuron left so neigh becomes a empty list and tree.data indexing is not possible
        return np.array([]), np.array([]) # return empty vector


def compute_correlation(calcium_signal: np.ndarray, ix_pt1: int, ix_pt2: int):
    """
    Pearson coefficient of correlation between the calcium signals of two neurons
    TODO: Better measure?

    Parameters
    ----------
    calcium_signal
    ix_pt1
    ix_pt2

    Returns
    -------
    r: float
        Coefficient of correlation
    """
    df1 = calcium_signal[ix_pt1, :]
    df2 = calcium_signal[ix_pt2, :]
    r = np.corrcoef(df1, df2)[0, 1]

    return r


def correlation(calcium_signal: np.ndarray, ix_pt1: int, ix_pt2: np.ndarray):
    """
    Pearson coefficient of correlation between the calcium signals of two neurons
    Calculated manually to be faster in a 1 vector x 1 matrix
    TODO: Better measure?

    Parameters
    ----------
    calcium_signal
    ix_pt1
    ix_pt2

    Returns
    -------
    r: float
        Coefficient of correlation
    """
    ix_pt2 = np.asarray(ix_pt2)
    df1 = calcium_signal[ix_pt1, :]
    df2 = calcium_signal[ix_pt2, :].transpose()
    cov = np.dot(df1 - df1.mean(), df2 - df2.mean(axis=0)) / (df2.shape[0] - 1)
    # ddof=1 necessary because covariance estimate is unbiased (divided by n-1)
    p_var = np.sqrt(np.var(df1, ddof=1) * np.var(df2, axis=0, ddof=1))
    r = cov / p_var
    return r


def lasso_neigh(calcium_signal: np.ndarray, ix_pt1: int, ix_pt2: np.ndarray):
    x = calcium_signal[ix_pt2, :]
    y = calcium_signal[ix_pt1, :]
    reg = Lasso(alpha=.05).fit(x.transpose(), y)
    coef = np.abs(reg.coef_)
    return coef


def compute_lasso(calcium_signal: np.ndarray, ix_pt1: int, ix_pt2: np.ndarray, alpha: float):
    """

    Parameters
    ----------
    calcium_signal
    ix_pt1
    ix_pt2
    alpha

    Returns
    -------
    coef: Numpy ndarray
        Lasso non-zero coefficients
    gi: Numpy ndarray
        Indices of contributing neurons
    """
    x = calcium_signal[ix_pt2, :].transpose()
    y = calcium_signal[ix_pt1, :]
    n_points = y.shape[0]
    half = n_points // 2
    train_x = x[:half, :]
    train_y = y[:half]
    reg = Lasso(alpha=alpha).fit(train_x, train_y)
    if reg.score(x[half:, :], y[half:]) < .2:
        return [], []
    coef = reg.coef_
    gi, = np.nonzero(coef) # comma?

    return coef[gi], gi


def bayesridge_neigh(calcium_signal: np.ndarray, ix_pt1: int, ix_pt2: np.ndarray):
    n_time_points = calcium_signal.shape[1]
    half = n_time_points // 2
    x = calcium_signal[ix_pt2, :].transpose()
    y = calcium_signal[ix_pt1, :]
    train_x = x[:half, :]
    train_y = y[:half]
    reg = BayesianRidge().fit(train_x, train_y)
    # predict = reg.predict(x)
    score = reg.score(x[half:, :], y[half:])
    if score < .5:
        return np.zeros(reg.coef_.shape)
    coef = reg.coef_
    gi = np.logical_or(coef < np.percentile(coef, 2.5), coef > np.percentile(coef, 100-2.5))
    s_coef = np.abs(coef)
    s_coef[np.logical_not(gi)] = -1
    return s_coef


def correlate_neighbourhood(calcium_signal: np.ndarray, kd_tree: cKDTree, center_ix: int,
                            init_radius=0.02, max_radius=.08, min_corr=.5, step=0.01, measure=correlation,
                            verbose=True):
    """
    Given a center neuron and parameters of the neighbourhood definition, tries to group neurons
    The basic idea is:
    1. Look at all neurons within a given radius of the center neurons,
    2. Correlate their calcium signal to the center's.
    3. Keep sufficiently highly correlated neurons as being part of the group.
    4. Compute the fraction correlated / all neighboring neurons
    5. Move the center to the neuron closest to the center of mass of this group
    6. Increase slightly the radius and start again.
    7. As long as the fraction of correlated neurons is not droppping significantly, keep on increasing the radius
    8. Label the neurons as being part of this group. If some were already part of another group,
       they belong to the biggest group

    Parameters
    ----------
    calcium_signal
    kd_tree
    center_ix
    init_radius
    max_radius
    min_corr
    step
    measure
    verbose

    Returns
    -------

    """
    FRAC_DEC = .95
    radii = np.arange(init_radius, max_radius, step)
    radius = radii[0] # not necessary due to loop?
    frac_corr = 0
    w_correlated = np.array([])
    for radius in radii:
        neighbors_ix, _ = get_neighbors(kd_tree, center_ix, radius)
        if len(neighbors_ix) == 0: # one neuron left so no neighbours
            break
        corr_neigh = measure(calcium_signal, center_ix, neighbors_ix)
        # Fraction of correlated neurons in the neighboorhod
        correlated = corr_neigh >= min_corr
        n_correlated = np.sum(correlated)
        new_frac_corr = n_correlated / len(corr_neigh)
        if verbose:
            print(f'Number of neurons: {len(corr_neigh)} ; fraction correlated: {new_frac_corr * 100:.2f}% ;'
                  f' Correlated neurons: {np.sum(correlated)}')
        # More correlations than before
        if new_frac_corr >= FRAC_DEC * frac_corr and n_correlated > 2: # 100
            frac_corr = new_frac_corr
            w_correlated = neighbors_ix[correlated]
            centroid = np.mean(kd_tree.data[w_correlated, :], 0)
            _, center_ix = kd_tree.query(centroid, 1)
        else:
            break
    if radius == radii[-1]:
        # print('\t >>> Reached maximum radius <<<')
        pass
    return w_correlated


def create_neighbourhoods(calcium_signal: np.ndarray, kd_tree: cKDTree, **kwargs):
    """
    Go through all neurons and try to create a neighbourhood around it. See :pyfunc:`correlate_neighbourhood`
    for more details. Neurons which are already part of a group are not considered as potential centers for new groups

    Parameters
    ----------
    calcium_signal: Numpy array
        Raw calcium signal
    kd_tree: cKDTree
        KD-tree in which neurons are sorted
    kwargs: dict
        Other arguments to be passed to :pyfunc:`correlate_neighbourhood`

    Returns
    -------

    """
    neurons = set(np.arange(calcium_signal.shape[0]))
    neighbourhoods = {n: (-1, 0) for n in neurons}  # neigh_id, number of neighbours
    neigh_id = 0
    while len(neurons) > 0:
        center_ix = neurons.pop()
        neigh = correlate_neighbourhood(calcium_signal, kd_tree, center_ix, **kwargs)
        n_neigh = len(neigh)
        if n_neigh < 2:  # 100
            continue
        neurons = neurons - set(neigh)
        for n in neigh:
            if n_neigh > neighbourhoods[n][1]:
                neighbourhoods[n] = (neigh_id, n_neigh)
        neigh_id += 1
        print(f'Remaining neurons: {len(neurons)} ; Cluster id: {neigh_id} ; Cluster size: {n_neigh} ; '
              f'Center: {center_ix}')
        # if neigh_id == 20:
        #     return neighbourhoods
    return neighbourhoods


def neigh_reg(calcium_signal: np.ndarray, kd_tree: cKDTree, center_ix: int,
              radius=0.05, alpha=0.05) -> Tuple[np.ndarray, np.ndarray]:
    """
    Select neighboring neurons if their regression coefficient is non zero

    Parameters
    ----------
    calcium_signal: Numpy array
        Raw calcium signal
    kd_tree: cKDTree
        KD-tree in which neurons are sorted
    center_ix: int
        Index of reference neuron
    radius: float
        Default to 0.05
    alpha: float
        Regressor hyperparameter
        Default to 0.05

    Returns
    -------
    coeff: Numpy array
        Non zero regression coefficient
        Shape 1xn
    """
    neighbors_ix, _ = get_neighbors(kd_tree, center_ix, radius)
    if neighbors_ix.shape[0] == 0:
        return np.array([]), np.array([])
    coeff, correlated_ix = compute_lasso(calcium_signal, center_ix, neighbors_ix, alpha)
    return coeff, correlated_ix


def create_cluster(calcium_signal: np.ndarray, kd_tree: cKDTree, center_ix: int, radius=0.05, alpha=0.05) -> set:
    c, g_ix = neigh_reg(calcium_signal, kd_tree, center_ix, radius, alpha)
    cluster = set(g_ix)
    pot_centers = set(g_ix)
    cluster.add(center_ix)
    while len(pot_centers) > 0:
        n_center = pot_centers.pop()
        c, g_ix = neigh_reg(calcium_signal, kd_tree, n_center, radius, alpha)
        if center_ix not in g_ix:
            continue
        new_cluster = set(g_ix)
        new_cluster.add(n_center)
        if len(new_cluster) < len(cluster):
            continue
        pot_centers.update(new_cluster)
        cluster = new_cluster
    return cluster


def assign_cluster(calcium_signal: np.ndarray, kd_tree: cKDTree, radius=0.05, alpha=0.05) -> dict:
    """

    Parameters
    ----------
    calcium_signal: Numpy array
        Raw calcium signal
    kd_tree: cKDTree
        KD-tree in which neurons are sorted
    radius: float
        Default to 0.05
    alpha: float
        Regressor hyperparameter
        Default to 0.05

    Returns
    -------
    d_clusters: dict
        cl_id: {neuron_0, neuron_1, ...}
    """
    neurons = set(np.arange(calcium_signal.shape[0]))
    neuron_cl = {n: -1 for n in neurons}  # neuron_id: cluster_id
    d_clusters = {}  # cl_id: {neuron_0, neuron_1, ...}
    cl_id = 0
    while len(neurons) > 0:
        center_ix = neurons.pop()
        cluster = create_cluster(calcium_signal, kd_tree, center_ix, radius, alpha)
        c_size = len(cluster)
        if c_size < 2:
            continue

        prev_assigned = set([n for n in list(cluster.difference(neurons)) if neuron_cl[n] != -1])
        for n in list(cluster):
            if n in prev_assigned:
                prev_cluster = neuron_cl[n]
                prev_size = len(d_clusters[prev_cluster])
                if prev_size > c_size:
                    cluster.remove(n)
                    continue
            neuron_cl[n] = cl_id

        if len(cluster) < 2:
            continue

        neurons.difference_update(cluster)
        print(f'Cluster {cl_id} of size {len(cluster)}. {len(neurons)} left')
        d_clusters[cl_id] = cluster
        cl_id += 1

    return d_clusters


def hierarchical_clustering(calcium_signal: np.ndarray, kd_tree: cKDTree, radius=0.05, alpha=0.05,
                            filepath='hierarchy.h5', level=0):
    """

    Parameters
    ----------
    calcium_signal: Numpy array
        Raw calcium signal
    kd_tree: cKDTree
        KD-tree in which neurons are sorted
    radius: float
        Default to 0.05
    alpha: float
        Regressor hyperparameter
        Default to 0.05
    filepath: str
        File to save results to
    level: int

    Returns
    -------

    """
    print(f'>>> LEVEL {level} <<<')
    print(f'time: {datetime.now()}')
    cl_neuron = np.zeros(calcium_signal.shape[0]) - 1
#    new_ix = np.arange(calcium_signal.shape[0])
    new_ix = np.zeros(calcium_signal.shape[0]) - 1
    clusters = assign_cluster(calcium_signal, kd_tree, radius, alpha) # get dictionary of [cluster_index] = set(all neurons in cluster)
    print(f'Number of clusters assigned: {len(clusters)}')
    if len(clusters) == 0:
        print('No clusters were assigned')
        return
    for cl_id, n_ix in clusters.items():
        ix = np.array(list(n_ix)) # np array of neurons in clusters
        cl_neuron[ix] = cl_id # assign cluster to neurons
#    n_clusters = np.sum(cl_neuron == -1) + len(clusters) # number of clusters + number of unassigned neurons
    n_clusters = len(clusters) # number of clusters
    new_calcium = np.zeros((n_clusters, calcium_signal.shape[1])) # use clusters as new 'neurons'
    new_coords = np.zeros((n_clusters, 3))
    n_empty_clusters = 0
    for ix_cl, cl_id in enumerate(list(clusters.keys())): # cluster ids (aren't the indices the same because first cluster index =0?)
        gi = cl_neuron == cl_id # true for all neurons in cluster cl_id
        if sum(gi) == 0:
            print(f'no neurons in cluster {cl_id}')
            n_empty_clusters = n_empty_clusters + 1
            continue
        ix_cl_corr = ix_cl - n_empty_clusters
        new_calcium[ix_cl_corr, :] = calcium_signal[gi, :].mean(0) # returns empty array for empty cluster
        new_coords[ix_cl_corr, :] = kd_tree.data[gi, :].mean(0) # what happens for empty cluster?
        new_ix[gi] = ix_cl_corr #=cl_id?
#    iso_ix, = np.where(cl_neuron == -1) # non clustered neurons
#    for ix,  iso_ix_val in enumerate(iso_ix):
#        new_calcium[ix+ix_cl_corr, :] = calcium_signal[iso_ix_val, :] # fill rest of new_calcium and new_coords
#        new_coords[ix+ix_cl_corr, :] = kd_tree.data[iso_ix_val, :]
    if len(clusters) > 1:
        new_tree = create_kdtree(new_coords)
        with h5py.File(filepath, 'a') as d_file:
            print(filepath)
            l_group = d_file.create_group(f'level_{level}')
            l_group.create_dataset('df', data=new_calcium)
            l_group.create_dataset('coords', data=new_coords)
            l_group.create_dataset('indices', data=new_ix)
            l_group.create_dataset('clusters', data=cl_neuron)
        if new_calcium.shape[0] < calcium_signal.shape[0]:
            if level < 5:
                hierarchical_clustering(new_calcium, new_tree, radius*3, alpha*.8, filepath, level+1) # should alpha not decline?
            else: # radius is already at max pairwise distance?
                hierarchical_clustering(new_calcium, new_tree, radius*1, alpha*1, filepath, level+1) # should alpha not decline?
        # l_clusters.append(hierarchical_lasso(clustered_calcium, new_tree, radius, alpha, l_clusters))
        # return l_clusters
    else:
        print('Only 1 cluster left, quitting')
        return


def partial_reconstruct(signal, nb_levels=3, rm_levels=None, wavelet='db1'):
    coeffs = pywt.wavedec(signal, wavelet, level=nb_levels)
    if rm_levels is not None:
        for i_lvl in rm_levels:
            coeffs[i_lvl] = np.zeros_like(coeffs[i_lvl])
    reconstruct = pywt.waverec(coeffs, wavelet)

    return reconstruct


def wvl_filter(df, axis=1, rm_levels=None, **kwargs):
    f_df = np.apply_along_axis(partial_reconstruct, axis, df, rm_levels=rm_levels, **kwargs)

    return f_df


if __name__ == '__main__':
    # PATH = expanduser('~/Programming/fishualizer/Data/SampleDataFull_20140827.h5')
    PATH = expanduser('/home/rproville/Programming/fishualizer/Data/Data20140827_spont/SampleData.h5')
    # PATH = expanduser('~/Desktop/thijs-repos/fishualizer/Data/Data20140827_spont/SampleData.h5')
    df, coords, times = open_data(PATH)
    tree = create_kdtree(coords)
    # hierarchical_clustering(df, tree, filepath=f"hierarchy_{datetime.now().strftime('%Y-%m-%d_%H_%M')}.h5")

    # ex_neigh = correlate_neighbourhood(df, tree, 51208, min_corr=.5)
    f_df = wvl_filter(df, rm_levels=(0, ), wavelet='db6')
    nh = create_neighbourhoods(f_df, tree, init_radius=0.05, verbose=False, min_corr=.6)
    # nh = create_neighbourhoods(df, tree, init_radius=0.05, verbose=False, min_corr=1e-6, measure=bayesridge_neigh,
    #                            max_radius=0.051, step=0.049)
    # Makes it a vector, with each neuron having a id number attached
    nh_arr = np.zeros(df.shape[0])
    for k, v in nh.items():
        nh_arr[k] = v[0]

    nh_arr.dump(f"localities_{datetime.now().strftime('%Y-%m-%d_%H_%M')}.npy")
    cl = pd.DataFrame({'cluster': nh_arr, 'neuron': np.arange(len(nh))})
    cl.set_index('neuron')
    cl.to_hdf(f"clusters_{datetime.now().strftime('%Y-%m-%d_%H_%M')}.h5", key='df_neur')

    # This localities file can be loaded in the fishualizer but is hard to visualize
    # because it is a categorical variable. One would need a better way of looking at it to see if it interesting
    # Important parameters are : minimum and maximum radius of the neighbourhood as well as the minimal correlation
    # coefficient from which two neurons are considered correlated.
    # Not all neurons end up in a group, some maybe be left alone
    # So far only tested on partial data
