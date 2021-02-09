import h5py
import numpy as np
import logging
import os
import json
import getpass
import scipy.interpolate as ip
from PyQt5 import QtWidgets
logger = logging.getLogger('Fishlog')
from inspect import getsourcefile
from os.path import abspath, dirname


def layer_lags_correction(dict_data, group, name_mapping, parent=None):
    """
    Time interpolation to correct time offset per layer. Interpolation parameters are
    set inside this function.

    Interpolation is performed over chunks of the data.


    Parameters
    ----------
    dict_data:
        dictionary containing data (as defined in utilities.open_h5_data())
    group:
        handle of group in hfile where df_aligned should be written to (typically the group of df)
    name_mapping: dict
        dictionary with default_name:data_set_path, corresponds to all_keys dict in open_h5_data()

    Returns
    -------
    data_corrected

    """
    if parent is not None:
        tmp_msg = QtWidgets.QMessageBox.information(parent, 'Computing time layer offset correction', 'This can take several minutes, please wait. \n'
                                                                                                      'You can follow the progress in the status bar at the bottom of the Fishualizer interface.')

    logger.debug('Interpolating df data in time to correct the time lags between the layers')
    df = dict_data['df']
    times = dict_data['times']
    lags = dict_data['layerlags']
    # lags = coords[:,2]/(np.max(coords[:,2])+1)*(times[1]-times[0])  # toy lags for testing
    n_cells = df.shape[0]

    try:
        shape_int = df.shape # if dataformat == 'cellsXtimes' else df.shape[::-1]
        logger.debug(f'df_aligned data set created in h5 file with size {shape_int}')
        df_corrected = group.create_dataset('df_aligned', shape_int, dtype='float32')  # to avoid writing float64s
    except RuntimeError:
        df_corrected = group['df_aligned']

    n_chunks = 1  # number of h5 chunks to create

    interpolation_kind = 'cubic'  # interpolation type
    n_overlap = 3  # number of overlap time points (needed for smooth beginning of chunks)

    chunk_ix = np.linspace(0, len(times), n_chunks + 1, dtype=np.intp)  # create h5 chunks
    dtime = 0.25 * (times[1] - times[0])  # resolution of interpolation
    nsteps = 7  # factor of time resolution enhancement
    localtimes = np.linspace(-dtime, dtime, nsteps)  # extra time points locally (around some time point )
    allintertimes = np.zeros((len(times), len(localtimes)))  # create matrix of all new times
    for it in np.arange(len(times)):
        allintertimes[it, :] = times[it] + localtimes

    print(f'Starting interpolation with {n_chunks} chunks and {n_cells} neurons. \n Printing progress during interpolation.')
    for ichunk in np.arange(n_chunks):
        # Define indices to be worked on in this chunk
        cchunk = np.arange(chunk_ix[ichunk], chunk_ix[ichunk + 1])
        cchunk_leadin = np.arange(max(chunk_ix[ichunk] - n_overlap, 0), chunk_ix[ichunk + 1])  # time indices of chunk including lead in

        cdf = df[:, cchunk_leadin]  # select chunk df data
        cdf_interp = np.zeros((n_cells, len(cchunk)))  # interpolated df
        cintertimes = np.reshape(allintertimes[cchunk, :], -1)  # high resolution interpolation times to 1d array

        # Separate the cases for the different chunks to avoid
        nan_ix_start = []
        nan_ix_end = []
        if ichunk == n_chunks - 1:  # special treatment of the last time-point
            nan_ix_end = np.empty(int(np.floor(nsteps / 2)))
            nan_ix_end[:] = np.nan
            cintertimes = cintertimes[0:int(-np.floor(nsteps / 2))]
        if ichunk == 0:  # special treatment of the first time-point
            nan_ix_start = np.empty(int(np.floor(nsteps / 2)))
            nan_ix_start[:] = np.nan
            cintertimes = cintertimes[int(np.floor(nsteps / 2)):]

        mod_count = np.round(n_cells / 100)
        for ineuron in range(n_cells):
            if np.mod(ineuron, mod_count) == 0:
                print(f'  Progress: {np.round((ineuron+1)/n_cells * 100 * (ichunk+1)/n_chunks, 1)}%')  # print progress in console
                if parent is not None:
                    parent.statusBar().showMessage(f'  Progress: {np.round((ineuron+1)/n_cells * 100 * (ichunk+1)/n_chunks, 1)}%')  # print progress in status bar
            x = np.squeeze(times[cchunk_leadin] + lags[ineuron])  # add lag to original data (to get the corrected timing)
            y = cdf[ineuron, :]  # get df of this chunk of ineuron

            # PERFORM INTERPOLATION
            cinterpolator = ip.interp1d(x, y, fill_value='extrapolate', assume_sorted=True, kind=interpolation_kind)  # create interpolation function
            interp_tmp = cinterpolator(cintertimes)  # interpolate high res time points

            # Account for ends of the range
            if len(nan_ix_start):
                interp_tmp = np.concatenate((nan_ix_start, interp_tmp))
            if len(nan_ix_end):
                interp_tmp = np.concatenate((interp_tmp, nan_ix_end))

            cdf_interp[ineuron, :] = np.nanmean(np.transpose(np.reshape(interp_tmp, (len(cchunk), nsteps))), 0)  # average over local high res time points to get original time res
        df_corrected[:, cchunk] = cdf_interp  # save this chunk
        if parent is not None:
            parent.statusBar().showMessage('Interpolation done.')


def load_config():
    """
    Load the JSON configuration file and return parameters corresponding to current user

    Returns
    -------
    user_params: dict
    """
    username = getpass.getuser()
    BasePath = dirname(abspath(getsourcefile(lambda: 0)))
    with open(BasePath + '/../../Content/Config/config.json', 'r') as config_file:
        all_params = json.load(config_file)
    try:
        user_params = all_params[username]
    except KeyError:
        user_params = all_params['default']
    user_params['paths'] = {k: os.path.expanduser(p) for k, p in user_params['paths'].items()}
    # user_params['load_ram'] = np.bool(user_params['load_ram'])
    return user_params


def create_density_map(gridfile, map_type='density_map', den_threshold=None, den_scale=5):
    """Create 4D matrix which can be used to draw a density map (by Fishualizer.draw_density_map()).

    A grid file is loaded which contains coords (n, 3) and clusters (n, 1) each.
    The resulting 4D matrix is (x, y, z, RGBA). Alternatively; one could think of
    this as 4 different 3D matrices (x, y, z). Every value indicates the Red, Green,
    Blue, Alpha value respectively for the 4 matrices. Importantly; coordinates are NOT
    encoded in the matrices. GLVolumeItem assumes a 1x1x1 grid. This is rescaled
    in this function (using the info from gridfile['coords']).

    Parameters:
    -------------
        gridfile: str
            directory where the grid file with clusters is located
        map_type; str ('density_map', 'hard_threshold')
        den_threshold: float or None
            threshold for cut-off of density map. If None, it defaults to 0 for map_type == 'density_map'
            and to 0,0005 for map_type == 'hard_threshold'
        den_scale: float, int
            if maptype == 'density_map', the density is normalized to the max value, and
            the intensity (alpha) value is subsequently linearly scaled with the normalized
            density, multiplied by density_scale_factor
    """
    hfile = h5py.File(gridfile, 'r')
    data_names = list(hfile['Data'].keys())
    data = {}
    for dn in data_names:  # extract all data sets
        data[dn] = hfile['Data'][dn].value.transpose()

    x_vals = np.unique(data['coords'][:, 0])  # x values in grid
    n_x = len(x_vals)  # number of x values
    y_vals = np.unique(data['coords'][:, 1])
    n_y = len(y_vals)
    z_vals = np.unique(data['coords'][:, 2])
    n_z = len(z_vals)
    resolution = [np.mean(np.diff(x_vals)), np.mean(np.diff(y_vals)), np.mean(
        np.diff(z_vals))]  # it is exported as a cubic grid so resolutions should be equal in all dimensions
    min_coords = data['coords'].min(axis=0)  # minimum used to translate density map in Fishualizer.draw_density_map()

    cluster_names = []
    for dn in data_names:
        if dn != 'coords':
            data[dn + '_nf'] = np.reshape(data[dn], (n_x, n_y, n_z))  # put cluster densities in new format (1D -> 3D)
            cluster_names.append(dn)
    nf_cluster_names = [x for x in list(data.keys()) if x[-3:] == '_nf']  # list of names to use

    ## Assign RGBA values in series
    if den_threshold is None:
        if map_type == 'density_map':
            den_threshold = 0  # DENSITY MAP
        elif map_type == 'hard_threshold':
            den_threshold = 0.00005  # HARD THRESHOLD MAP

    n_clusters = len(nf_cluster_names)
    if n_clusters == 1: # single cluster = classical density plot
        colour = np.array([255,0,0,0])
        dataplot = np.tile(colour, (n_x, n_y, n_z, 1))
        cn = nf_cluster_names[0]
        cdensity = data[cn]
        cdensity[np.where(cdensity<den_threshold)] = 0
        tmp = cdensity / cdensity.mean() * 200 / n_x
        dataplot[:,:,:,3] = tmp.astype(np.ubyte)

    else: # multiple clusters
        dataplot = np.zeros((n_x, n_y, n_z) + (4,), dtype=np.ubyte)  # create 4D data matrix to plot (x,y,z,RGBA)
        colours = {nf_cluster_names[xx]: [255, 0, 113, 0] for xx in range(len(nf_cluster_names))}
        # colours = {nf_cluster_names[0]: [255, 0, 0, 0],
        #             nf_cluster_names[1]: [0, 255, 0, 0],
        #             nf_cluster_names[2]: [0, 0, 255, 0],
        #             nf_cluster_names[3]: [128, 128, 0, 0],
        #             nf_cluster_names[4]: [0, 128, 128, 0]}  # colours of clusters  # TODO: import color dict
        # nf_cluster_names = ['positive_mixed_nf', 'negative_mixed_nf', 'posnegderiv_high95_nf']
        # colours = {nf_cluster_names[0]: [255, 0, 113, 0],
        #            nf_cluster_names[1]: [0, 255, 157, 0],
        #            nf_cluster_names[2]: [184, 134, 11, 0]}  # hard coded colors of regression clusters Migault et al., 2018

        # maxnorm = {cn: data[cn].max() for cn in nf_cluster_names}  # max density per cluster (for colour normalization)
        maxnorm = {cn: 0.0025 for cn in nf_cluster_names}  # uniform max density (for colour normalization)

        for x in range(n_x):  # loop through all coords to assign RGBA
            for y in range(n_y):
                for z in range(n_z):
                    max_den = 0
                    for cn in nf_cluster_names:  # check all clusters to find max one
                        if (data[cn][x, y, z] > den_threshold) and (data[cn][x, y, z] > max_den):
                            max_den = np.maximum(data[cn][x, y, z], max_den)  # DENSITY MAP
                            dataplot[x, y, z, :] = colours[cn]
                            if map_type == 'density_map':
                                dataplot[x, y, z, 3] = (max_den / maxnorm[cn] * 100) * den_scale  # DENSITY MAP
                            elif map_type == 'hard_threshold':
                                dataplot[x, y, z, 3] = 100  # HARD THRESHOLD MAP

    return dataplot, resolution, min_coords


def load_zbrain_regions(recording, zbrainfile=None):
    """Load ZBrainAtlas regions that are saved in the custom-format .h5 file.

    Parameters:
    ------------
        recording: instance of Zecording class
            Data is added to this recording
        zbrainfile: str (default None)
            directory where file is located, if None it defaults to hard-coded dir.

    Returns:
    ----------
        bool: indicating success

    """
    if zbrainfile is None:
        BasePath = dirname(abspath(getsourcefile(lambda: 0)))   # goes to parent folder of utilities.py?  needs 2 extra parent commands, see next line
        BasePath = dirname(dirname(BasePath))
        zbrainfile = BasePath + '/Content/ZBrainAtlas_Outlines.h5'

    if zbrainfile[-3:] == '.h5':
        hfile = h5py.File(zbrainfile, 'r')
        data_names = list(hfile.keys())
        data = {}
        for dn in data_names:  # extract all data sets
            data[dn] = hfile[dn].value.transpose()

        if 'region_indices' in data_names and 'grid_coordinates' in data_names:
            data['region_indices'] = data['region_indices'].astype(
                'int') - 1  # convert from 1-indexing (matlab) to zero-indexing

            ## Below: change to correct orientation
            max_grid_coords = np.squeeze(data['resolution']) * np.squeeze([data['height'], data['width'], data['Zs']])
            long_axis_flipped = max_grid_coords[0] - data['grid_coordinates'][:, 0]
            # data['grid_coordinates'][:, 0] = long_axis_flipped.copy()
            data['grid_coordinates'][:, 0], data['grid_coordinates'][:, 1] = (data['grid_coordinates'][:, 1]).copy(), (
                data['grid_coordinates'][:, 0]).copy()
            data['grid_coordinates'] = data['grid_coordinates'] / 1000  # go to mm

            setattr(recording, 'zbrainatlas_coordinates', data['grid_coordinates'])  # add to recording
            recording.available_data.add(
                'zbrainatlas_coordinates')  # don't use [..]_coords here because coords is used for plotting
            setattr(recording, 'zbrainatlas_regions', data['region_indices'])
            recording.available_data.add('zbrainatlas_regions')
            logger.info('ZBrainAtlas succesfully added')
            return True
        else:
            logger.warning(
                f'ZBrainAtlas not loaded because region_indices and grid_coordinates were not found in the file {zbrainfile}')
            return False
