from PyQt5 import QtWidgets, QtCore, QtGui
import h5py
import json
import logging
from functools import wraps
from inspect import signature, getdoc
import shelve
import numpy as np
from collections import defaultdict
import scipy.sparse
from pathlib import Path
import psutil as ps
from typing import Any, Tuple
from Controls import DsetConflictDialog
logger = logging.getLogger('Fishlog')
from inspect import getsourcefile
from os.path import abspath, dirname


def get_data_orientation(coords: np.ndarray, data: np.ndarray):
    """
    Try to guesstimate the orientation of some data based on the coordinates

    Parameters
    ----------
    coords: 2D-array
        Spatial coordinates of neurons
    data: 2D-array
        Any kind of 2D array with one dimension being cells and the other one being time

    Returns
    -------
    n_cells: int
    n_times: int
    orientation: str
        Can be 'cellsXtimes', 'timesXcells'
    """
    n_cells = max(coords.shape)
    if data.shape[0] == n_cells:
        return n_cells, data.shape[1], 'cellsXtimes'
    elif data.shape[1] == n_cells:
        return n_cells, data.shape[0], 'timesXcells'
    else:
        raise ValueError('Data shape does not match the coordinates')


def open_h5_data(filepath, ignorelags=False, forceinterpolation=False,
                 ignoreunknowndata=False, parent=None, loadram=False):
    """
    Soft-load data names from hdf5 file. Guess by name, toggle between coords and
    ref_coords, load labels from hdf5 file, if not load from filepath (in load_data())
    in Fishualizer.py.

    Parameters
    ----------
    filepath: str or Path
    ignorelags: bool (default False)
        whether to ignore time lags between z layers
    forceinterpolation: bool (default False)
        whether to force time interpolation (between layers) even if df_aligned is provided in the h5 file
    ignoreunknowndata: bool (default False)
        whether to ignore unknown data, meaning data that is not recognised by the
        default names dictionary. If true, the user is prompted to assign it to a
        default data set (or cancel), if false this is not asked.
    parent: None or QDialog?
    loadram: boolean

    Returns
    -------
    data: dict
          Dictionary containing the loaded data only, possibilities:
              df: Calcium signal, default shape is Neurons X Times (note: inverted in Matlab H5 saving)
              coords: Position of cells in own space
              ref_coords: Position of cells in reference space
              behavior: behavioral data
              stimulus: stimulus data
              times: Time points
              spikes: spike data
              labels: labels of anatomical alignment with ZBrainAtlas
              not_neuropil: boolean values, True=not neuropil, False=neuropil
    h5_file: File handle
        Handle on the hdf5 file with the data
    assigned_names: dict
        Dictionary containing the mappings between default names (as keys) and
        hdf5 file names (as values)
    """
    # dataformat: 'cellsXtimes' or 'timesXcells' or None (default None)
    #     data orientation
    #    if None, tries to guess with `get_data_orientation`
    # open h5 file in read+ modus
    logger.debug(f'Opening file {filepath} with following options: '
                 f'ignorelags: {ignorelags}, forceinterpolation: {forceinterpolation}, '
                 f'ignoreunknowndata: {ignoreunknowndata}, loadram: {loadram}')

    all_keys = dict()

    def get_datasets(data_path, obj):
        if isinstance(obj, h5py.Dataset):  # obj is a dataset
            tmp_name = data_path.split('/')[-1].lower()
            all_keys[tmp_name] = obj.name

    h5_file = h5py.File(filepath, 'r+')
    h5_file.visititems(get_datasets)
    BasePath = dirname(abspath(getsourcefile(lambda: 0)))
    with open(BasePath + '/../../Content/data_names_links.json', 'r') as tmp:
        names_dict = json.load(tmp)
    # import default names as sets to  {default name: set(possible names)}
    # default_names = {x: set(y) for x, y in names_dict.items()}
    reverse_default_names_map = {vv: k for k, v in names_dict.items() for vv in  #TODO: force all lowercase (in case users input a non-lowercase data name in names_dict file)
                                 v}  # {possible name: default name} mapping
    logger.debug(f'All data paths in h5 file: {all_keys}')
    non_assigned_names = {}
    assigned_names = {}
    data = {}
    static_data = {}
    name_mapping = {}
    conflicting = {}
    # Loop through all data sets in all groups in h5 file and map the names of the datasets to predefined names
    for name, dataset_path in all_keys.items():
        # check data set against standard names (in .json dictionary of links?)
        try:
            tmp_default_name = reverse_default_names_map[name]  # only works if 'name' is among predefined names that map to this dict item, abort (via try) if fails, otherwise continue
            assigned_names[tmp_default_name] = name
            if tmp_default_name in name_mapping.keys():
                # TODO: test this functionality
                names = [name_mapping[tmp_default_name], dataset_path]
                names.extend(conflicting.get(tmp_default_name, []))
                conflicting[tmp_default_name] = names
                logger.debug(conflicting)
            else:  # not defined yet, add to data
                name_mapping[tmp_default_name] = dataset_path

        except KeyError:  # data name not defined
            # TODO: write to json file?
            # Not sure what to do here because if we think of the new format the LJP wants to adopt
            # it would be a lot of non-assigned datasets
            non_assigned_names[name] = dataset_path

    for dset_name, conflicts in conflicting.items():
        if parent is not None:
            chosen_win = DsetConflictDialog(parent, 'Conflicting data sets',
                                            f'Which dataset from the file should be used as {dset_name}?', conflicts)
            chosen = chosen_win.get_datapath()
            name_mapping[dset_name] = chosen

    # Guessing data orientation of df
    n_cells, n_times, dataformat = get_data_orientation(h5_file[name_mapping['coords']], h5_file[name_mapping['df']])
    logger.debug(f'df specifics: n_cells: {n_cells}, n_times: {n_times}, dataformat: {dataformat}')

    # Loop through all the mapped datasets, load, transpose ...
    for name, dataset_path in name_mapping.items():
        if name not in ['df', 'df_aligned', 'spikes']:
            # This is not activity data, so we load in memory
            data[name] = np.float32(h5_file[dataset_path][()])  # load data from h5file  in RAM
            # Sometimes data are in the wrong orientation (X, n_cells). We correct for this here
            if data[name].shape[1] == n_cells and data[name].shape[0] != n_cells:
                data[name] = data[name].T
            if (name == 'behavior' or name == 'stimulus') and data[name].shape[-1] != n_times:
                data[name] = data[name].T
            logger.info(f'{name} is loaded in RAM with shape {data[name].shape}')
        else:
            # If df_aligned is available and we don't want to do the interpolation again, we don't load df
            if name == 'df' and assigned_names.get('df_aligned', None) in all_keys.keys() and not forceinterpolation and not ignorelags:
                logger.debug('df data not loaded because df_aligned in data set.')
                continue
            elif name == 'df_aligned' and assigned_names.get('df', None) in all_keys.keys() and (forceinterpolation or ignorelags):
                logger.debug(f'df_aligned data not loaded because df in data set and ignorelags={ignorelags}.')
                continue

            if name == 'df_aligned':
                new_name = 'df'  # save as df (because df is not loaded anyway)
            else:
                new_name = name

            # Guessing data orientation of current data (df, df_aligned and spikes are not necessarily all aligned)
            n_cells, n_times, dataformat = get_data_orientation(h5_file[name_mapping['coords']], h5_file[name_mapping[name]])
            logger.debug(f'{name} specifics: n_cells: {n_cells}, n_times: {n_times}, dataformat: {dataformat}')
            if not loadram and dataformat == 'timesXcells' and parent is not None:
                # Should we load in RAM because wrong orientation for memory mapping
                bytes_per_el = np.dtype(h5_file[name_mapping[name]].dtype).itemsize  # number of bytes per element (dependent on data type)
                n_els = np.product(h5_file[name_mapping[name]].shape)  # number of elements ( in 2D array)
                d_size = np.round((n_els * bytes_per_el) / (1024 ** 2))  # data size in MB
                available_ram_memory = np.round(ps.virtual_memory().available / (1024 **2))  # in MB
                logger.debug(f'Data {name} with size {d_size}MB, RAM availability {available_ram_memory}MB.')
                if d_size > (0.2 * available_ram_memory):  # if greater than 20% of available ram, ask for permission to load in ram
                    logger.debug('Prompting whether data can be loaded in RAM because it must be transposed')
                    loadram = QtWidgets.QMessageBox.question(parent, 'Loading data in memory',
                                                             f'The {name} data must be loaded in memory, '
                                                             f'because its orientation is not handled otherwise. '
                                                             f'This will require {d_size:.0f}MB of RAM ({available_ram_memory:.0f}MB currently available). \n'
                                                             'Would you like to continue (press Yes) or abort loading (press No)?')
                    loadram = True if loadram == QtWidgets.QMessageBox.Yes else False  # enable loading in RAM
                else:
                    logger.debug(f'Data {name} takes less than 20% of the available RAM, so it is automatically loaded in RAM.')
                    loadram = True

            if loadram:
                data[new_name] = h5_file[dataset_path][()]  # load in RAM
                if dataformat == 'timesXcells':
                    data[new_name] = data[new_name].T  # transpose if necessary
                logger.info(f'{name} is loaded in RAM with shape {data[new_name].shape}')
            elif not loadram and dataformat == 'cellsXtimes':  # if to be memory-mapping and in right format
                data[new_name] = h5_file[dataset_path]  # memory map
                logger.info(f'{name} is loaded with memory-mapping with shape {data[new_name].shape}')
            elif not loadram and dataformat == 'timesXcells' and name != 'spikes':  # if to be memory mapping, but in incorrect format
                # Memory mapping not possible; Abort loading
                h5_file.close()
                logger.info(f'{name} not loaded, loading procedure has been aborted. \n Please load this data set in RAM, or choose a different data set.')
                print(f'{name} not loaded, loading procedure has been aborted. \n Please load this data set in RAM, or choose a different data set.')
                return None, None, None

    bool_static_added = False
    for name, dataset_path in non_assigned_names.items():  # loop through non assignmend/recognised name to see if Nx1 data sets exist
        tmp_size = h5_file[dataset_path].shape
        if len(tmp_size) == 1 and np.squeeze(tmp_size) == n_cells:
            static_data[name] = np.float32(np.squeeze(h5_file[dataset_path].value))
            logger.info(f'{name} added as static data set.')
            bool_static_added = True
        else:
            logger.warning(f'{name} with shape {tmp_size} is not recognized, so it cannot be loaded.')
    if bool_static_added:
        data['_additional_static_data'] = static_data  # will be unpacked in zecording

    if (not ignorelags and ("df_aligned" not in all_keys.keys())) or forceinterpolation:  # if open to interpolation
        if ('layerlags' in name_mapping.keys()):# or forceinterpolation:  # thijs; I have removed force here, because no automated layer creation is implemented (I would do this by hand when testing, the relevant line is in comments in layer_lags_correction())
            if ('times' in name_mapping.keys()) and ('df' in name_mapping.keys()):  #required data for layer lag correction
                interp_warn = True
                if parent is not None:
                    tmp_name_layers = name_mapping['layerlags']
                    interp_warn = QtWidgets.QMessageBox.question(parent, 'Time correction for different layers',
                                                                 f'Time delays per layer were found in the h5 data set ({tmp_name_layers}). \n'
                                                                 'Do you want to correct the timing? This is done by interpolating all layers by '
                                                                 'their respective offset. This will take several minutes, but the result'
                                                                 ' will be saved as an additional data set in the h5 file, called df_aligned, so that'
                                                                 ' it can readily be used in the future.')
                    interp_warn = True if interp_warn == QtWidgets.QMessageBox.Yes else False
                if interp_warn:
                    group_df = h5_file[name_mapping['df'].rsplit('/', 1)[0]]  # the group where df is located, df_aligned will be saved here
                    layer_lags_correction(data, group_df, name_mapping, parent=parent)
                    data['df'] = group_df['df_aligned']
                    logger.info(f'Time layer correction is a success! df_aligned saved in {group_df} and loaded.')
                else:
                    logger.info('Time correction of data was canceled by user. `ignorelags` set to True')
                    ignorelags = True
        else:
            logger.warning(
                f"No lags across layers provided in HDF5 file ({filepath}), using uncorrected data, 'ignorelags' set to True.")
            ignorelags = True

    if 'df' not in data:  # necessary to plot
        logger.error(f'No df/f values loaded from data file {filepath}')

    # TODO: align behavior? see below
    else:  # use shape of df to transpose other data sets if needed
        # if 'behavior' in data:
        #     if len(data['behavior'].shape) > 1:  # in case 'behavior' is squeezed 1D, not needed.
        #         if data['behavior'].shape[1] is not data['df'].shape[1]:
        #             data['behavior'] = data['behavior'].transpose() doesn't work with one dataset  # numpy transpose

        if 'stimulus' in data:
            if len(data['stimulus'].shape) > 1:
                if data['stimulus'].shape[1] is not data['df'].shape[1]:
                    data['stimulus'] = data['stimulus'].transpose()

    if any([(x in data.keys()) for x in
            ['coords', 'ref_coords', 'zbrain_coords']]) is False:  # no coordinates available
        logger.error(f'No coordinates loaded from data file {filepath}')

    if 'times' not in data.keys():  # necessary to plot, create np.arange(T) otherwise
        data['times'] = np.arange(np.shape(data['df'])[1])
        data['times'] = data['times'][np.newaxis, :]  # it must be 2D for somewhere in Fishualizer code
        logger.info('time vector created')

    if 'labels' in data:
        """
        HDF5 can not directly store sparse matrices, so it is imported as full
        matrix. It is changed to scipy.sparse so it is compatible with Fishualizer code and for
        memory usage.
        """
        data['labels'] = scipy.sparse.csr_matrix(data['labels'])

    # TODO: Let's get rid of this neuropil data sets/condition?
    if 'neuropil' in data and 'not_neuropil' not in data:  # not_neuropil is needed for Fishualizer.py
        data['not_neuropil'] = np.logical_not(data['neuropil'])
        del data['neuropil']  # because only not_neuropil is needed (and they contain the same information)

    if 'not_neuropil' in data:
        data['not_neuropil'] = data['not_neuropil'].astype('bool')
        data['not_neuropil'] = np.squeeze(
            data['not_neuropil'])  # because it has to be 2D to be loaded by transpose (1,0)

    # assigned_names is returned for coordinate choice when multiple coordinate sets are present (in Fishualizer.py)
    return data, h5_file, assigned_names  # pass on to Zecording class


class KeyDefaultDict(defaultdict):
    def __missing__(self, key):
        if self.default_factory is None:
            raise KeyError(key)
        else:
            ret = self[key] = self.default_factory(key)
            return ret


class Zecording(object):
    # TODO: Implement the loading of static data set from shelve in the GUI
    def __init__(self, path: str, **kwargs) -> None:
        if path == '':
            self.path = None
            return
        self.path = Path(path)
        if not self.path.is_file():
            raise ValueError(f'{path} does not exist')
        shelf_path = self.path.parent / self.path.stem
        self.shelf_base_path = shelf_path.as_posix()
        # Shelf structure:
        # Function_name: {'name': function name
        #                 'args': list argument names,
        #                 'data': [{((arg1, value1), (arg2, value2): results},
        #                 ]}
        self._data, self.file, self._data_name_mappings = open_h5_data(self.path.as_posix(),
                                                                       **kwargs)  # load data
        if self._data is None:  # None is returned if data loading was aborted
            pass
            #TODO: handle ensuing errors (e.g. return Zecording=None, and handle in Fishualizer?)

        self._sel_coords = self.coords
        self._avail_data = None
        self._single_data = None
        self._multi_data = None
        self._sr = None  # sampling rate
        self._structural_data = None
        self._spatial_scale_factor = np.array([1, 1, 1])  # default to (effectively) no scaling
        self.datasets = KeyDefaultDict(self.__getattribute__,
                                       {'df': self.df, 'spikes': self.spikes, 'input': self.input,
                                        'output': self.output})
        if 'parent' not in kwargs.keys() and '_additional_static_data' in self._data.keys():  # if no parent (ie loaded from script != fishualizer), and static data => add static data right now (because otherwise it is done in Fishualizer.load_data())
            for s_data_name, s_data_set in self._data['_additional_static_data'].items():
                self.add_supp_single_data(s_data=s_data_set, s_name=s_data_name)

        # TODO: Keep track of the static datasets?
        logger.info('Zecording object created')

    @property
    def n_cells(self):
        return self.df.shape[0]

    @property
    def df(self):
        return self._data['df']

    @property
    def coords(self):
        return self._data['coords']

    @property
    def time(self):
        t = np.squeeze(self._data['times'])
        return t

    # The class can also be used to provide aliases
    calcium = df
    times = time
    t = time

    @property
    def n_times(self):
        return len(self.times)

    @property
    def structural_data(self):
        try:
            return self._data['structural_data']
        except KeyError:
            logger.debug(f'Data does not contain structural_data')
            return None

    @property
    def sampling_rate(self):
        # we can additionally load from hdf5 (as attribute?)
        if self._sr is None:  # if not defined, guess
            # the advantage of doing this here instead of in __init__() is that now you can always reset SR to None
            # (in the console)
            self._sr = 1 / (np.mean(np.diff(self.times)))  # guess sr
        return self._sr

    @sampling_rate.setter
    def sampling_rate(self, value):
        self._sr = value

    @property
    def data_names(self):
        return self._data_name_mappings

    @property
    def ref_coords(self):
        try:
            return self._data['ref_coords']
        except KeyError:
            logger.debug(f'Data does not contain reference coordinates')
            return None

    @property
    def zbrain_coords(self):
        try:
            return self._data['zbrain_coords']
        except KeyError:
            logger.debug(f'Data does not contain ZBrain coordinates')
            return None

    @property
    def behavior(self):
        try:
            if len(self._data['behavior'].shape) == 2:
                return np.squeeze(self._data['behavior'][0, :])
            elif len(self._data['behavior'].shape) == 1:
                return np.squeeze(self._data['behavior'])
            else:
                return self._data['behavior']
        except KeyError:
            logger.debug(f'Data does not contain behavioral data')
            return None

    @property
    def spikes(self):
        try:
            return self._data['spikes']
        except KeyError:
            logger.debug(f'Data does not contain spiking data')
            return None

    @property
    def stimulus(self):
        try:
            return np.squeeze(self._data['stimulus'])
        except KeyError:
            logger.debug(f'Data does not contain stimulus data')
            return None

    @property
    def not_neuropil(self):
        try:
            return self._data['not_neuropil']
        except KeyError:
            logger.debug(f'Data does not contain neuropil information')
            return None

    @property
    def labels(self):
        try:
            return self._data['labels']
        except KeyError:
            logger.debug(f'Data does not contain ZBrainAtlas labels')
            return None

    @labels.setter
    def labels(self, value):
        # This is implemented just to make the 'old way' of loading labels work
        # It might be removed once the large SampleData file is converted
        self._data['labels'] = value

    @property
    def phase_map(self):
        try:
            return self._data['phase_map']
        except KeyError:
            return None

    @property
    def layerlags(self):
        try:
            return self._data['layerlags']
        except KeyError:
            return None

    @property
    def input(self):
        try:
            return self._data['input']
        except KeyError:
            return None

    @property
    def output(self):
        try:
            return self._data['output']
        except KeyError:
            return None

    @output.setter
    def output(self, value):
        # Seems to be required in one place in the Fishualizer (when input and output are the same)
        self._data['output'] = value

    @property
    def spatial_scale_factor(self):
        return self._spatial_scale_factor

    @spatial_scale_factor.setter
    def spatial_scale_factor(self, scaling_factors):
        try:
            x_scalef, y_scalef, z_scalef = scaling_factors
        except ValueError:
            logger.warning('Pass an iterable with three items (xscale, yscale, zscale)')
            x_scalef, y_scalef, z_scalef = (1, 1, 1)  # use default scaling
        self._spatial_scale_factor = np.array([x_scalef, y_scalef, z_scalef])

    @property
    def sel_frame(self):
        raw_coords = self._sel_coords  # current selection
        scaling = self.spatial_scale_factor  # current scaling
        scaled_coords = scaling * raw_coords  # scale coordinates
        return scaled_coords

    @sel_frame.setter
    def sel_frame(self, value):
        """
        Ability to select a property of Zecording object (eg for plotting purposes)
        One should prefer to set it either to self.coords, or self.ref_coords or any properties like this.
        It could also be set to any other function (even outside of this class) as long as it
        returns data in the proper format

        Parameters
        ----------
        value: function
            Method to call to get neuron coordinates
        """
        self._sel_coords = value

    def close(self):
        self.file.close()
        logger.info(f'File {self.path} closed')

    @property
    def available_data(self):
        if self._avail_data is None:
            property_names = [p for p in dir(Zecording) if isinstance(getattr(Zecording, p), property)]
            available = [p for p in property_names if p != 'available_data' and self.__getattribute__(p) is not None]
            self._avail_data = set(available)
        return self._avail_data

    @property
    def single_data(self):
        """Names (str) of data sets in Zecording that are a single time traces
        """
        if self._single_data is None:
            default_options = {'behavior', 'stimulus'}
            self._single_data = set()
            for name in default_options:
                if name in self.available_data:
                    self._single_data.add(name)
        return self._single_data

    @single_data.setter
    def single_data(self, value):
        self._single_data = value

    @property
    def multi_data(self):
        """Names (str) of data sets in Zecording that are multi time traces
        """
        if self._multi_data is None:
            self._multi_data = {'df', 'spikes'}
        return self._multi_data

    @multi_data.setter
    def multi_data(self, value):
        self._multi_data = value

    # Fixed: Make the saving optional: are the extra () annoying or OK?
    def register_func(self, func, save=True):
        sig = signature(func)
        arg_names = list(sig.parameters.keys())[1:]  # First argument must be Zecording
        f_name = func.__name__
        shelve_path = self.shelf_base_path + f'_{f_name}'
        if save:
            with shelve.open(shelve_path) as db:
                if 'args' not in list(db.keys()):
                    db['args'] = arg_names
                    db['name'] = f_name
                    db['doc'] = getdoc(func)
                    if 'data' not in list(db.keys()):
                        db['data'] = {}  # This will store all outputs of the registered function

        @wraps(func)
        def wrapped_func(*args, **kwargs):
            c_args = tuple([(n, v) for n, v in zip(arg_names, args)])
            for k, v in kwargs:
                c_args += ((k, v),)
            def_args = set(arg_names) - set([a[0] for a in c_args])  # Get missing arguments
            # Assign their default values to the missing arguments
            for da in def_args:
                c_args += ((da, sig.parameters[da].default),)
            r = func(self, *args, **kwargs)
            c_data = {c_args: r}  # {(arguments, values) : returned result}
            if save:
                with shelve.open(shelve_path) as f_db:
                    prev_res = f_db['data']
                    prev_res.update(c_data)
                    f_db['data'] = prev_res

            return r
        # TODO: store in a sub-object (self.compute.f_name)
        setattr(self, f_name, wrapped_func)  # We make 'func' a method of this Zecording
        return wrapped_func

    def compute_analysis(self, parent, func, **kwargs):
        """
        Applying some analysis function to a Zecording object
        Named parameters can be passed
        Result is stored in `func_name_res`

        Parameters
        ----------
        parent: the Fishualizer class
            which contains all its functions
        func: function
            Function to apply
            Must have signature of the form func(zecording, kwargs)
        kwargs: dict
            Additional arguments to be passed to `func`

        Returns
        -------

        res
            Result from the analysis function
        res_name: str
            Assigned name to the analysis result as added to the Zecording available data

        """
        res_name = f'{func.__name__}_res'
        parent.statusBar().showMessage(func.__name__)  # FIXME: Not a great idea when this is used outside the GUI
        res = func(self, parent, **kwargs)
        # setattr(self, res_name, res)  # this is done in the Fishualizer.py in add_static()
        self._avail_data.add(res_name)
        return res, res_name  # return to feed into add_static

    def add_supp_single_data(self, s_name, s_data):
        """Function to add supplementary single data to rec.
        Parameters:
            - s_name, str: name of new data trace
            - s_data, float with shape of single time trace

        """
        setattr(self, s_name, s_data)
        self.available_data.add(s_name)
        self.single_data.add(s_name)

    def reverse_coords(self, dim=2):
        """ Function to reverse the z coordinates.

        Parameters:
         dim: int 0 1 or 2
            dimension of coords to reverse (0 x, 1 y, 2 z)
        """
        if 'coords' in self.available_data:
            self.coords[:, dim] = self.coords[:, dim].max() + self.coords[:, dim].min() - self.coords[:, dim]
        if 'ref_coords' in self.available_data:
            self.ref_coords[:, dim] = self.ref_coords[:, dim].max() + self.ref_coords[:, dim].min() - self.ref_coords[:, dim]
        if 'zbrain_coords' in self.available_data:
            self.zbrain_coords[:, dim] = self.zbrain_coords[:, dim].max() + self.zbrain_coords[:, dim].min() - self.zbrain_coords[:, dim]
        if 'zbrainatlas_coordinates' in self.available_data:
            self.zbrainatlas_coordinates[:, dim] = self.zbrainatlas_coordinates[:, dim].max() + self.zbrainatlas_coordinates[:, dim].min() - self.zbrainatlas_coordinates[:, dim]
        else:
            print('WARNING: Zbrain atlas coordinates not yet loaded and therefore not reversed!! ')

    def __getitem__(self, dataset: str):
        try:
            return getattr(self, dataset)
        except AttributeError:
            return ValueError(f'{dataset} not a valid data set')

    def __repr__(self) -> str:
        return f'Recording from {self.path}'

    __str__ = __repr__


def zanalysis(save=True):
    def zdecorator(func):
        @wraps(func)
        def wrapped(*args, **kwargs):
            args = list(args)
            z: Zecording = args.pop(0)
            z.register_func(func, save=save)
            z_meth = getattr(z, func.__name__)
            return z_meth(*args, **kwargs)
        return wrapped
    return zdecorator


def list_calls(shelve_path: str):
    """
    List all the function calls saved in a shelve file

    Parameters
    ----------
    shelve_path: str
        Path to a shelve file

    Returns
    -------
    f_name: str
        Function name
    f_doc: str
        Docstring of the saved function
    args: list
        List of arguments name
    calls: list of tuple
        Each element represent a function call (and its corresponding result)
        Each element is a tuple with all parameters and its corresponding value
    """
    with shelve.open(shelve_path) as db:
        if 'args' not in db.keys() or 'name' not in db.keys():
            raise ValueError('Invalid shelve file')
        args = db['args']
        f_name = db['name']
        f_doc = db.get('doc', '')
        if 'data' not in db.keys():
            return f_name, f_doc, args, []
        return f_name, f_doc, args, list(db['data'].keys())


def get_call(shelve_path: str, call: tuple) -> Tuple[str, Any]:
    """
    Open a shelve and extract a given call.
    Returns the corresponding data
    Parameters
    ----------
    shelve_path
    call

    Returns
    -------
    dataset_name: str
    data: Any
    """
    with shelve.open(shelve_path) as db:
        func_name = db['name']
        # FIXME: This loads in memory, we could be smarter by memmapping for example
        data = db['data'].get(call)
    fragments = [f'{arg_name}={value}' for arg_name, value in call]
    dataset_name = f'{func_name}({", ".join(fragments)})'
    return dataset_name, data


@zanalysis()
def f(z: Zecording, x, y=3):
    """
    This function would take a Zecording object as its first argument
    Other parameters, with optional default values
    Computes something and then returns it
    """
    return z.n_cells


@zanalysis(save=False)
def g(z: Zecording, x, y=3):
    """
    This function would take a Zecording object as its first argument
    Other parameters, with optional default values
    Computes something and then returns it
    """
    return z.n_cells


class DataTableDelegate(QtWidgets.QItemDelegate):
    """
    Take care of creating the widgets used to select data sets
    """
    def __init__(self, parent) -> None:
        super(DataTableDelegate, self).__init__()
        self.parent = parent

    def createEditor(self, parent: QtWidgets.QWidget, option: 'QStyleOptionViewItem',
                     index: QtCore.QModelIndex) -> QtWidgets.QWidget:
        if index.column() == index.model().columnCount()-1:
            # Checkbox column is weird
            return None

        return super(DataTableDelegate, self).createEditor(parent, option, index)

    def paint(self, painter: QtGui.QPainter, option: 'QStyleOptionViewItem',
              index: QtCore.QModelIndex) -> None:
        if index.column() == index.model().columnCount()-1:
            new_rect = QtWidgets.QStyle.alignedRect(option.direction, QtCore.Qt.AlignCenter,
                                                    QtCore.QSize(option.decorationSize.width(),
                                                                 option.decorationSize.height()),
                                                    QtCore.QRect(option.rect.x(), option.rect.y(),
                                                                 option.rect.width(), option.rect.height())
                                                   )
            self.drawCheck(painter, option, new_rect,
                           QtCore.Qt.Unchecked if int(index.data()) == 0 else QtCore.Qt.Checked)
        else:
            return super(DataTableDelegate, self).paint(painter, option, index)

    def editorEvent(self, event, model, option, index):
        """
        Change the data in the model and the state of the checkbox
        if the user presses the left mousebutton and this cell is editable. Otherwise do nothing.
        """
        if index.column() == index.model().columnCount()-1 and\
                event.type() == QtCore.QEvent.MouseButtonRelease and\
                event.button() == QtCore.Qt.LeftButton:
            # Change the checkbox-state
            logger.debug(index.data())
            model.setData(index, True if int(index.data()) == 0 else False, QtCore.Qt.EditRole)
            return True
        return super(DataTableDelegate, self).editorEvent(event, model, option, index)


class ArgModel(QtCore.QAbstractTableModel):
    def __init__(self, data: list, columns=(), parent=None):
        """
        Initialize a Abstract Table Model to store the log data

        Parameters
        ----------
        data: list
            Each element is a tuple of tuples, each pair containing the arg name and its value
        columns: tuple of str
            All arguments name
        parent
        """
        super().__init__(parent)
        # if len(data) == 0:
        #     all_args = tuple([(arg, 'None') for arg in columns])
        #     data = {all_args: ''}
        self._data = data
        self.to_import = [False for _ in data]
        self.columns = tuple(columns) + ('Import?', )

    def setData(self, index, value, role=None):
        if not index.isValid():
            return False
        if index.column() != self.columnCount(index.parent()) - 1:
            return False
        self.to_import[index.row()] = value
        self.dataChanged.emit(index, index, (QtCore.Qt.DisplayRole, ))
        return True

    def flags(self, index):
        if index.column() == self.columnCount() - 1:
            return QtCore.Qt.ItemIsEnabled | QtCore.Qt.ItemIsEditable
        else:
            return QtCore.Qt.ItemIsEnabled

    def rowCount(self, parent=None):
        return len(self._data)

    def columnCount(self, parent=None):
        try:
            return len(self.columns)
        except IndexError:
            return 0

    def data(self, index, role):
        if not index.isValid():
            return QtCore.QVariant()
        elif role == QtCore.Qt.DisplayRole:
            if index.column() == self.columnCount(index.parent()) - 1:
                return QtCore.QVariant(self.to_import[index.row()])
            c_call = self._data[index.row()]
            value_arg = c_call[index.column()][1]
            return QtCore.QVariant(value_arg)
        else:
            return QtCore.QVariant()

    def headerData(self, col, orientation, role):
        if orientation == QtCore.Qt.Horizontal and role == QtCore.Qt.DisplayRole:
            return QtCore.QVariant(self.columns[col])
        return QtCore.QVariant()


class ShelveWindow(QtWidgets.QDialog):
    logger = logging.getLogger('Fishlog')

    def __init__(self, parent, shelve_path: str):
        super().__init__(parent)
        # self.shelve_path = Path(shelve_path)
        sp = Path(shelve_path)
        self.shelve_path = sp.parent / sp.stem
        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        self.setWindowTitle("Choose datasets to import")
        v_layout = QtWidgets.QVBoxLayout()
        btn_lyt = QtWidgets.QHBoxLayout()
        self.setLayout(v_layout)

        self.ok_btn = QtWidgets.QPushButton('&OK')
        self.cancel_btn = QtWidgets.QPushButton('&Cancel')
        self.ok_btn.clicked.connect(self.accept)
        self.cancel_btn.clicked.connect(self.reject)
        self.data_table = QtWidgets.QTableView(self)

        self.func_label = QtWidgets.QLabel('Function name')
        self.doc_label = QtWidgets.QLabel('')
        btn_lyt.addSpacerItem(QtWidgets.QSpacerItem(10, 1, QtWidgets.QSizePolicy.Expanding))
        btn_lyt.addWidget(self.cancel_btn)
        btn_lyt.addWidget(self.ok_btn)
        v_layout.addWidget(self.func_label)
        v_layout.addWidget(self.doc_label)
        v_layout.addWidget(self.data_table)
        v_layout.addLayout(btn_lyt)

        f_name, f_doc, f_args, data = self.get_shelve_info()
        self.func_label.setText(f'Data computed by the function: {f_name}')
        self.doc_label.setText(f_doc)
        self.model = ArgModel(data, f_args, self.data_table)
        self.data_table.setModel(self.model)
        self.data_table.setItemDelegate(DataTableDelegate(self.data_table))

    def get_shelve_info(self):
        f_name, f_doc, f_args, calls = list_calls(shelve_path=self.shelve_path.as_posix())
        return f_name, f_doc, f_args, calls

    def get_imported_datasets(self):
        if self.exec_():
            calls_to_export = {}
            for call, to_imp in zip(self.model._data, self.model.to_import):
                if not to_imp:
                    continue
                dset_name, data = get_call(self.shelve_path.as_posix(), call)
                calls_to_export[dset_name] = data
            return calls_to_export



@zanalysis()
def mean_rate(z: Zecording):
    """
    This function would take a Zecording object as its first argument
    Other parameters, with optional default values
    Computes something and then returns it
    """
    mean_rate = np.mean(z['df'],1) # average over time, mean rate per neuron

    return mean_rate


if __name__ == '__main__':
    z = Zecording('/Users/englitz/GoogleDrive/Code/python/Fishualizer/Data/Data20140827_spont/SampleData.h5',
    loadram=True)
    # z.register_func(za.mean_rate)
    # z.mean_rate()
    mean_rate(z)
    with shelve.open('/Users/englitz/GoogleDrive/Code/python/Fishualizer/Data/Data20140827_spont/SampleData_mean_rate') as db:
        print(list(db.keys()))
    # f(z, 0)
    # g(z, 0)
    # g(z, 0)
    # f(z, 1)
