import numpy as np
import scipy
from sklearn.linear_model import Lasso
from sklearn.cluster import KMeans
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
import h5py
import dask
import joblib
from dask_glm.estimators import LinearRegression
import time

#%% parameters
data_origin = 'misha'

#%% Load data
if data_origin == 'georges':
    data_name_import = 'Data_20180131Run03.h5' # h5 data file to import (calcium/coords/times)

    def load_data(): # load the SampleData, labels and names 
        F = h5py.File(data_name_import,'r')
        data_temp = {'values': np.transpose(F['Data']['Values'].value,(1,0)), 
                'coords': np.transpose(F['Data']['Coordinates'].value,(1,0)),
                'times': np.transpose(F['Data']['Times'].value,(1,0)),
                'labels': np.transpose(F['Data']['ZBrainAtlas_Labels'])}
        return data_temp
   
    data = load_data() # load into Data    
    
    SR = 3.2 # input manually
    length_data = len(data['values'])  
    
    # trim data based on neuropil labelling
    def load_labels():
        try:
            #labels = scipy.sparse.load_npz(labels_name_import)
            labels = scipy.sparse.csr_matrix(data['labels'])
            def find_neuropil():
                neuropil_neurons = np.zeros(length_data, dtype='bool')
                ZBrainAtlas_neuropil = [106, 128, 194, 195, 196, 197, 198, 267] # ZBrainAtlas region names containing 'neuropil'
                for iRegion in ZBrainAtlas_neuropil:
                    current_neurons = scipy.sparse.find(labels[:, iRegion])[0]
                    neuropil_neurons[current_neurons] = True
                print(f'{neuropil_neurons.sum()} neuropil `neurons` out of {length_data} in total')
                return neuropil_neurons
            neuropil_neurons = find_neuropil()
            return labels, neuropil_neurons
        except FileNotFoundError:
            print('Labels File could not be found')
    
    labels, neuropil_neurons = load_labels()
    data_use = xr.DataArray(data = data['values'][np.logical_not(neuropil_neurons), :],
                           coords={'neurons': np.where(np.logical_not(neuropil_neurons))[0],
                                   'times': np.squeeze(data['times'])},
                                   dims=['neurons', 'times'])
    data_use.set_index(np.arange(len(data_use)))
    
if data_origin == 'misha':
    data_name_import = 'SampleData_all.h5'
    def load_data():
        F = h5py.File(data_name_import,'r')
        data_temp = {'values': np.transpose(F['Data']['Values'].value,(1,0)), 
                'coords': np.transpose(F['Data']['Coordinates'].value,(1,0)),
                'times': np.transpose(F['Data']['Times'].value,(1,0))}
        return data_temp
   
    data = load_data() # load into Data    
    length_data = len(data['values'])  
    SR = 3.2 # input manually
    
    labels = scipy.sparse.load_npz('Labels.npz')
    spikes = scipy.sparse.load_npz('SpikeTrains.npz')
    total_spikes = spikes.sum(axis=1)
    
    # trim data based on spike activity
    def find_neg_calc(N = length_data):#% There can be a few weird calcium signals (negative values)
        if N == length_data:
            mincalc = data['values'][:,:].min(axis = 1)
            neg_ind = np.nonzero(mincalc < 0)# indices of neurons with negative minimum
            mincalc_sort = np.sort(mincalc) # sorted minima
            mincalc_argsort = np.argsort(mincalc) # argsort of minima
        return (neg_ind[0], mincalc_sort, mincalc_argsort) 
    neg_calc_traces = find_neg_calc() # get negative calcium traces

    neurons_use = list(set(np.where(total_spikes > 0)[0]) - set(neg_calc_traces[0]))
    data_use = xr.DataArray(data=data['values'][neurons_use, :], 
                            coords={'neurons': neurons_use, 'times': np.squeeze(data['times'])},
                            dims=['neurons', 'times'])
    data_use.set_index(np.arange(len(data_use)))
    spikes_use = spikes[neurons_use, :]

#%% Lasso regression
n_subset=5000

subset_neurons = np.random.choice(len(data_use), size=n_subset)
n_neurons_subset = np.shape(subset_neurons)[0]
data_use = data['values'][subset_neurons, :]
n_neurons_all = np.shape(data_use)[0]
data_use = data_use.transpose()
alpha_val = 0.002
def lasso_reg(n):
    mask = np.ones(n_neurons_all, dtype = 'bool')
    mask[n] = False
    model = Lasso(alpha=alpha_val)
    model.fit(X=data_use[:, mask], y=data_use[:, n])
    return model.score(X=data_use[:, mask], y=data_use[:, n])

def lasso_reg_daskml(n):
    mask = np.ones(n_neurons_all, dtype = 'bool')
    mask[n] = False
    model = LinearRegression(regularizer='l1')
    model.fit(X=data_use[:, mask], y=data_use[:, n])
    return model.score(X=data_use[:, mask], y=data_use[:, n])#.compute()

#%% Performance testing # I put my times (with %time) of n_subset=5000 run.
######### sequential
start = time.time()
#%time test_sequential = np.array([lasso_reg(iLoop) for iLoop, _ in enumerate(subset_neurons)])
test_sequential = np.array([lasso_reg(iLoop) for iLoop, _ in enumerate(subset_neurons)])
end = time.time()
print(f'time elapsed sequential {end - start}')
#CPU times: user 1h 16min 23s, sys: 12min 2s, total: 1h 28min 25s
#Wall time: 24min 7s
# Comment T; The process is limited to one 4-CPU process

##### joblib
start = time.time()
#%time test_par_joblib = joblib.Parallel(n_jobs=2)(joblib.delayed(lasso_reg)(iLoop) for iLoop, iNeuron in enumerate(subset_neurons))
test_par_joblib = joblib.Parallel(n_jobs=2)(joblib.delayed(lasso_reg)(iLoop) for iLoop, iNeuron in enumerate(subset_neurons))
end = time.time()
print(f'time elapsed joblib {end - start}')
#CPU times: user 2.1 s, sys: 316 ms, total: 2.41 s
#Wall time: 14min 49s
# Comment T: This function runs n_jobs=2 processes of ~4 CPUs each (no speed-up for more n_jobs for a few other runs I tried)

###### dask
start = time.time()
vals = [dask.delayed(lasso_reg)(iLoop) for iLoop, _ in enumerate(subset_neurons)]
#%time test_par_dask = dask.compute(vals)
test_par_dask = dask.compute(vals)
end = time.time()
print(f'time elapsed dask {end - start}')
#CPU times: user 55min 11s, sys: 51min 44s, total: 1h 46min 55s
#Wall time: 13min 39s
# Comment T: This function runs 1 process of ~7.5 CPUs

assert (test_sequential == test_par_joblib).all()
assert (test_sequential == test_par_dask).all()


start = time.time()
vals = [dask.delayed(lasso_reg_daskml)(iLoop) for iLoop, _ in enumerate(subset_neurons)]
#%time test_par_dask = dask.compute(vals)
%time test_par_daskml = dask.compute(vals)
end = time.time()
## aborted after 2hours for n_subset=5000












