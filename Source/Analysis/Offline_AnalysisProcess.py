#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 12 11:34:51 2018

@author: ljp
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 12 16:22:52 2018

@author: thijs
"""

"""
Pipeline to deconvolve new datasets and get their statistics
Essentially a merge of Deconvolution.py and CorrelationDistance.py
Without labels..! So ROI definition is not needed!
"""


#%% Import
import numpy as np
import numpy.random as rnd
import matplotlib.pyplot as plt
from sys import path
try:
    path.append('/home/thijs/anaconda3/pkgs/OASIS-master')
    #path.append('C:/Users/Thijs/repos/fishualizer/data')
    
    """ For OASIS packages download & demo, see 
    https://github.com/j-friedrich/OASIS
    """
    from oasis import oasisAR1, oasisAR2
    from oasis.functions import deconvolve, estimate_parameters
except ModuleNotFoundError:
    pass

import h5py
import scipy
from scipy import sparse
from scipy import signal
from scipy import ndimage
import seaborn as sns; sns.set()
import matplotlib as mpl
import sklearn
from sklearn.cluster import KMeans
from sklearn.linear_model import Ridge, RidgeCV, Lasso, BayesianRidge, LinearRegression
from sklearn.decomposition import PCA
import pandas as pd
from datetime import datetime
import pyamg
import dask
import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout # to compute graph node positions

#%% parameters
data_name_import = 'Data_20180131Run03.h5' # h5 data file to import (calcium/coords/times)
#data_name_import = 'SampleData_all.h5'
labels_name_import = 'Labels_transf_direct2.npz'
data_type = 'Georges' # Misha or Georges

min_spikes_for_analysis = 1 # you cannot take all neurons into account due to size 
                            # of correlation matrix etc. (insufficient RAM), so
                            # I now select by a minimal threshold on the # of spikes

deconvolve_calcium = False # False: do not deconvolve (you can load pre-deconvovled spikes), True: deconvolve
spike_train_export_name = 'SpikeTrains_spont05'# if deconvolve_calium == True, spiketrains are saved with this name.
spike_train_import_name = 'SpikeTrains.npz' # if deconvolve_calcium == False, spiketrains are loaded with this name

save_df = False
data_name_export = 'df_statistics_20180131Run03_ref.h5' # If save_df == 1,  all single neurons statistics are saved in this (.h5) name

#%% Load data
def load_data(): # load the SampleData, labels and names 
    F = h5py.File(data_name_import,'r')
    data = {'values': np.transpose(F['Data']['Values'].value,(1,0)), 
            'coords': np.transpose(F['Data']['Coordinates'].value,(1,0)),
            'times': np.transpose(F['Data']['Times'].value,(1,0))} 
    return data
   
Data = load_data() # load into Data    

SR = 3.2 # input manually
LengthData = len(Data['values'])  

def load_labels():
    try:
        labels = scipy.sparse.load_npz(labels_name_import)
        def find_neuropil():
            neuropil_neurons = np.zeros(LengthData, dtype='bool')
            ZBrainAtlas_neuropil = [106, 128, 194, 195, 196, 197, 198, 267] # ZBrainAtlas region names containing 'neuropil'
            for iRegion in ZBrainAtlas_neuropil:
                current_neurons = scipy.sparse.find(labels[:, iRegion])[0]
                neuropil_neurons[current_neurons] = True
            print(f'{neuropil_neurons.sum()} neuropil `neurons` out of {LengthData} in total')
            return neuropil_neurons
        neuropil_neurons = find_neuropil()
        return labels, neuropil_neurons
    except FileNotFoundError:
        print('Labels File could not be found')

labels, neuropil_neurons = load_labels()

#%% Deconvolving
if deconvolve_calcium == True:
    def fast_deconvolv(n):
        """
        Deconvolve one neuron (param n) with OASIS
        Caveat: s_min free parameter!
        Returns deconvolved spike train only
        """
        y = Data['values'][n,:] # get calcium trace
        sig_deconv = deconvolve(y, penalty=0, optimize_g = 5) # deconvolve, optimize gamma
        c, s = oasisAR1(y - sig_deconv[2], sig_deconv[3], s_min = 0.3) # deconvolve with warm start,
        return s
    
    def save_spiketrains(neurons):
        """
        Return spike trains of neurons (param, list)
        """
        N = len(neurons) # number of neurons
        T = int(max(Data['times'][0])) # max time
        spike_trains = sparse.lil_matrix((N,T)) # creat empty matrix
        for n in range(N):
            if n % 1000 == 0:
                print(f'Progress: {n} of {N}')
            spike_trains[n,:] = fast_deconvolv(neurons[n])
    #    sparse.vstack([fast_deconvolv(n) for n in range(spike_trains.shape[0])]) # I don't get this to work, but would probably be better memory-wise
        return spike_trains
    
    #% Fetch spiketrains:
    ind = np.arange(LengthData) 
    sparse_spiketrains = save_spiketrains(ind)
    sparse_spiketrains = scipy.sparse.csr_matrix(sparse_spiketrains) # put in sparse format 
    scipy.sparse.save_npz(spike_train_export_name, sparse_spiketrains)
elif deconvolve_calcium == False:
    sparse_spiketrains = scipy.sparse.load_npz(spike_train_import_name) # load already deconvolved trains
    
def find_neg_calc(N = LengthData):#% There can be a few weird calcium signals (negative values)
    if N == LengthData:
        mincalc = Data['values'][:,:].min(axis = 1)
        neg_ind = np.nonzero(mincalc < 0)# indices of neurons with negative minimum
        mincalc_sort = np.sort(mincalc) # sorted minima
        mincalc_argsort = np.argsort(mincalc) # argsort of minima
    return (neg_ind[0], mincalc_sort, mincalc_argsort) 
neg_calc_traces = find_neg_calc() # get negative calcium traces
    
#%% Spike trains
if data_type == 'Misha':
    CumSpikeValuePerNeuron = sum(sparse_spiketrains.T).A # Cumulative Values of spikes per neuron; transpose for correct sum, set to array
    CumSpikeValuePerNeuron = np.squeeze(CumSpikeValuePerNeuron) 
    SpikesPerNeuron = np.zeros(len(CumSpikeValuePerNeuron))  # number of spike events (independent of spike height)
    temp = np.bincount(scipy.sparse.find(sparse_spiketrains)[0]) # via temp to include last neurons without spikes in SpikesPerNeuron
    SpikesPerNeuron[0:(len(temp))] = temp 
    FiringRate = CumSpikeValuePerNeuron * SR / (0.3 * 3550)
    del temp

elif data_type == 'Georges':
    CumSpikeValuePerNeuron = np.sum(Data['values'], axis=1)
    SpikesPerNeuron = np.isclose(Data['values'], 0, atol=0.001) # awkward for these data 
    SpikesPerNeuron = np.logical_not(SpikesPerNeuron)
    SpikesPerNeuron = np.sum(SpikesPerNeuron, axis=1)
    FiringRate = CumSpikeValuePerNeuron
    
#%% Functions
def comp_dist(coords): # pairwise distance between neurons
    dist = scipy.spatial.distance.pdist(coords, metric = 'euclidean')
    return scipy.spatial.distance.squareform(dist)   # this (going back to matrix representation) is a bit slow, but way more intuitive..  
        
def comp_corr(values): # compute the pairwise correlation coefficient
    corr_mat = np.corrcoef(values)
    print("Correlation matrix computed")
    return corr_mat

def select_neurons(threshold = 1): # find neurons where number of spikes > threshold 
    bool_thresh = (SpikesPerNeuron >= threshold) ## lines can be merged?
    ind_thresh = np.where(bool_thresh)[0] # get their indices
    return ind_thresh

def get_train(n): # convert one sparse spiketrain to array
    t = scipy.sparse.csr_matrix.todense(sparse_spiketrains[n,:])
    return np.array(t)[0]

#%%% This section defines functions to get statistics per neuron, saved in a pd.df
def get_statistics(thresh = 5, save_corr = False):
    """
    Get some basic statistics in pandas dataframe (df) of neurons with number of spikes > thresh
    In later functions, this df is used to compute additional statistics, which
    are subsequently added to the df.
    The indexing of the df is regular (i.e. [0,1,2,3,....]), and is only changed
    upon exporting to match the originals neuron index (i.e. [n0, n1, n2 ...]).
    s_min hard-coded (again!) for Firing Rate computation
    Params:
        thresh: float/int, minimal number of spikes
        save_corr: bool, whether to also return the correlation matrix 
    Returns:
        df_new, dataframe
        CorrMat, correlation matrix (optional)
    """
    s_min = 0.3
    NOI = select_neurons(threshold = thresh)
    N_neurons = len(NOI) # number of neurons
    print("Number of neurons: {}".format(N_neurons))
    df_new = pd.DataFrame({'neuron':  NOI}) # create df with neuron indices
    CorrMat = comp_corr(Data['values'][list(df_new['neuron']), :]) # get correlation matrix
    df_new['#SpikeEvents'] = SpikesPerNeuron[df_new['neuron']]
    df_new['Cumulative_SpikesValue'] = CumSpikeValuePerNeuron[df_new['neuron']]
    df_new['FiringRate'] = df_new['Cumulative_SpikesValue'] * SR / (len(Data['times'][0] * s_min)) # CumSpikes used 
    get_corr_ranges(df_new, CorrMat) # bin correlation counts per neuron
    if save_corr == True:
        return (df_new, CorrMat) # exporting corrmat is bad for memory
    elif save_corr == False:
        del CorrMat
        return df_new # only export df
    
"""
In the functions below I often assumend one is working with a df already    
So I have often hard-coded the names df_neur, C_neur and D_neur (which call the functions)
"""

def get_corr_ranges(df, CorrMat): 
    """
    Bin the correlation counts in intervals.
    Interval is now fixed at 0.1
    Params:
        df, CorrMat
    Returns (adds to df):
        Number of pairwise correlation counts per neuron within intervals of 0.1
    """
    ## Negative:
    for c in range(10):
        c_name = int(9 - c)
        name = 'global_#corr_leq_m0{}'.format(c_name)
        thresh = -0.9 + (c / 10)
        df[name] = sum(CorrMat < thresh)
    
    ## Positive:   
    for c in range(10):
        name = 'global_#corr_geq_0{}'.format(c)
        thresh = 0 + (c / 10)
        df[name] = sum(CorrMat > thresh) - 1 # -1 to correct for autocorrelation (=1)
           
def get_coord(df, coords): ## add coordinates to df
    df['x_coord'] = coords[df['neuron'], 0]
    df['y_coord'] = coords[df['neuron'], 1]
    df['z_coord'] = coords[df['neuron'], 2]
    
def total_corr_counts(df): #  count all correlation counts
    df['#totalcorrcounts'] = df.iloc[:,3:23].sum(axis = 1) # hard coded indices! # TODO: delete function?
    
def median_dist_for_corr_th(thresh = 0.6): 
    """
    Get median pairwise distance of (pairwise corrlation correlations > threshold)
    Params:
        thresh: float, correlation threshold
    Returns (adds to df):
        median distance to other neurons with corr >th
        median distance [..] > th, normalized by mean distance to ALL neurons (per neuron)
        TODO: Why not median or so the normalize? See email discussion with B
    """
    N = len(df_neur['neuron'])
    median_save = np.zeros(N)
    ind =  np.where(C_neur >= thresh) # find above thresh values
    dist = D_neur[ind] # find corresponding distancesn
    av_dist_all = np.mean(D_neur, axis = 0) # take mean pwd per neuron for norm.
    change_of_neuron = np.where(np.diff(ind[0]))[0] # get diff indces
    change_of_neuron = np.append(change_of_neuron, len(dist)) # add final bin
    for n in range(N-1):
        median_save[n+1] = np.median(dist[change_of_neuron[n]:change_of_neuron[n+1]])
    if change_of_neuron[0] != 0: # first one cannot be done, so do here manually
        median_save[0] = np.median(dist[0:change_of_neuron[0]])
    median_name = 'median_dist_corrgeq{}'.format(thresh)
    median_name_norm = 'median_dist_corrgeq{}_norm'.format(thresh)
    df_neur[median_name]= median_save
    df_neur[median_name_norm] = median_save / av_dist_all # normalized result!

def get_corr_vicinity(df, CorrMat, DistMat, r_th = 0.02, c_th = 0.4):
    """
    Add local correlation statistics per neuron to df 
    Consider all pairwise connections within radius r_th (maybe extend to kNN neurons or so?)
    Params:
        df, CorrMat, DistMat
        r_th: float, maximum radius to define locality
        c_th: float, correlation threshold to use for counting number of correlations >th in vicinity
    Returns (adds to df):
        Mean correlation in vicinity
        Number of Nearby Neurons: number of neurons within radius threshold (i.e. local density)
        Summed correlation in vicinity (I wanted to use this to check if negative correlations have a large effect)
            + normalize by Number of Nearby Neurons
        Number of Correlations in vicinity with correlation > c_th
            + normalized by Number of Nearby Neurons
    NB: I interchangeably use definitions vicinity, locality, neighbourhood (etc.?)
    """
    ## Currently by radius threshold: (also possible: k nearest neurons would be an approach, argsort matrix?)
    neurons_vic = np.where(DistMat < r_th) # find neuron pairs below thresh
    corr_vic = CorrMat[neurons_vic] # Get pairwise corr per relevant pair
    N = len(df) # number of neurons
    MeanCorrVic = np.zeros(N) # to save mean corr per neuron within vicinity
    CorrGeqc_thVic = np.zeros(N) # to save count of neurons >= 0.4 corr
    SumCorrVic = np.zeros(N)
    NumberOfNearbyNeurons = np.zeros(N)
    for n in range(N):# loop through all neurons
        ind = np.where(neurons_vic[0] == n)  # find entries of neuron n
        currentcorr = corr_vic[ind] # find corresponding correlations
        MeanCorrVic[n] = np.mean(currentcorr) # get mean
        CorrGeqc_thVic[n] = len(np.where(currentcorr > c_th)[0]) # get count
        SumCorrVic[n] = sum(currentcorr)
        NumberOfNearbyNeurons[n] = len(currentcorr)
    
    ## Write back into df
    df['MeanCorr_Vicinity'] = MeanCorrVic 
    name_CorrGeqc_thVic = '#CorrGeq{}Vicinity'.format(c_th)
    df[name_CorrGeqc_thVic] = CorrGeqc_thVic
    df['SumCorr_Vicinity'] = SumCorrVic 
    df['NumberOfNearbyNeurons'] = NumberOfNearbyNeurons 
    name_CorrGeqc_thVic_norm = '#CorrGeq{}Vicinity_norm'.format(c_th)
    df[name_CorrGeqc_thVic_norm] = df[name_CorrGeqc_thVic] / df['NumberOfNearbyNeurons'] 
    df['SumCorr_Vicinity_norm'] = df['SumCorr_Vicinity'] / df['NumberOfNearbyNeurons']
    
## Execute:
execute_df_statistics = False
if execute_df_statistics == True:
    """
    Get df statistics, CorrMat, DistMat
    These names for these three are sometimes hardcoded elsewhere in the code
    If memory allows, you can also get the correlation and distance matrix:
    """
    (df_neur, C_neur) = get_statistics(thresh =  min_spikes_for_analysis, save_corr = 1) # get dataframe and correlation matrix
    D_neur = comp_dist(Data['coords'][df_neur['neuron'], :]) # get distance matrix
    total_corr_counts(df_neur)
    get_coord(df_neur, Data['coords']) 
    median_dist_for_corr_th(thresh = 0.6)   
    get_corr_vicinity(df_neur, C_neur, D_neur) 
    df_neur['neuropil'] = neuropil_neurons[df_neur['neurons']].astype(float)
    
#%% Sparseness measures
def get_windowed_spikes(df = None, NOI = None, sigma = 30, filter_type = 'Gaussian'): 
    """
    Window spikes from sparse_spiketrains
    Either uniform or Gaussian, meaning of sigma changes with these (remains lead param)
    Params:
        df
        sigma; float/int, uniform: window size, gaussian: std size
        filter_type: 'Gaussian' or 'Uniform'
    Returns:
        filtered signal
    """
    if NOI == None: #inherently assuming df is not None
        NOI = list(df['neuron']) 
    spikes = scipy.sparse.csr_matrix.todense(sparse_spiketrains[NOI,:]) # fetch spikes intof ull matrix
    if filter_type == 'Gaussian':
        filtered_sig = scipy.ndimage.gaussian_filter1d(spikes, sigma, axis = 1) # row wise filtering
    elif filter_type == 'Uniform':
        filtered_sig = scipy.ndimage.uniform_filter1d(spikes, sigma, axis = 1)
    return filtered_sig

def get_sparseness(spikes): 
    """
    Compute lifetime sparseness (LTS), definition from Willlmore et al. 2011
    Params:
        spikes, full matrix (!!) 
    Returns:
        LTS, lifetime sparseness 
    """
    Time = np.shape(spikes)[1]
    E_r = sum(spikes.T) / Time # expectation value / normalized mean
    sq_E_r = E_r ** 2 # squared expect. value
    
    sp_trains_sq = np.power(spikes, 2) # square spike values element wise
    sq_sums = np.squeeze(sum(sp_trains_sq.T)) # sum squared values
    E_r_sq = sq_sums / Time # normalize
    LTS = 1 - (sq_E_r / E_r_sq) # calculate Lifetime sparseness
    return LTS 

def get_sparseness_newdef(spikes, wind, thresh = 5): 
    """
    New metric cooked up by B & T
    The goal was to look for sporadically spiking neurons. It worked ok on simulated data
    but in practice the method was not really showing this..
    Params:
        spikes: full matrix (I presume)
        wind: int, window size
        thresh: minimum number of spikes to detect within window
    Returns:
        sparse_new: new metric
        sparse_new_LTS: new metric multiplied with LTS. Results on simulated data were much improved this way
        LTS
    """
    th_wind = thresh / wind # divided because spikes are averaged/spread out in window
    Time = np.shape(spikes)[1]
    N = np.shape(spikes)[0]
    sparse_new = np.zeros(N)
    for n in range(N): # loop through neurons
        sparse_new[n] = len(np.where(np.squeeze(spikes[n,:]) <= th_wind)[0]) 
        if sparse_new[n] == Time:
            sparse_new[n] = 0
    sparse_new = sparse_new / Time
    sparse_new_LTS = ( sparse_new * get_sparseness(spikes) ) / Time
    LTS = get_sparseness(spikes)
    return (sparse_new, sparse_new_LTS, LTS)

run_sparseness = False
if run_sparseness == True:
    """
    Filter spikes, calculate sparseness and add this to df
    Returns (adds to df):
        three defs of sparseness as in return of get_sparsenss_newdef()
    """
    filt_spikes = get_windowed_spikes(df = df_neur, sigma = 50, filter_type = 'Uniform')
    Sparseness = get_sparseness_newdef(filt_spikes, wind = 50, thresh = 2)
    #plt.hist(Sparseness[1], 50)
    df_neur['sparse_newdef'] = Sparseness[0]
    df_neur['sparse_newdef_multipliedw_LTS'] = Sparseness[1]
    df_neur['sparse_LTS'] = Sparseness[2]

#%%%% Export to fishualize
def export_df(df, ename=data_name_export):
    """
    Store df in hdf5 format (from pd module)
    """
    df = df.set_index('neuron') # set index to df['neuron]'(for compatibility with the fishualizer)
    store_df = pd.HDFStore(ename)
    store_df['df_neur'] = df
    store_df.close()
    
save_df=False
if save_df == True: # value set in beginning of script
     export_df(df_neur)


#%% Ridge regression (version 2)
#neurons = [24237, 24243, 27835, 81506, 84485, 87001, 87140] 
neurons = [ 4522,  9080,  9438, 50335, 50369, 54209, 57854, 59844, 61095,
   64476, 70144, 79961, 81173]  # some "good" example neurons 
def plot_single_traces(neurons = [0], plot_legend = False): # plot calcium traces of some list of neuron indices
    plt.close(1)
    plt.figure(1)
    for n in range(len(neurons)):   
        plt.plot(Data['values'][neurons[n], :])
        plt.hold(True)
    if plot_legend == True:
        plt.legend([n for n in neurons])
        plt.hold(False)
        
def calculate_FVU(y_true, y_pred): 
    """ 
    R^2 = 1 - FVU
    R^2 metric is embedded in sklearn regressions. I keep this in for now 
    in case it might be useful for some other regression function (e.g. Bayesian
    Lasso)
    """
    MSE = ((y_true - y_pred) ** 2).sum() # definition from sklearn, except normalization
    VAR = ((y_true - y_true.mean()) ** 2).sum()
    FVU = MSE / VAR 
    return FVU   

def regress_few(neurons, alpha_lasso = None, verbose = True):
    """
    Lasso and BayesianRidge regression in an all vs all fashion
    Params:
        neurons: list of neuron indices to use
        alpha_lasso: alpha value (i.e. langrange multiplier for L1 norm)
    Returns:
        All regression coefficients (2 NxN matrices)
        R^2 scores computed based on all times fit and all times score (same data) 
    """
    data = Data['values'][neurons, :]
    N_neurons = len(neurons)        
    data = data.T 
    regr_coef = {'bayridge': np.zeros([N_neurons, N_neurons]), 
                 'lasso': np.zeros([N_neurons, N_neurons])} # save coef
    R2 = {'bayridge': np.zeros(N_neurons),
           'lasso': np.zeros(N_neurons)} # save R^2
    for n in range(N_neurons):
        if (n % 30 == 0) and (verbose == True):
            print(f'Progress: {n/N_neurons}')
        mask = np.ones(N_neurons, dtype = bool)
        mask[n] = 0
        data_use = data[:, mask] # exclude self regression
        # Bayesian Ridge
        bayridge_regr = BayesianRidge()
        bayridge_regr.fit(data_use, data[:, n])
        regr_coef['bayridge'][n, mask] = bayridge_regr.coef_
        R2['bayridge'][n] = bayridge_regr.score(data_use, data[:,n])
        # Lasso
        if alpha_lasso == None:
            alpha_lasso = 0.005 # this is usually not very good I believe..
        lasso_regr = Lasso(alpha = alpha_lasso) # use previous alpha value?
        lasso_regr.fit(data_use, data[:, n])
        regr_coef['lasso'][n, mask] = lasso_regr.coef_
        R2['lasso'][n] = lasso_regr.score(data_use, data[:, n])
        
    return (regr_coef, R2)

def regress_predict(neurons, neuron_n = 0, use_mask = True, plot_result = True, T_train = 1500): # predict test data from training data regression
    """
    Bayesian Regression only.
    Train model with times until T_train, test with both training and testing.
    Plot result of true and fitted calcium traces
    Params:
        neurons: list of neuron indices to use as regressors
        neuron_n: single int index to indicate what neuron to regress
        use_mask: bool, whether to include mask to avoid autoregression
        plot_result: bool
        T_train: int times in training set.
    Returns:
        Regression coefficients
        Train scores
        Test scores
    """
    data = Data['values'][neurons,:]
    N_neurons = len(neurons)
    if use_mask == True:
        mask = np.ones(N_neurons, dtype = 'bool')
        mask[neuron_n] = 0
    elif use_mask == False:
        mask = np.ones(N_neurons)
        
    T_total = np.shape(data)[1] # time
    reg = BayesianRidge() # Only Ridge at the moment:
    reg.fit(data[mask, 0:T_train].transpose(), data[neuron_n, 0:T_train].transpose()) # train with training data
    R2_train = reg.score(data[mask, 0:T_train].transpose(), data[neuron_n, 0:T_train].transpose()) # R^2 of training
    R2_test = reg.score(data[mask, T_train::].transpose(), data[neuron_n, T_train:T_total + 1].transpose()) # R^2 of test
    y_est = reg.predict(data[mask, T_train:-1].transpose()) # test on new data
    if plot_result == True:
        plt.close(4)
        plt.figure(4)
        plt.plot(np.arange(1,T_total+1), np.dot(reg.coef_, data[mask, :]) + reg.intercept_, 'b', 
                 np.arange(1, T_total + 1), data[neuron_n, :], 'r',
                 np.arange(T_train+1, T_total ), y_est, 'g') # to verify that reg.predict() = np.dot(coef, data)
        plt.legend(['regressed', 'true signal', 'predicted'])
    all_coef = np.zeros(N_neurons)
    all_coef[mask] = reg.coef_
    return (all_coef, R2_train, R2_test)

###### Regress based on correlation clusters
load_clusters = True
if load_clusters == True:
    """
    Load clusters as calculated (or in similar format as) Remi's clustering
    """
    correlation_clusters = np.load('localities_2018-02-22_14_32.npy')
    N_neurons_assigned = len(np.where(correlation_clusters != -1)[0])
    N_clusters = int(max(correlation_clusters) + 1) # + cluster_id = 0
    cluster_sizes = [len(np.where(correlation_clusters == iC)[0]) for iC in range(N_clusters)]

def compute_corr_regr(iCluster = 0, a_las = 0.005, verb = True):
    """
    Bayesian Ridge and Lasso regression all vs all on neurons of cluster iCluster
    params:
        iCluster: int, cluster index
        a_las: alpha value for lasso
    Returns:
        return of regress_few()
    """
    neurons_cluster = np.where(correlation_clusters == iCluster)[0]
    if cluster_sizes[iCluster] > 1: # some cluster sizes are 0 or 1
        regression_cluster = regress_few(neurons = neurons_cluster, alpha_lasso = a_las, verbose = verb) # manual alpha value..
        return regression_cluster
        
def compute_clusters_corr_regr(cluster_range = range(N_clusters), alpha_las = 0.005, 
                               max_cluster_size = 1000, verbose_incluster = True):
    """
    Bayesian Ridge and Lasso regression for range of clusters
    params:
        cluster_range: range of cluster indices
        alpha_las: alpha value for lasso
        max_cluster_size: possible maximal number of elements in cluster (for comp. feas.)
    Returns:
        Regression coefs for all clusters
        Mean scores per cluster
    """                                                                                  
    methods = ['bayridge', 'lasso']
    quality_of_prediction_neuron = {iM: np.zeros(len(Data['values'])) for iM in methods} # save R^2 metric
    quality_of_prediction_cluster = {iM: np.zeros(len(cluster_range)) for iM in methods}
    regression_coef = {iM: {} for iM in methods} # save coefs
    for iC in cluster_range:
        print(f'Progress: cluster_id: {iC}, cluster size = {cluster_sizes[iC]}')
        if cluster_sizes[iC] > 1 and cluster_sizes[iC] < max_cluster_size:
            neurons_cluster = np.where(correlation_clusters == iC)[0]
            regression_cluster = compute_corr_regr(iCluster = iC, a_las = alpha_las, verb = verbose_incluster)
            for iM in methods: # saving
                quality_of_prediction_neuron[str(iM)][neurons_cluster] = regression_cluster[1][str(iM)]
                quality_of_prediction_cluster[str(iM)][iC] = np.mean(regression_cluster[1][str(iM)])
                regression_coef[str(iM)][iC] = regression_cluster[0][str(iM)]
    return(regression_coef, quality_of_prediction_neuron, quality_of_prediction_cluster)    
    
def plot_regr_coef(regression_cluster, sort_by_qop = False):
    """
    Plot heatmap of regression coefficients
    Params:
        regression_cluster: Result of regression in format of regress_few() return
        sort_by_qop: bool, whether to sort heatmap by score TODO: neatly change ouput for true of false respectively
    Returns:
        plot only
    """
    if sort_by_qop == True:
        ind_sort_lasso = np.argsort(regression_cluster[1]['lasso']) 
        reg_mat_lasso = regression_cluster[0]['lasso']
        reg_mat_sorted_lasso = reg_mat_lasso[[x for x in ind_sort_lasso], :] 
        qop_norm = regression_cluster[1]['lasso']
        qop_sorted = regression_cluster[1]['lasso'][[x for x in ind_sort_lasso]]
        print(f'R2 lasso normal: {np.round(qop_norm, 3)}')
        print(f'R2 lasso sorted: {np.round(qop_sorted,3 )}')
        print([x for x in ind_sort_lasso])
        ## Is this right?nbh
            
        ind_sort_bayridge = np.argsort(regression_cluster[1]['bayridge'])
        reg_mat_bayridge = regression_cluster[0]['bayridge']
        reg_mat_sorted_bayridge = reg_mat_bayridge[[x for x in ind_sort_bayridge], :] 
        qop_norm = regression_cluster[1]['bayridge']
        qop_sorted = regression_cluster[1]['bayridge'][[x for x in ind_sort_bayridge]]
        print(f'R2 ridge normal: {np.round(qop_norm, 3)}')
        print(f'R2 ridge sorted: {np.round(qop_sorted, 3)}')
        print([x for x in ind_sort_bayridge])
    plt.close(1)
    plt.figure(1) # plot heatmap for 1 cluster
    plt.subplot(2,1,1)
    sns.heatmap(regression_cluster[0]['lasso'], cmap = "BuGn_r")
    plt.title('Lasso Coefficients')
    plt.subplot(2,1,2)
    if sort_by_qop == True:
        sns.heatmap(reg_mat_sorted_lasso, cmap = "BuGn_r")
    plt.title('Lasso Coefficients sorted')
    plt.close(2)
    plt.figure(2)
#    
    plt.subplot(2,1,1)
    sns.heatmap(regression_cluster[0]['bayridge'], cmap = "BuGn_r")
    plt.title('Bayesian Ridge Coefficients')
    plt.subplot(2,1,2)
    if sort_by_qop == True:
        sns.heatmap(reg_mat_sorted_bayridge, cmap = "BuGn_r")
    plt.title('Bayesian Ridge sorted Coefficients')
        
def variance_cluster_neurons(iC): # calculate variance of a cluster
    if cluster_sizes[iC] > 0:
        neurons_cluster = np.where(correlation_clusters == iC)[0]
        variance_cluster = np.var(Data['values'][neurons_cluster, :], axis = 1)
        return variance_cluster

def plot_QoP_histogram(regression_cluster):
    """
    Plot histogram of scores
    Params:
        regression_cluster: result of regression in format of regress_few()
    Returns:
        plot only
    """
    quality_of_prediction = regression_cluster[1]
    plt.figure(5)
    plt.subplot(2,1,1)
    plt.title('Histogram R^2 values Bayesian Ridge')
    plt.hist(quality_of_prediction['bayridge'][quality_of_prediction['bayridge'] > 0])
    plt.subplot(2,1,2)
    plt.title('Histogram R^2 values Lasso')
    plt.hist(quality_of_prediction['lasso'][quality_of_prediction['lasso'] > 0])

def save_QoP(quality_of_prediction): # get quality of prediction from compute_clusters_corr_regr()
    np.save('Rsquared_BayRidge.npy', quality_of_prediction['bayridge'])
    np.save('Rsquared_Lasso.npy', quality_of_prediction['lasso'])
    
def run_diff_bayridge_lasso():
    """
    Compare differences between Lasso and Bayesian Ridge regression
    TODO: ineffective because BayRidge is run multiple times?
    Run Lasso for a few different alpha values for all clusters, TODO: soft-code this range
    Plot BayRidge vs Lasso mean scores per cluster. Different colours denote 
    different alpha values.
    
    No Params
    Returns:
        n_lasso_comps_rel: relative number of lasso components per cluster regression (mean om std)
    """
    regr = {}
    n_lasso_comps = {} # save number of nonzero lasso coefs
    n_bayridge_comps = {}
    alphas = np.logspace(-3, -2, num=4) # alphas to use in lasso
    max_cluster_size_bound = 400
    n_used_clusters = N_clusters # how many clusters to evaluate
    for a_index, a in enumerate(alphas):
        print(f'alpha index = {a_index} / {len(alphas) - 1}')
        regr[a_index] = compute_clusters_corr_regr(cluster_range = range(n_used_clusters), 
            alpha_las=a, max_cluster_size = max_cluster_size_bound, verbose_incluster = False)
        n_lasso_comps[a_index] = {}
        for nn in range(n_used_clusters):  # save number of nonzero coefs
            if cluster_sizes[nn] > 1 and cluster_sizes[nn] < max_cluster_size_bound:
                n_lasso_comps[a_index][nn] = len(np.where(regr[a_index][0]['lasso'][nn])[0])
                if a_index == 0: # assuming bayridge always yields the same result
                    n_bayridge_comps[nn] = len(np.where(regr[a_index][0]['bayridge'][nn])[0])                   
    n_lasso_comps_rel = {} # fraction of lasso comps, will be filled in during plotting
    n_bayridge_comps_rel = (np.round(np.mean(([n_bayridge_comps[x] / cluster_sizes[x]**2 for x in n_bayridge_comps])), 3),
                         np.round(np.std(([n_bayridge_comps[x] / cluster_sizes[x]**2 for x in n_bayridge_comps])), 3))
        
    plt.close(6)
    plt.figure(6)  
#    plt.subplot(1,2,1)     
    plt.title('Explained variance, every dot is the dual mean of 1 cluster, \n'
              'All clusters means are calculated for every alpha value \n'
              f'Fraction Bayesian non zero coefs: {n_bayridge_comps_rel[0]} pm {n_bayridge_comps_rel[1]}')
    plt.xlabel('Lasso mean score per cluster')
    plt.ylabel('Bayesian Ridge mean score per cluster')
    plt.hold(True)
    for a_index, a in enumerate(alphas):
        # get mean and std of fraction of lasso componentns
        n_lasso_comps_rel[a_index] = (np.round(np.mean(([n_lasso_comps[a_index][x] / cluster_sizes[x]**2 for x in n_lasso_comps[a_index]])), 3),
                         np.round(np.std(([n_lasso_comps[a_index][x] / cluster_sizes[x]**2 for x in n_lasso_comps[a_index]])), 3))
        plt.plot(regr[a_index][2]['lasso'], regr[a_index][2]['bayridge'], '.') # plot all clusters for alpha value
        plt.hold(True)
    
    plt.legend([f'alpha: {np.round(alphas[a], 3)}, Fraction Lasso coefs>0: {n_lasso_comps_rel[a][0]} pm {n_lasso_comps_rel[a][1]}' for a in range(len(alphas))])
    plt.plot([0, 1], [0, 1], color='k', lw=0.5) # reference line
    plt.hold(False)
    plt.show()
#    plt.subplot(1,2,2)
#    info_str = 'Fraction of nonzero regression coefficients: \n Lasso (alpha: mean \pm std): \n {} \n BayesianRidge {}'.format(n_lasso_comps_rel, n_bayridge_comps_rel)
#    plt.text(0,.5, info_str)
    return (n_lasso_comps_rel, n_bayridge_comps_rel)

#%%
def fun_remi(split_train_test = True, plot_results = True, variance_cutoff = True,
             max_cluster_size = 400, var_th = 0.00001, T_train = 1750, n_clusters_use = N_clusters):
    """
    Replicate of function/experiment that Remi e-mailed me about with some adjustments.
    Per cluster, compute Bayesian Regresson with training data. 
    Then pick random neuron and start adding regressors one by one, sorted by coefficient magnitude
    TODO: add abs() for large negative values
    Regress again for every combination of regressors (TODO: only renormalize?)
    Create function score(number of regressors)
    If converged -> break    
    plot results
    Params:
        split_train_test: bool, whether to split the data set for score calculation
        plot_results; bool
        variance_cutoff: bool, whether to cut off converged scores with a variance threshold
        max_cluster_size; int, max size take into consideration
        var_th: float, threshold for variance cut-off (last 5 elements)
        T_train: int, times to use for training
        n_cluster_use: int, number of clusters to use    
    Returns:
        main_regressors: per cluster
        organized_mr: unique set of all MR
        neurons_all: all neurons used in this simulation
    """    
    R2_regressors = {c: [] for c in range(n_clusters_use)}
    main_regressors = {c: [] for c in range(n_clusters_use)} 
    neurons_all =[]
    for iC in range(n_clusters_use):
        print(f'Progress: cluster_id: {iC}, cluster size = {cluster_sizes[iC]}')
        if cluster_sizes[iC] > 1 and cluster_sizes[iC] < max_cluster_size:
            neurons_cluster = np.where(correlation_clusters == iC)[0] # get data
            for n_app, v in enumerate(neurons_cluster):
                neurons_all.append(v)
            data = Data['values'][neurons_cluster, :]
            N_neurons = len(neurons_cluster)        
            data = data.T 
            
            neuron_id = np.random.randint(0, N_neurons, 1) # pick random neuron or first or so
            mask = np.ones(N_neurons, dtype = bool)
            mask[neuron_id] = 0
            data_use = data[:, mask]
            reg = BayesianRidge()
            reg.fit(data_use, data[:, neuron_id]) # train model
            
            coef = np.zeros(len(neurons_cluster))
            coef[mask] = reg.coef_
            sorted_coef = np.argsort(coef) # sort neuron_id's coefs by small to large
            
            for n in range(len(neurons_cluster)): # auto (with coef =0) also taken into acocuntp
                try:
                    del reg_new # to make sure it is not overtrained, should be unnecessary (TODO: remove)
                except:
                    pass # for the first n =0
                ind_neurons_use = int(-1 - n) # get n largest components
                neurons_use = sorted_coef[ind_neurons_use::].tolist()
                mask_use = np.zeros(len(neurons_cluster), dtype = bool)
                mask_use[neurons_use] = 1 # only use n largest components
                mask_use[neuron_id] = 0 # set own contribution to zero
                
                reg_new = BayesianRidge() # regress to get new score
                if split_train_test == False:
                    reg_new.fit(data[:, mask_use], data[:, neuron_id])
                    current_score = reg_new.score(data[:, mask_use], data[:, neuron_id])
                elif split_train_test == True:
                    reg_new.fit(data[:T_train, mask_use], data[:T_train, neuron_id])
                    current_score = reg_new.score(data[T_train:, mask_use], data[T_train:, neuron_id])
                R2_regressors[iC].append(current_score) # append score (for this n)
                if n > 5 and variance_cutoff == True: # test for convergence over prev. 5
                    var_recent_score = np.var(R2_regressors[iC][-5::])
                    if var_recent_score < var_th: # if converged
                        print(f'converged, n = {n}')
                        main_regressors[iC] = neurons_cluster[neurons_use]
                        break
                
    if plot_results == True:
        plt.figure(1)
        plt.xlabel('Number of regressors')
        plt.ylabel('R2')
        plt.hold(True)
        for r in R2_regressors:
            plt.plot(R2_regressors[r])
            plt.hold(True)
            
    organized_mr = set() # organized main regressors, no doubles
    for n in range(n_clusters_use):
        organized_mr.update(main_regressors[n])        
    return (main_regressors, organized_mr, neurons_all)

def find_large_lasso_comps(regression_cluster, return_cumcount_th = 1):
    """
    Sum absolute values column wise (so per regressor)
    Plot result, and return all regressors > return_cumcount_th (param)
    """
    coef = regression_cluster[0]['lasso']
    val = np.sum(np.abs(coef), axis=0)
    plt.figure(2)
    plt.hist(val, int(np.ceil(max(val))))
    return [v > return_cumcount_th for v in val]
    
def lasso_1vsall(n, split_train_test = True): # TODO: improve or delete? Is this useful?
    """
    Compute Lasso regression for 1 neuron (param n) vs all others (94k)
    NB: essentially regress_predict() is the Bayesian Ridge variant. these could be combined?
    """
    mask = np.ones(LengthData, dtype = bool)
    mask[n] = 0
    T_train = 1750
    alpha_train = 0.005
    regr = Lasso(alpha = alpha_train)
    regr.fit(X = Data['values'][mask, :T_train].transpose(), y=Data['values'][n, :T_train].transpose())
    print(f'Number and fraction of nonzeros: {len(np.where(regr.coef_)[0])}, {len(np.where(regr.coef_)[0]) / (LengthData ** 2)}')
    R2_train = regr.score(X = Data['values'][mask, :T_train].transpose(), y=Data['values'][n, :T_train].transpose())
    R2_test = regr.score(X = Data['values'][mask, T_train:].transpose(), y=Data['values'][n, T_train:].transpose())
    print(f'R2 train {R2_train}, R2 test {R2_test}')
    
def complete_sparse_lasso(neurons = [0, 1], split_train_test = True, T_train = 2500, alpha_use = 0.005):
    """
    Compute all vs all Lasso regression. 
    Save coefs in sparse format (lil_matrix).
    Split train and test data
    
    Params:
        neurons: list of neurons to use
        split_train_test: bool
        T_train: int of times for training
        alpha_use: float for Lasso param
    Returns:
        fun_mat: sparse matrix of all Lasso coefs
        scores: dict of ['train'] and ['test'] scores
        neurons: list of neurons used
    """
    
    N_neurons = len(neurons)
    print(f'Lasso, Number of neurons: {N_neurons}')
    fun_mat = sparse.lil_matrix((N_neurons, N_neurons)) # lil is good for building? change to csr later perhaps
    scores = {'train': np.zeros(N_neurons), 'test': np.zeros(N_neurons)}
    mask = np.ones(N_neurons, dtype = bool) # change relevant neuron on each it    
    data_use = Data['values'][neurons, :].transpose()
    for n in range(N_neurons):
        if n % round(N_neurons / 100) == 0: # print progress
            print(f'Lasso Progress: {np.round(n / N_neurons, 4)}, time ={str(datetime.now())}')
        if n > 0:
            del regr # to be sure old model is not used, probably unnecessary
            mask[n-1] = 1 # reset mask of previous run to True
        mask[n] = 0 # set new mask
        temp_coef = np.zeros(N_neurons)
        regr = Lasso(alpha = alpha_use)
        regr.fit(X = data_use[:T_train, mask], y = data_use[:T_train, n])
        temp_coef[mask] = regr.coef_
        fun_mat[n, :] = temp_coef
        scores['train'][n] = regr.score(X = data_use[:T_train, mask], y = data_use[:T_train, n])
        scores['test'][n] = regr.score(X = data_use[T_train:, mask], y = data_use[T_train:, n])
    return(fun_mat, scores, neurons)
    
def complete_full_bayridge(neurons = [0, 1], split_train_test = True, T_train = 1750):
    """
    Bayesian Ridge all vs all, returns coefs in full matrix format.
    For full description see complete_sparse_lasso() above
    """
    N_neurons = len(neurons)
    print(f'Bayesian Ridge, Number of neurons: {N_neurons}')
    fun_mat = np.zeros([N_neurons, N_neurons]) # big full  mat
    scores = {'train': np.zeros(N_neurons), 'test': np.zeros(N_neurons)}
    mask = np.ones(N_neurons, dtype = bool) # change relevant neuron on each it    
    data_use = Data['values'][neurons, :].transpose()
    for n in range(N_neurons):
        if n % round(N_neurons / 100) == 0:
            print(f'Bayridge Progress: {np.round(n / N_neurons, 4)}')
        if n > 0:
            del regr
            mask[n-1] = 1
        mask[n] = 0 # set new mask
        temp_coef = np.zeros(N_neurons)
        regr = BayesianRidge()
        regr.fit(X = data_use[:T_train, mask], y = data_use[:T_train, n])
        temp_coef[mask] = regr.coef_
        fun_mat[n, :] = temp_coef
        scores['train'][n] = regr.score(X = data_use[:T_train, mask], y = data_use[:T_train, n])
        scores['test'][n] = regr.score(X = data_use[T_train:, mask], y = data_use[T_train:, n])
    return(fun_mat, scores, neurons)
    
def bayridge_few_many(n_regressors = [], n_regressands = [], T_train = 1750 ):
    """
    Bayesian Regress different sets of regressands and regressors.
    Splitting of training and testing is hard-coded
    Params:
        n_regressors: list of indices of regressors (i.e. independent variables)
        n_regressands: list of indices of regressands (i.e. dep. var.)
        T_train: int of times to use for training
    Returns:
        fun_mat: all regression coefs
        scores: dict of ['train'] and ['test'] scores        
    """ 
    N_regressors= len(n_regressors)
    N_regressands = len(n_regressands)
    fun_mat = np.zeros([N_regressands, N_regressors]) # big full  mat
    scores = {'train': np.zeros(N_regressands), 'test': np.zeros(N_regressands)}
    mask = np.ones(N_regressors, dtype = bool) # change relevant neuron on each it    
    data_regressors = Data['values'][n_regressors, :].transpose()
    data_regressands = Data['values'][n_regressands, :].transpose()
    for n in range(N_regressands):
        if n > 0:
            del regr
            mask = np.ones(N_regressors, dtype = bool) # change relevant neuron on each it    
        if n in n_regressors:
            mask[n] = 0
        temp_coef = np.zeros(N_regressors)
        regr = BayesianRidge()
        regr.fit(X = data_regressors[:T_train, mask], y = data_regressands[:T_train, n])
        temp_coef[mask] = regr.coef_
        fun_mat[n, :] = temp_coef
        scores['train'][n] = regr.score(X = data_regressors[:T_train, mask], y = data_regressands[:T_train, n])
        scores['test'][n] = regr.score(X = data_regressors[T_train:, mask], y = data_regressands[T_train:, n])
    return(fun_mat, scores)

run_remi_analysis = False
if run_remi_analysis == True:   
    """
    Compute fun_remi()
    Try to get smaller network with which to explain large initial set of neurons
    GOALS
    ## Make a network (schematic graph.?) to show the flow of top components?
        # run top comps against all others for explained variance (via just 30 neurons)
            # see figs, quite good for training. tough for testing.
    """
    regress_remi = fun_remi(variance_cutoff=False, split_train_test=True)
    new_network = regress_few(neurons=[n for n in regress_remi[1]]) # all vs all regression of Main Regressors from line above
    network_neurons = np.array([n for n in regress_remi[1]])
    plot_regr_coef(new_network)
    
    top_comps = find_large_lasso_comps(new_network) # found in a subjective way (> th) 
    top_network = regress_few(neurons=network_neurons[top_comps])
    plot_regr_coef(top_network, sort_by_qop=True)
    plot_QoP_histogram(top_network)
    plot_single_traces(network_neurons[top_comps])
    
#    test = bayridge_few_many(n_regressors = network_neurons[top_comps], n_regressands = [x for x in np.sort(regress_remi[2])])
    test2 = bayridge_few_many(n_regressors = network_neurons[top_comps], n_regressands = [x for x in network_neurons])
    plt.figure(3)
    plt.subplot(1,2,1)
    plt.title('train')
    plt.hist(test2[1]['train'])
    plt.subplot(1,2,2)
    plt.title('test')
    plt.hist(test2[1]['test'], 200)
    
      
def run_all_lasso_bayridge(save_coefsscores = True, some_lasso_analysis = True, 
                           run_bayridge = False):
    """
    Run the all vs all lasso for all spiking neurons (27k) excluding negative
    calcium traces (measurement artefacts? with big influence)
    Save results
    Do some analysis and plots
    TODO: implement bayesian ridge way of things
    TODO: Compare strong Lasso comps with Bayesian equivalents and vice versa
    TODO: Automate bayesian ridge progress to find sparse bayridge representation
            # check sparsity and then do same analysis as lasso and compare
        Note that Bayesian Ridge takes very long.. Maybe use Bayesian Ridge to improve
        on Lasso estimates with low R^2? Although this would be numerous as well.
        
    Params:
        save_coefsscores: bool, to save on computer
        some_lasso_analysis: bool, all analysis and plotting
        run_bayridge: bool, WIP
    Returns:
        TODO: should return many things (I currently run the code by section at the moment..)
    """
    neg_calc_traces = find_neg_calc() # find negative calcium traces
    spiking_neurons = set([x for x in np.where(SpikesPerNeuron > 0)[0]]) # set of all spiking neurons
    spiking_neurons_pos = list(spiking_neurons - set(neg_calc_traces[0])) # subtract set of negative neurons, make list
    use_neurons = list(np.sort(spiking_neurons_pos)) # sort for intuitive compatability
    
    lasso_reg = complete_sparse_lasso(neurons = use_neurons, T_train=2500, alpha_use=0.001) # calculate lasso
    CSR_FullLassoR = sparse.csr_matrix(lasso_reg[0]) # go from lil_mat to csr_mat (I read this is comp. more efficient)
    real_ind = np.array(lasso_reg[2]) # mapping between relative indices (of use_neurons) and absolute indices (of Data)
    if save_coefsscores == True:
        sparse.save_npz('sparse_lassoregr_allneurons', CSR_FullLassoR)
        np.save('lasso_scores_Ttrain2400_alpha0.005_all', lasso_reg[1])
        lasso_scores_all = {'train': np.zeros(LengthData), 'test': np.zeros(LengthData)} # put scores in zeros for all neurons
        lasso_scores_all['train'][lasso_reg[2]] = lasso_reg[1]['train']
        lasso_scores_all['test'][lasso_reg[2]] = lasso_reg[1]['test']
        np.save('lasso_scores_all_train', lasso_scores_all['train'])
#        np.save('lasso_scores_all_test', lasso_reg[1]['test'])
        np.save('lasso_scores_all_test', lasso_scores_all['test'])
        np.save('AbsoluteNeuronIndices', real_ind)
    
    load_previous_data = False
    if load_previous_data == True:
        lasso_reg = {}
        lasso_reg[0] = sparse.load_npz('sparse_lassoregr_allneurons.npz')
        lasso_reg[1] = np.load('lasso_scores_Ttrain2400_alpha0.005_all.npy')
        lasso_reg[1] = lasso_reg[1][()]
        lasso_reg[2] = np.load('AbsoluteNeuronIndices.npy')
        lasso_scores_all = {'train': [], 'test': []}
        lasso_scores_all['train'] = np.load('lasso_scores_all_train_0002.npy')
        lasso_scores_all['test'] = np.load('lasso_scores_all_test_0002.npy')
        real_ind = np.array(lasso_reg[2]) # mapping between relative indices (of use_neurons) and absolute indices (of Data)
   
        
    plt.close(19)
    plt.figure(19)
    prec = 0.0 # minimum coefficient size to plt.spy()
    plt.title(f'Lasso coefficients geq {prec}')
    plt.spy(test4, precision=prec, markersize=1) # spy plot to see nonzero coefs > precision
    plt.xlabel('Relative indices')
    plt.ylabel('Relative indices')
    plt.show()
    
    if some_lasso_analysis == True:
        dist_mat = comp_dist(Data['coords'][use_neurons, :]) # get pw distance matrix
        regressors_ind = sparse.find(lasso_reg[0]) # get indices of nonzero lasso coefs
        dist_mat_sparse = sparse.lil_matrix((len(spiking_neurons_pos), len(spiking_neurons_pos)))
        dist_mat_sparse[regressors_ind[0], regressors_ind[1]] = dist_mat[regressors_ind[0], regressors_ind[1]] # create sparse dist mat with entries only at coef locations
        del dist_mat # delete for memory
        distance_th = 0.15
        coef_th =.1
        dist_mat_th = dist_mat_sparse > distance_th # bool coef > threshold
        coef_mat_th = lasso_reg[0] > coef_th
        combined_mat_th = dist_mat_th.multiply(coef_mat_th) # combined bool 
        
        ind_dth = sparse.find(dist_mat_th)
        ind_cth = sparse.find(coef_mat_th)
        ind_combined = sparse.find(combined_mat_th)
        print(f'# > dth {len(ind_dth[0])},# > cth {len(ind_cth[0])},# combined {len(ind_combined[0])}')
        regressors_dth = dist_mat_sparse[ind_combined[0], ind_combined[1]].A
        regressors_dth = regressors_dth[0]
        regressors_cth = lasso_reg[0][ind_combined[0], ind_combined[1]].A
        regressors_cth = regressors_cth[0]
        plt.close(4)
        plt.figure(4)
        plt.hist2d(regressors_dth, regressors_cth, bins = 60)
        plt.title(f'Histogram of regressors > threshold (distance {distance_th}, coefficient {coef_th})')
        plt.xlabel('Distance')
        plt.ylabel('Coefficient magnitude (abs)')
        plt.show()

        plt.figure(5)
        sns.regplot(lasso_scores_all['train'], lasso_scores_all['test'])
        plt.xlabel('Training scores')
        plt.ylabel('Testing scores')
        plt.title('Lasso scores')
        
        regressor_magnitude = np.array(abs(lasso_reg[0]).sum(axis = 0))[0] # column sum ()
        regressand_magnitude = np.squeeze(np.array(abs(lasso_reg[0]).sum(axis = 1))) # row sum
        n_regressors_perneuron = np.squeeze(np.array((lasso_reg[0] != 0).sum(1))) # row sum
        n_regressands_perneuron = np.squeeze(np.array((lasso_reg[0] != 0).sum(0))) # column sum
        
        maximum_regressors_pn = sparse.csr_matrix.todense(sparse.csr_matrix(lasso_reg[0]).max(axis = 0)) # TODO: this is column wise correect?
        minimum_regressors_pn = sparse.csr_matrix.todense(sparse.csr_matrix(lasso_reg[0]).min(axis=0))
        maximum_regressors_pn = np.squeeze(np.array(maximum_regressors_pn))
        minimum_regressors_pn = np.squeeze(np.array(minimum_regressors_pn))
        maxabs_regressors_pn = np.array([max(maximum_regressors_pn[x], abs(minimum_regressors_pn[x])) for x in range(len(maximum_regressors_pn))])

        maximum_regressands_pn = sparse.csr_matrix.todense(sparse.csr_matrix(lasso_reg[0]).max(axis = 1)) # TODO: this is column wise correect?
        minimum_regressands_pn = sparse.csr_matrix.todense(sparse.csr_matrix(lasso_reg[0]).min(axis=1))
        maximum_regressands_pn = np.squeeze(np.array(maximum_regressands_pn))
        minimum_regressands_pn = np.squeeze(np.array(minimum_regressands_pn))
        maxabs_regressands_pn = np.array([max(maximum_regressands_pn[x], abs(minimum_regressands_pn[x])) for x in range(len(maximum_regressands_pn))])


        plt.figure(6)
        plt.subplot(1,2,1)
        sns.regplot(x = n_regressors_perneuron, y = lasso_reg[1]['train'], x_estimator = np.mean)# compare number of regressors with score
        plt.xlabel('# regressors per neuron (row-wise sum)')
        plt.ylabel('Training score')
        plt.subplot(1,2,2)
        sns.regplot(x = n_regressors_perneuron, y = lasso_reg[1]['test'], x_estimator = np.mean)# compare number of regressors with score
        plt.xlabel('# regressors per neuron (row-wise sum)')
        plt.ylabel('Test score')
    
        main_regressors = (n_regressands_perneuron > np.percentile(n_regressands_perneuron, 99.95))
        plot_single_traces(neurons= real_ind[np.where(main_regressors)[0]]) 
        
        plt.figure(10)
        sns.jointplot(x = n_regressands_perneuron, y = n_regressors_perneuron)
        plt.xlabel('n regressions per neuron')
        plt.ylabel('n regressors per neuron')
        
        plt.figure(13)
        sns.jointplot(x = maximum_regressors_pn, y = lasso_reg[1]['train'])
        plt.xlabel('maximum regressor value')
        plt.ylabel('training R2')
        
        sns.jointplot(x = maximum_regressors_pn, y = lasso_reg[1]['test'])
        plt.xlabel('maximum regressor value')
        plt.ylabel('test R2')
        
        plt.close(22)
        plt.figure(22)
        plt.scatter(FiringRate[lasso_reg[2]], lasso_scores_all['test'][lasso_reg[2]], s=0.3)
        plt.xlabel('Firing Rate')
        plt.ylabel('test score')
        plt.title('test')
        plt.show()
        
        plt.close(23)
        plt.figure(23)
        plt.scatter(FiringRate[lasso_reg[2]], 0.5*(lasso_scores_all['test'][lasso_reg[2]] + lasso_scores_all['train'][lasso_reg[2]]), s=0.3)
        plt.xlabel('Firing Rate')
        plt.ylabel('test score')
        plt.title('test + train /2')
        plt.show()
        
        # Negative (inhibition) connections:
        neg_reg_mat = -1*lasso_reg[0]
        neg_reg_mat_ind = sparse.find(neg_reg_mat > 0)
        neg_only_reg_mat = lasso_reg[0][neg_reg_mat_ind]
        pos_reg_mat = lasso_reg[0]
        plt.figure(24)
        plt.spy(neg_reg_mat, precision = 0, markersize=1)        
        
    if run_bayridge == True:
        """
        WIP
        """
        bayridge_reg = complete_full_bayridge(neurons = spiking_neurons_pos)
    #    bayridge_reg = complete_full_bayridge(neurons = [x for x in range(150)])
        if save_coefsscores == True:
            np.save('bayridge_coefs', bayridge_reg[0])
            np.save('bayridge_scores', bayridge_reg[1])
            
 
    save_stats_in_df = True
    if save_stats_in_df == True:
        """
        Put individual neurons statistics based on regression in the general df_neur
        """
        ## Create mask to remove negative neurons
        mask_df = np.ones(len(spiking_neurons), dtype = 'bool')
        for index, iNeg in enumerate(neg_calc_traces[0]): # loop through negative neurons
            index_df = np.where(df_neur['neuron'] == iNeg)[0]  # find corresponding df index
            mask_df[index_df] = False # set to false
        mask_df = mask_df.tolist()   
        df_neur['min_calc'] = np.min(Data['values'][df_neur['neuron'], :],axis=1) # add minima to df_neur
       
        ## Create new columns in masked df and add statistics
        def add_column(name = 'test', values = np.ones(len(spiking_neurons_pos))):
            """
            Add an array of length spiking_neurons_pos to df
            Params:
                name: str, column name
                values: array with values
            Returns:
                Adds values to df_neur with name
            """
            L_df = len(df_neur)
#            df_neur[name] = np.zeros(L_df) # set zero to others for fishualizing (rather than Nans)
    #        df_neur[name][mask_df] = values
    #        df_neur[mask_df][name] = values
#            df_neur[df_neur['min_calc'] > 0].loc[:, name] = values
            temp = np.zeros(L_df)
            temp[mask_df] = values
            df_neur[name] = temp
            

        add_column(name = 'regression_magnitude', values=regressand_magnitude)
        add_column(name='regressor_magnitude', values=regressor_magnitude)
        add_column(name='n_regressors_pn', values=n_regressors_perneuron)
        add_column(name='n_regressands_pn', values=n_regressands_perneuron)
              
        def dist_coef(distance_th = 0.1, coef_th = 0.1):
            """
            Find all coefs > coef_th (abs?) and distance > distance_th
            Params:
                distance_th; float, distance threshold
                coef_th, float, coefficient magnitude threshold
            Returns:
                adds to df_neur (number of Trues)
            """
            dist_mat_th = dist_mat_sparse > distance_th # bool coef > threshold
            coef_mat_th = abs(lasso_reg[0]) > coef_th ## Absolute values!
            combined_mat_th = dist_mat_th.multiply(coef_mat_th) # combined bool 
            
            ind_dth = sparse.find(dist_mat_th)
            ind_cth = sparse.find(coef_mat_th)
            ind_combined = sparse.find(combined_mat_th)
            print(f'# > dth {len(ind_dth[0])},# > cth {len(ind_cth[0])},# combined {len(ind_combined[0])}')
           
            count = {'rows': np.zeros(np.shape(dist_mat_sparse)[0]),
                     'columns': np.zeros(np.shape(dist_mat_sparse)[0])}
            for iN in ind_combined[0]:
                count['rows'][iN] += 1 # add one
            for iN in ind_combined[1]:
                count['columns'][iN] += 1
                
            cname_rows = 'number_regressors_dth{}_cth{}'.format(distance_th, coef_th)
            cname_columns = 'number_regressands_dth{}_cth{}'.format(distance_th, coef_th)
            
            add_column(name=cname_rows, values=count['rows'])
            add_column(name=cname_columns, values=count['columns'])
                  
        def find_regressor_regressand_n(n = 35998):
            ind_regr = sparse.find(lasso_reg[0])
            regressands = np.where(ind_regr[1] == n)[0]
            distance_th = 0.1
            coef_th = 0.1
            rel_n = np.where(real_ind == n)[0] # find relative index (to only pos spiking neurons)
            ## Regressor:
            dist_mat_th_sor = dist_mat_sparse[:, rel_n] > distance_th # bool coef > threshold
            coef_mat_th_sor = abs(lasso_reg[0][:, rel_n]) > coef_th ## Absolute values!
            combined_mat_th_sor = dist_mat_th_sor.multiply(coef_mat_th_sor) # combined bool 
            neurons_sor = real_ind[sparse.find(combined_mat_th_sor)[0]]

            ## Regressands;
            dist_mat_th_sand = dist_mat_sparse[rel_n, :] > distance_th # bool coef > threshold
            coef_mat_th_sand = abs(lasso_reg[0][rel_n, :]) > coef_th ## Absolute values!
            combined_mat_th_sand = dist_mat_th_sand.multiply(coef_mat_th_sand) # combined bool 
            neurons_sand = real_ind[sparse.find(combined_mat_th_sand)[1]]
#            coef_sand = scipy.sparse.find(lasso_reg[0][rel_n, sparse.find(combined_mat_th_sand)[1]])
#            estimate = np.dot(np.squeeze(coef_sand[2]), Data['values'][neurons_sand, :])
#            plt.figure(20)
#            plt.hold(True)
#            plt.plot(estimate, Data['values'][n, :], '.')
            
            plt.figure(21)
            plt.subplot(2,1,1)
            plt.plot(Data['behavior'])
            plt.subplot(2,1,2)
            plt.plot(Data['values'][n, :])
            plt.legend(['behavior', 'data'])
            
        def analysis_complete_network(coef_th = 0.1, distance_th = 0.1, save_clusters = False, 
                                      minimal_cluster_size = 10, verbose = True, plot_results = True):
            """
            GOAL:
                To gain an exhaustive clusterization based on the Lasso coefs
                by 'following the trace'. 
                - Fetch the regressor subset of the full Lasso map with element-whise abs(coef) > coef_th
                and distance > distance_th
                - Start at some seed, connect to all neurons in aforementioned subset
                - Move to those neurons and connect to their neurons
                - Repeat until no more neurons can be added -> assign cluster
                
                Extensions:
                    - Assay quality of cluster; take e.g. R2 into account when connecting
                    
            General notes:
                Note that the clusterization is exhaustive and therefore not dependent 
                on random seed etc. All neurons within the same intra-connecting
                network are uniquely confined to 1 cluster.
                Number of regressors is leq than number of regressands, so loop through those (mayebe it does not matter efficiency wise, not sure)
                    
            """
#            dist_mat_th = dist_mat_sparse > distance_th # Distance threshold
            coef_mat_th = abs(lasso_reg[0]) > coef_th # Coefficient magnitude threshold
#            combined_mat_th = dist_mat_th.multiply(coef_mat_th) # Combined threshold
            combined_mat_th = coef_mat_th
            
            n_connections = len(sparse.find(combined_mat_th)[0])
            regressands = list(np.sort(np.unique(sparse.find(combined_mat_th)[0])))
            regressors = list(np.sort(np.unique(sparse.find(combined_mat_th)[1])))
            if verbose:
                print(f'Number of regressands: {len(set(regressands))}, regressors: {len(set(regressors))}, n_connections: {n_connections} \n'
                        f'for coeff threshold {coef_th} and distance th {distance_th}')
            
            def get_connecting_neurons(iNeuron): # function to return all neurons with which iNeuron connects
                new_connecting_regressands = sparse.find(combined_mat_th[:, iNeuron]) # tuple: (neurons, 0's, coef_values)
                new_connecting_neurons = set(new_connecting_regressands[0]) # a little tedious at the moment
                new_connecting_regressors = sparse.find(combined_mat_th[iNeuron, :]) # tuple: (neurons, 0's, coef_values)
                new_connecting_regressors = set(new_connecting_regressors[1]) # a little tedious at the moment
                new_connecting_neurons.update(new_connecting_regressors)
                new_connecting_neurons.update([iNeuron]) # add itself (needed for every initial seed)
                return list(new_connecting_neurons)
                        
            clusters = {0: []} #cluster_id: [neurons]
            cluster_id = 0
            
            remaining_neurons = set(regressors)
            while len(remaining_neurons) > 0:
                if verbose:
                    print(f'Cluster id: {cluster_id}, remaining regressors: {len(remaining_neurons)}')
                seed_neuron = remaining_neurons.pop() # get seed
                cluster_neurons = []
                cluster_neurons = get_connecting_neurons(iNeuron = seed_neuron) # first batch of cluster neurons
                for iloop, iCN in enumerate(cluster_neurons): # loop through cluster neurons
                    connections_to_iCN = np.array(get_connecting_neurons(iNeuron = iCN)) # per neuron, get its connections
                    mask_new = np.array([(connections_to_iCN[ii] not in cluster_neurons) for ii in range(len(connections_to_iCN))])
                    if len(mask_new) > 0: # if any new connections are found
                        new_neurons = list(connections_to_iCN[mask_new])
                        if len(new_neurons) > 0: # if any new connections are found
                            [cluster_neurons.append(x) for x in new_neurons] # if so, append at the back of the list (so the for loop will evenutally reach it)
                clusters[cluster_id] = cluster_neurons
                cluster_id += 1
                remaining_neurons = remaining_neurons - set(cluster_neurons)
            cl_sizes = [len(clusters[cc]) for cc in clusters]
            if plot_results:
                plt.close(12)
                plt.figure(12)
                plt.plot(cl_sizes, '.-')
                plt.xlabel('cluster id')
                plt.ylabel('cluster size')
                plt.show()
            
            neurons_clusterid = np.zeros(LengthData) - 1
            large_clusters = [cl_sizes[jj] > minimal_cluster_size for jj in range(len(cl_sizes))]
            if verbose: 
                print(f'{sum(large_clusters)} clusters geq than {minimal_cluster_size}')
            iCluster_large = 0
            for iCluster, sizeCluster in enumerate(large_clusters):
                if sizeCluster == True:
                    neurons_clusterid[real_ind[np.array(clusters[iCluster])]] = iCluster_large 
                    iCluster_large += 1
                    
            if save_clusters == True:
                filename = 'LassoNetwork_cth{}_dth{}'.format(coef_th, distance_th)
                np.save(filename, neurons_clusterid)
                
            return clusters, cl_sizes, combined_mat_th, neurons_clusterid
            
#        test = analysis_complete_network(coef_th=0.25, distance_th=0, save_clusters=True, minimal_cluster_size = 10)
        
        
        def nclusters_per_cth_dth(type_plot_analysis = 'Regular'):
            if type_plot_analysis == 'Regular':
                coefs = np.linspace(0.1, 0.5, 20)
#                dists = np.linspace(0,0.5,10)
                dists = [0]
                cluster_min_size = 20
                n_large_clusters = {x: np.zeros(len(coefs)) for x in range(len(dists))}
                for iDLoop, iDist in  enumerate(dists):
                    print(f'Distance loop {iDLoop} / {len(dists) - 1}')
                    for iCLoop, iCoef in enumerate(coefs):
                        analysis = analysis_complete_network(coef_th = iCoef, distance_th=iDist, save_clusters=False, verbose=False, plot_results=False)
                        n_large_clusters[iDLoop][iCLoop] = sum([analysis[1][j] > cluster_min_size for j in range(len(analysis[1]))])
                        if n_large_clusters[iDLoop][iCLoop] == 0: # then also =0 for all larger thresholds:
                            n_large_clusters[iDLoop][iCLoop:] = np.zeros(len(n_large_clusters[iDLoop][iCLoop:]))
                            break
                                
                plot_legend = []
                plt.close(20)
                plt.figure(20)
                for iDLoop, iDist in enumerate(dists):
                    plt.plot(coefs, n_large_clusters[iDLoop], '.-')
                    plot_legend.append(iDist)
                    plt.hold(True)
                plt.xlabel('coefs')
                plt.ylabel(f'# cluster greater than {cluster_min_size}')
                plt.legend(plot_legend)
                plt.show()
            
            if type_plot_analysis == 'Surf':
                N_coefs = len(coefs)
                N_dists = len(dists)
                coefs, dists = np.meshgrid(coefs, dists)
                n_large_clusters = np.zeros(np.shape(coefs))
                for iC in range(N_coefs):
                    print(f'Progress {iC} / {N_coefs}')
                    for iD in range(N_dists):
                        analysis = analysis_complete_network(coef_th = coefs[iC][iD], distance_th=dists[iC][iD], save_clusters=False, verbose=False, plot_results=False)
                        n_large_clusters[iC][iD] = sum([analysis[1][j] > cluster_min_size for j in range(len(analysis[1]))])
                        if n_large_clusters[iC][iD] == 0:
                            n_large_clusters[iC][iD:] = np.zeros(len(n_large_clusters[iC][iD:]))
                            break
                from mpl_toolkits.mplot3d import Axes3D # from https://matplotlib.org/examples/mplot3d/surface3d_demo.html example
                fig = plt.figure(21)
                ax = fig.gca(projection='3d')
                surf = ax.plot_surface(coefs, dists, n_large_clusters)
                plt.xlabel('coef threshold')
                plt.ylabel('dist threshold')
                plt.zlabel('n clusters')
                
                plt.show()
            
            return n_large_clusters
            
            
def sparse_regression():
    data = Data['values']
    ind_zeros = np.where(data < 0.001)
    data[ind_zeros] = 0
    frac_nonzeros = sum(sum(data != 0)) / (np.shape(data)[0] * np.shape(data)[1])
    sparse_data = sparse.csr_matrix(data)
    del data
    N_neurons = np.shape(sparse_data)[0]
    FR = np.squeeze(np.array(sparse_data.sum(axis=1)))
    FR_arg = np.argsort(FR) # small to large
    
    ## Lasso:
    data_use = sparse_data.T

    def sparse_lasso_go():
        T_split = 1593 ## NB: Manually input from Matlab file.!!
        n_nonzero_coefs_stim = sparse.lil_matrix((N_neurons, N_neurons))
        scores_stim = {'train': np.zeros(N_neurons), 'test': np.zeros(N_neurons)}
        n_nonzero_coefs_spont = sparse.lil_matrix((N_neurons, N_neurons))
        scores_spont = {'train': np.zeros(N_neurons), 'test': np.zeros(N_neurons)}
#        neurons_use = np.flip(FR_arg, axis =0)  # sort from large to small
#        neurons_use = np.random.randint(0, N_neurons, 1000)
        neurons_use = np.where(np.logical_not(neuropil_neurons))[0]
        mask = np.zeros(N_neurons, dtype = 'bool')
        mask[neurons_use] = True
        for iLoop, iNeuron in enumerate(neurons_use):
            if iLoop % 250 == 0:
                print(f'Loop: {iLoop} / {len(neurons_use)}, time {str(datetime.now())}')
            mask[iNeuron] = False
            if iLoop > 0:
                mask[current_neuron] = True # in fact previous neuron
                del reg_model_stim
                del reg_model_spont
            current_neuron = iNeuron
            reg_model_stim = Lasso(alpha=0.002)
            reg_model_stim.fit(X=data_use[:T_split, mask], y=data_use[:T_split, iNeuron].toarray())
            temp_coef_stim = np.zeros(N_neurons)
            temp_coef_stim[mask] = reg_model_stim.coef_
            n_nonzero_coefs_stim[iNeuron, :] = temp_coef_stim   
            scores_stim['train'][iNeuron] = reg_model_stim.score(X=data_use[:T_split, mask], y=data_use[:T_split, iNeuron].toarray())
            scores_stim['test'][iNeuron] = reg_model_stim.score(X=data_use[T_split:, mask], y=data_use[T_split:, iNeuron].toarray()) 
            
            reg_model_spont = Lasso(alpha=0.002)
            reg_model_spont.fit(X=data_use[T_split:, mask], y=data_use[T_split:, iNeuron].toarray())
            temp_coef_spont = np.zeros(N_neurons)
            temp_coef_spont[mask] = reg_model_spont.coef_
            n_nonzero_coefs_spont[iNeuron, :] = temp_coef_spont  
            scores_spont['test'][iNeuron] = reg_model_spont.score(X=data_use[:T_split, mask], y=data_use[:T_split, iNeuron].toarray())
            scores_spont['train'][iNeuron] = reg_model_spont.score(X=data_use[T_split:, mask], y=data_use[T_split:, iNeuron].toarray()) 
            
            
            
            
            
            
        save_dict = {'trained_stim': {}, 'trained_spont': {}, 'info': {}}
        save_dict['trained_stim']['coefs'] = scipy.sparse.csr_matrix(n_nonzero_coefs_stim)
        save_dict['trained_stim']['score_train'] = scores_stim['train']
        save_dict['trained_stim']['score_test'] = scores_stim['test']
    
        save_dict['trained_spont']['coefs'] = scipy.sparse.csr_matrix(n_nonzero_coefs_spont)
        save_dict['trained_spont']['score_train'] = scores_spont['train']
        save_dict['trained_spont']['score_test'] = scores_spont['test']
    
        save_dict['info']['neurons_used'] = neurons_use
        save_dict['info']['method'] = 'lasso regression'
        save_dict['info']['alpha'] = 0.002
        save_dict['info']['T_split'] = T_split
        save_dict['info']['runtime'] = '47 hours'
        
        np.save('lasso_reg_spont_stim_0002_all', save_dict)
        save_dict = np.load('lasso_reg_spont_stim_0002_all.npy')
        save_dict = save_dict[()]
        
        df_reg = pd.DataFrame({'neuron':  [x for x in range(N_neurons)]}) # create df with neuron indices
        df_reg.loc[neurons_use, '#SpikeEvents'] = SpikesPerNeuron[neurons_use]
        df_reg.loc[neurons_use, 'Cumulative_SpikesValue'] = CumSpikeValuePerNeuron[neurons_use]
        df_reg.loc[:, 'spont_train'] = scores_spont['train']
        df_reg.loc[:, 'spont_test'] = scores_spont['test']
        df_reg.loc[:, 'stim_train'] = scores_stim['train']
        df_reg.loc[:, 'stim_test'] = scores_stim['test']
        
        df_reg.loc[:, 'spont_sum_sions_p_sor'] = np.array(abs(save_dict['trained_spont']['coefs']).sum(axis = 0))[0] # column sum ()
        df_reg.loc[:, 'spont_sum_sors_p_sion'] = np.squeeze(np.array(abs(save_dict['trained_spont']['coefs']).sum(axis = 1))) # row sum
        df_reg.loc[:, 'stim_sum_sions_p_sor'] = np.array(abs(save_dict['trained_stim']['coefs']).sum(axis = 0))[0] # column sum ()
        df_reg.loc[:, 'stim_sum_sors_p_sion']= np.squeeze(np.array(abs(save_dict['trained_stim']['coefs']).sum(axis = 1))) # row sum
        
        df_reg.loc[:, 'spont_sors_per_sion'] = np.squeeze(np.array((save_dict['trained_spont']['coefs'] != 0).sum(1))) # row sum
        df_reg.loc[:, 'spont_sions_per_sor'] = np.squeeze(np.array((save_dict['trained_spont']['coefs'] != 0).sum(0))) # column sum
        df_reg.loc[:, 'stim_sors_per_sion'] = np.squeeze(np.array((save_dict['trained_stim']['coefs'] != 0).sum(1))) # row sum
        df_reg.loc[:,'stim_sions_per_sor'] = np.squeeze(np.array((save_dict['trained_stim']['coefs'] != 0).sum(0))) # column sum
        
        export_df(df=df_reg, ename='lassoreg_stim_spont_0002_df.h5')
        
    lasso_reg = {}
    lasso_reg[0] = n_nonzero_coefs
    lasso_reg[1] = scores
    lasso_reg[2] = list(np.sort(neurons_use))
        
    plt.plot(scores['train'][neurons_use], scores['test'][neurons_use],'.')
    plt.spy(n_nonzero_coefs_spont, markersize=1)
    
    
    
    
def spectral_clustering(number_clusters = 30):
    mat = scipy.sparse.load_npz('sparse_lassoregr_spikingneurons.npz')
    matabs = np.abs(mat) # affinity matrix must be all positives...
    sc = sklearn.cluster.SpectralClustering(n_clusters=number_clusters, eigen_solver='amg',
                                            affinity='precomputed', n_jobs=2)
    sc.fit(matabs) # mat is sparse nxn affinity matrix.
    cluster_labels = sc.labels_
    plt.close(31)
    plt.figure(31)
    plt.hist(cluster_labels, number_clusters)
    
spectral_clustering(number_clusters=100)
    
def change_sparse_connectivity_mat():
    full_len = len(data['values']) 
    partial_mat = scipy.sparse.load_npz('sparse_lassoregr_spikingneurons.npz')
    partial_len = partial_mat.shape[0]
    conversion_indices = np.load('AbsoluteNeuronIndices.npy')
    full_mat = scipy.sparse.lil_matrix((full_len, full_len))
    assert len(conversion_indices) == partial_len    
    rows_partial, cols_partial, vals_partial = scipy.sparse.find(partial_mat)
    n_elements = len(rows_partial)
    for i_element in range(n_elements):
        full_mat[conversion_indices[rows_partial[i_element]], conversion_indices[cols_partial[i_element]]] = vals_partial[i_element]
    full_mat = scipy.sparse.csr_matrix(full_mat)
    scipy.sparse.save_npz('sparse_lasso_regr_spikingneurons_full.npz', full_mat)    
    
#    partial_mat = save_dict['trained_spont']['coefs']
#    scipy.sparse.save_npz('sparse_lasso_regr_noneuropil_spont_full.npz', partial_mat)
#    
        
    
WIP_CGC = True
if WIP_CGC:
    """
    Work on using a Conditional Granger Causality metric to estimate connectivity.
    Based on Francis et al. (2018, neuron) and the Sheikhattar/Babadi papers from 2016.
    
    Method:
        1) Compute connectivity matrix by (lasso) regression, this is done beforehand
           and loaded here.
        2) Compute the score metric for all regressions (the full model FM). 
            I use the MSE as done by Francis et alii (2018). They name this sigma-hat squared.
        3) Redo the regression, leaving 1 existing connection out per time (the reduced model, RM)
            This can either be done in full, or greedy (only using nonzero components
            of the FM to compute). Compute score metric and compute the deviance from this.
            
            The difference between greedy and full regression is that in full regression
            'connections' that are 0 in the Full model, can be changed to nonzero
            components in the Reduced model, while in the greedy regressions only 
            nonzero components are considered. This difference is inspected visually below.
    
    """
    load_data = False
    if load_data:
        # load data and make subsets for WIP
        lasso_reg = {}
        lasso_reg[0] = scipy.sparse.load_npz('sparse_lasso_regr_noneuropil_stim_full.npz')
        plt.figure(19)
        plt.spy(lasso_reg[0], markersize=1, aspect='auto')
        test5 = lasso_reg[0][36000:41000, 36000:41000] # 5k neurons subset
        plt.figure(20)
        plt.spy(test5, markersize=1, aspect='auto')
        
        test2 = lasso_reg[0][36000:38000, 36000:38000] # 2k neurons subset
        plt.figure(21)
        plt.spy(test2, markersize=1, aspect='auto')
    
    data_use = Data['values'][36000:38000, :]
    data_use = Data['values']
    def lasso_fun(dat, n_reg, n_red=None, mode='f', alp=0.002, t_end=1593, splitup=True,
                  return_metric='R2', method='lasso', put_in_connmat=False): #'f' full or 'r' reduced 
   
     # assuming transformed [time, neurons]
        if n_reg % 100 == 0:
            print(f'Neuron {n_reg}')
            
        mask = np.ones(dat.shape[1], dtype='bool')
        if mode == 'r':
            mask[n_red] = False
        
        if method == 'lasso':
            reg = Lasso(alpha=alp)
            reg.fit(dat[:t_end, mask], dat[:t_end, n_reg])
        elif method == 'OLS':
            reg = LinearRegression()
            reg.fit(dat[:t_end, mask], dat[:t_end, n_reg])
     
        if put_in_connmat:
            if mode == 'f':
                fm_mat[n_reg, mask] = reg.coef_
            elif mode == 'r':
                rm_mat[n_reg, mask] = reg.coef_
        
        return_coefs = np.zeros(dat.shape[1]) # return coefs
        return_coefs[mask] = reg.coef_
        
        if return_metric == 'R2':
            if splitup:
                rm = reg.score(dat[t_end:, mask], dat[t_end:, n_reg]) # return R^2 value
            elif not splitup:
                rm = reg.score(dat[:t_end, mask], dat[:t_end, n_reg]) 
        elif return_metric == 'MSE':
            if splitup:
                test_len= len(np.squeeze(dat[t_end:, n_reg]))
                rm = (1/test_len) * sum((np.dot(dat[t_end:, mask], reg.coef_) - dat[t_end:, n_reg]) ** 2) # return MSE
            elif not splitup:
                test_len= len(np.squeeze(dat[:t_end, n_reg]))
                rm = (1/test_len) * sum((np.dot(dat[:t_end, mask], reg.coef_) - dat[:t_end, n_reg]) ** 2) # return MSE
                
        return rm, return_coefs, reg.intercept_
    
    def lasso_fun_part(dat, n_reg, n_predictors, alp=0.002, t_end=1593):
        # Greedy Lasso, only use given n_predictors
        mask = np.zeros(dat.shape[1], dtype='bool')
        mask[n_predictors] = True
        las = Lasso(alpha=alp)
        las.fit(dat[:t_end, mask], dat[:t_end, n_reg])
        MSE = (1/1507) * sum((np.dot(dat[t_end:, mask], las.coef_) - dat[t_end:, n_reg]) ** 2) # return MSE
        return MSE
    
    def compute_fmpreloaded_scores(dat, mat, alp=0.002, t_end=1593): # calculate MSE scores of preloaded connectivity mat
        # This takes a while (20 minutes) for the full ~80kx80k matrix
        #TODO: Find bottleneck.?
        n_neurons = mat.shape[0]
        scores = np.zeros(n_neurons)
        def compute_score(n): # compute MSE
            if n % 1000 == 0:
                print(n)
            sc = (1/1507) * sum((np.dot(dat[t_end:, :], np.squeeze(mat[n, :].A)) - dat[t_end:, n]) ** 2)
            return sc
        scores_temp = [dask.delayed(compute_score)(n) for n in np.arange(n_neurons)]
        scores = dask.compute(scores_temp)
        return scores
    fm_scores_all = compute_fmpreloaded_scores(dat=data_reg, mat=lasso_reg[0])
    fm_scores_all = np.squeeze(fm_scores_all)
    print(datetime.now())
    
    data_reg = data_use.transpose() # transpose for Lasso function
    
    fm_mat = scipy.sparse.lil_matrix((2000, 2000)) # compute the full model, for a small subset
    fm_score_temp = [dask.delayed(lasso_fun)(data_reg, n_reg=i_neuron) for i_neuron in range(2000)]
    fm_scores = np.array(dask.compute(fm_score_temp))[0] 
    
    plot_fm_and_test = False # plot fm and same neurons from original mat to see difference
    # typically the original (test2) has less connections 
    if plot_fm_and_test:
        plt.figure(34)
        plt.subplot(1,2,1)
        plt.spy(test2, markersize=1, aspect='auto')
        plt.title('test 2')
        
        plt.subplot(1,2,2)
        plt.spy(fm_mat, markersize=1, aspect='auto')
        plt.title('fm')
        
    def one_RM(n_reduced=1860):
        # leave one neuron out, for all regressions. Compare the two visually. SLOW version!
        rm_mat = scipy.sparse.lil_matrix((2000, 2000))
        rm_score_temp = [dask.delayed(lasso_fun)(data_reg, n_reg=i_neuron, n_red=n_reduced, mode='r') for i_neuron in range(2000)]
        rm_scores = np.array(dask.compute(rm_score_temp))[0]
        fraction_diffs = fm_scores / rm_scores
        deviance = 1507 * np.log(rm_scores / fm_scores)
        
        
        plt.close(23)
        plt.figure(23)
        plt.subplot(2,2,2)
        plt.spy(rm_mat, markersize=1, aspect='auto')
        plt.title('Spy matrix of neurons 36k:38k')
        
        plt.subplot(2,2,3)
        plt.plot(np.arange(2000), np.squeeze(deviance), '*')
        plt.ylabel('Scores, fm/rm')
        plt.xlabel('neurons')
        
        plt.subplot(2,2,1)
        plt.spy(fm_mat, markersize=1, aspect='auto')
        plt.title('Spy matrix of neurons 36k:38k, fm')
        
        plt.show()
        
    def all_RM():
        """
        Leave one connections out 1 by 1
        
        """
#        connections = scipy.sparse.find(fm_mat)
        connections = scipy.sparse.find(lasso_reg[0]) # get nonzero connections
#        deviance_connections = np.zeros_like(connections) # to put deviances of full regression
        deviance_greedy_connections = np.zeros_like(connections) # only use nonzero comps from fm_mat to evaluate (greedy)
        
        def leaveoneout_reg_n(n_row): # for regression of neuron n_row, leave all its components out 1 by 1
            index_connections = np.where(connections[0] == n_row)[0] # find it in connections array
            for k, n_leaveout in enumerate(connections[1][index_connections]): # leave one out each time
                # full regression!
#                mse = lasso_fun(dat=data_reg, n_reg=n_row, n_red=n_leaveout, mode='r') 
#                dev = 1507 * np.log(mse / fm_scores[n_row]) 
#                deviance_connections[:, index_connections[k]] = [n_row, n_leaveout, dev] # add to deviances mat
                
                # Greedy regression!
                if len(index_connections) > 1: # only execute if more than 1 element was predictor   
                    predictors = set(connections[1][index_connections])
                    predictors.remove(n_leaveout)
                    predictors = np.array(list(predictors)) 
                    mse_greedy = lasso_fun_part(dat=data_reg, n_reg=n_row, n_predictors=predictors)
                    dev_greedy = 1507 * np.log(mse_greedy / fm_scores_all[n_row])
                    deviance_greedy_connections[:, index_connections[k]] = [n_row, n_leaveout, dev_greedy] # add to greedy deviances mat
                
        add_deviances_temp = [dask.delayed(leaveoneout_reg_n)(n) for n in np.unique(connections[0])]
        add_deviances = dask.compute(add_deviances_temp) # compute all deviances
        
        deviance_mat = scipy.sparse.lil_matrix((data_use.shape[0], data_use.shape[0])) # put deviances in same format as connectivity mat
        deviance_mat[np.squeeze(deviance_connections[0, :]), np.squeeze(deviance_connections[1, :])] = np.squeeze(deviance_connections[2, :])
        
        deviance_greedy_mat = scipy.sparse.lil_matrix((data_use.shape[0], data_use.shape[0])) # put deviances in same format as connectivity mat
        deviance_greedy_mat[np.squeeze(deviance_greedy_connections[0, :]), np.squeeze(deviance_greedy_connections[1, :])] = np.squeeze(deviance_greedy_connections[2, :])
        
        percentile_threshold = 98 # percentile threshold to plot in spy()
#        deviance_marker = np.percentile(deviance_connections[2, :], percentile_threshold)
        deviance_greedy_marker = np.percentile(deviance_greedy_connections[2, :], percentile_threshold)
        regress_coef_marker = np.percentile(connections[2], percentile_threshold)
        print(f'Number of elements of {percentile_threshold} percentile: '
              f'Reg. coef.: {sum(connections[2] > regress_coef_marker)}  '
              f'Greedy Dev.: {sum(deviance_greedy_connections[2, :] >  deviance_greedy_marker)}') # sanity check
        
        #TODO: Add OLS and Ridge regression!!
        #TODO: Difference auto versus nonauto regression
        
        plt.figure(36) 
        plt.subplot(1,2,1)
        plt.spy(lasso_reg[0], precision=regress_coef_marker, markersize=1, aspect='auto')
        plt.title('regression coefs')

#        plt.subplot(1,3,2)
#        plt.spy(deviance_mat, precision=deviance_marker, markersize=1, aspect='auto')
#        plt.title('deviances')
#    
        plt.subplot(1,2,2)
        plt.spy(deviance_greedy_mat, precision=deviance_greedy_marker, markersize=1, aspect='auto')
        plt.title('deviances greedy method')
        

#%% 
analyze_locals_cl = False
if analyze_locals_cl:
    localcl = np.load('localities_2018-02-22_14_32.npy')
    cl_activities = np.zeros((int(max(localcl) + 1), len(np.squeeze(data['times']))))
    for cl in range(0, int(max(localcl) + 1)): # loop through clusters
        cl_neurons = np.where(localcl == cl)[0]
        
        mean_activity = data['values'][cl_neurons, :].mean(axis=0)
        cl_activities[cl, :] = mean_activity
    
    corrmat = np.corrcoef(cl_activities)
    plt.figure(2)
    sns.heatmap(corrmat)
    
    
def temp_makeh5frommat():
    HUweights = h5py.File('weight_matrix_RBM.h5', 'r')
    weights = np.transpose(HUweights['Data']['weights'].value, (1,0))
    df_weights = pd.DataFrame(weights)
    df_weights.columns= [f'HU_{x}' for x in range(50)]
    store_df = pd.HDFStore('pd_RBM_weights.h5')
    store_df['weights'] = df_weights
    store_df.close()
    
    
def make_regression_network():    
    neurons = [ 4522,  9080,  9438, 50335, 50369, 54209, 57854, 59844, 61095,
64476, 70144, 79961, 81173]
    neurons = [24237, 24243, 27835, 81506, 84485, 87001, 87140] 
       
    plt.figure(1)
    for n in range(62):
        plt.plot(np.squeeze(data['times']), np.squeeze(data_use[: ,n]))
        plt.hold(True)
    plt.legend([f'{x}' for x in neurons])
    plt.hold(False)
    
    corrmat = np.corrcoef(data['values'][neurons, :])
    plt.figure(2)    
    sns.heatmap(corrmat)
    
    def draw_graph(corrmat, positions=None, fig=3):
        connections = np.where(np.abs(corrmat)) # get indices
        df_edges = pd.DataFrame({'from': connections[0], 'to': connections[1], 'weight': corrmat[connections]}) # create edge list pd.df with weights
        df_edges_filt = df_edges[df_edges['from'] != df_edges['to']] # filter out self connections
        
        G = nx.from_numpy_matrix(corrmat, create_using=nx.DiGraph().reverse())
#        G = nx.from_pandas_edgelist(df_edges_filt, 'from', 'to', create_using=nx.DiGraph()) # create graph from edgelist
        if positions == None:
            positions = graphviz_layout(G)
        plt.figure(fig)
        GR = G.reverse()
        nx.draw(GR, pos=positions, with_labels=True, node_size=300, width=list(df_edges_filt['weight'])) # draw graph with edge width ~ weights^4
        plt.show()
        return positions
        
    
#      def lasso_fun(dat, n_reg, n_red=None, mode='f', alp=0.002, t_end=1593, splitup=True,
#                  return_metric='R2', method='lasso'): #'f' full or 'r' reduced 
    data_use = data['values'][neurons, :].transpose() # transpose for regression fucntion
    coefs_mat = {}
    coefs_mat['lasso'] = np.zeros((data_use.shape[1], data_use.shape[1]))
    coefs_mat['OLS'] = np.zeros((data_use.shape[1], data_use.shape[1]))
    R2_array_lasso = np.zeros(data_use.shape[1])
    R2_array_OLS = np.zeros(data_use.shape[1])
    R2_array = {}
    R2_array['OLS'] = np.zeros(data_use.shape[1])
    R2_array['lasso'] = np.zeros(data_use.shape[1])
    intercepts = {}
    intercepts['OLS'] = np.zeros(data_use.shape[1])
    intercepts['lasso'] = np.zeros(data_use.shape[1])
    for n in range(data_use.shape[1]):
        regression = lasso_fun(data_use, n_reg=n, n_red=n, mode='r', t_end=1500,
                               splitup=True, method='lasso', alp=0.002)
        R2_array_lasso[n] = regression[0]
        R2_array['lasso'][n] = regression[0]
        coefs_mat['lasso'][n, :] = regression[1]
        intercepts['lasso'][n] = regression[2]
         
        regression = lasso_fun(data_use, n_reg=n, n_red=n, mode='r', t_end=1500,
                               splitup=True, method='OLS')
        R2_array_OLS[n] = regression[0]
        R2_array['OLS'][n] = regression[0]
        coefs_mat['OLS'][n, :] = regression[1]
        intercepts['OLS'][n] = regression[2]
    
    plt.close(4)
    plt.figure(4)
     
    plt.subplot(1,2,1)
    plt.title('OLS')
    sns.heatmap(coefs_mat['OLS'])
    
    plt.subplot(1,2,2)
    plt.title('Lasso')
    sns.heatmap(coefs_mat['lasso'])
    
    print(f'Scores OLS: {R2_array_OLS}, Lasso: {R2_array_lasso}')
    print(f'Ratio scores Lasso/OLS: {R2_array_lasso / R2_array_OLS}')
    
    pos = draw_graph(coefs_mat['OLS'], fig=31)
    draw_graph(coefs_mat['lasso'], positions=pos, fig=7)
    
   plt.figure(13)
   plt.plot(R2_array['OLS'], R2_array['lasso'],'o')
   plt.xlabel('OLS R2 score')
   plt.ylabel('Lasso R2 score')
   
    def plot_one_neuron_regression(n, t_total=data_use.shape[0], t_train=1500, 
                                   fig=9, method='OLS'):
        plt.close(fig)
        plt.figure(fig)
        plt.plot(np.arange(t_total), data_use[:, n], 'r',
                 np.arange(t_train), np.dot(np.squeeze(coefs_mat[method][n, :]), data_use[:t_train, :].transpose()) +  intercepts[method][n], 'b', 
                 np.arange(t_train, t_total), np.dot(np.squeeze(coefs_mat[method][n, :]), data_use[t_train:, :].transpose()) +  intercepts[method][n], 'g') # to verify that reg.predict() = np.dot(coef, data)
        plt.legend(['true signal', 'trained regression', 'predicted regression'])
        plt.xlabel('time frame')
        plt.ylabel('DF/F')
        r2 = np.round(R2_array[method][n], 3)
        stringtitle = method + f' regression, neuron {neurons[n]}, R2 = {r2}'
        plt.title(stringtitle)
        plt.show()
        
    plot_one_neuron_regression(n=6,  method='lasso', fig=11)

    # run localcl analysis from above
    mean_cl_data = cl_activities.transpose()
    zero_el_cl = np.unique(np.where(np.isnan(cl_activities))[0])
    data_use = np.delete(mean_cl_data, zero_el_cl, axis=1)

    
#%% PCA
    
def PCA_analysis(iC = 0, neurons = [0,1], n_pca_comps = 10, NeuronsCluster = 'Cluster', plot_expl_var = True):
    """
    Either input a cluster of neurons,
    Compute PCA, return desired number of components
    Plot explained variance per component and cumulative,
    Plot reconstructed signal.
    Params:
        iC: int, cluster id
        neurons: list of neuron ids
        n_pca_comps: number of principal components to return
        NeuronsCluster: 'Cluster' for cluster, 'Neurons' for neurons
        plot_expl_var: bool
    Returns:
        Plots
        pca_cluster; sklearn module (correct terminology?)
    """
    pca_cluster = PCA(n_components=n_pca_comps) # define PCA
    if NeuronsCluster == 'Cluster':
        neurons_pca = np.where(correlation_clusters == iC)[0]
    elif NeuronsCluster == 'Neurons':
        neurons_pca = neurons
    if len(neurons_pca) > 1:
        pca_cluster.fit(Data['values'][neurons_pca, :])
        cum_expl_var_rat = np.cumsum(pca_cluster.explained_variance_ratio_) # cumulative explained variance
        if plot_expl_var == True:
            print(f'Cluster id: {iC}, cluster size  = {cluster_sizes[iC]}') 
            plt.figure(7) # plot cumulative normalized explained variance
            plt.plot(cum_expl_var_rat)
            plt.xlabel('Number of principal components (right?)')
            plt.ylabel('Cumulative explained variance (ratio)')
            
            plt.figure(8) # plot normalized explained variance
            plt.plot(pca_cluster.explained_variance_ratio_, '.-')
            plt.xlabel('Principal component')
            plt.ylabel('Explained variance (Ratio)')
            
        reconstruct = True
        if reconstruct == True: # reconstruct signal from components
            reconstructed_data = pca_cluster.inverse_transform(pca_cluster.transform(Data['values'][neurons_pca, :]))
            plt.figure(11)
            n_comps_plot = 5
            sns.tsplot(data=reconstructed_data[0:n_comps_plot,:], ci="sd", color="green") # plot mean pm std
#            behav_corr = np.corrcoef(reconstructed_data, Data['behavior'].transpose()) # correlate with behav
            plt.title(f'Reconstructed neuronal signal from {n_pca_comps} principal components')
            plt.xlabel('time')
            plt.ylabel('df/f')
        return(pca_cluster)#, behav_corr)    
    
    
    
    
    

    
    
#%%##################
#Some irrelevant code:
###################
    
## Plotting things :
def plot_total_corr_dist_hist(CorrMat, DistMat, nbins = 40):
    histogram_correlation = np.histogram(CorrMat, bins = nbins, range=(-1,1))
    histogram_correlation[0][-1] = histogram_correlation[0][-1] - len(CorrMat) # substract all autocorrelations
    
    histogram_distance = np.histogram(DistMat, bins =int(0.5 * nbins))
    histogram_distance[0][0] = histogram_distance[0][0] - len(DistMat) # remove autodistances
    plt.figure(5)
    plt.subplot(1,3,)
    plt.step(histogram_correlation[1][1::], 0.5 * histogram_correlation[0], '.-')
    plt.xlabel('Pairwise Correlation coefficient')
    plt.ylabel('Number of neurons')
#    plt.legend(['correlation', ' distance'])
    plt.yscale('log')
    plt.subplot(1,3,3)
    plt.step(histogram_distance[1][1::], 0.5 * histogram_distance[0], '.-')
    plt.xlabel('Pairwise Distance')
    plt.legend(['correlation', ' distance'])
    plt.yscale('log')
    
def plot_hist_corr_pos(df): # plot the histograms for positive correlation counts
    nbins = 50
    N = len(df)
    for c in range(10): # loop through thresholds (/10)
        name = 'global_#corr_geq_0{}'.format(c) # define name correlation greater or equal than 0.c
        plt.hist(df[name]/N, bins = nbins, histtype = 'step', linewidth = 1) # plot hist and hold on till next hist
        plt.hold(True)
#    hs = plt.hist(df['SpikeEvents']/3550, bins = nbins, histtype = 'step', linewidth = 1, color = 'k') # plot spikes
    plt.legend(['0','0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9', 'spikes'])
    plt.xlabel('Bin of fraction of neurons with correlation x')
    plt.ylabel('Number of neurons in that bin')
    plt.yscale('log')
    
def plot_hist_corr_neg(df):    # plot the histograms for negative correlation counts 
    nbins = 50
    N = len(df)
    for c in range(10):
        name = 'global_#corr_leq_m0{}'.format(c)
        plt.hist(df[name]/N, bins = nbins, histtype = 'step', linewidth = 1)
        plt.hold(True)
#    hs = plt.hist(df['SpikeEvents']/3550, bins = nbins, histtype = 'step', linewidth = 1, color = 'k')
    plt.legend(['0', '-0.1', '-0.2', '-0.3', '-0.4', '-0.5', '-0.6', '-0.7', '-0.8', '-0.9', 'spikes'])
    plt.yscale('log')
    
def plot_corr_1neuron(df, reln = 0): # plot correlation(bins) for one neuron
    N = len(df)
    plt.plot(np.linspace(-0.9,0,10), (df.iloc[reln, list(range(3,13))] / N), 'o-')
    plt.plot(np.linspace(0,0.9,10), (df.iloc[reln, list(range(13,23))] / N), '*-')
    plt.ylabel('Fraction neurons')
    plt.xlabel('Correlation minimum (pos) or maximum (neg) threshold')
#    plt.yscale('log')
  
def plot_corr_1neuron_rev(df, reln = 0): # plot bins(correlation) for one neuron
    N = len(df)
    plt.plot((df.iloc[reln, list(range(3,13))] / N), np.linspace(-0.9,0,10), 'o-')
    plt.plot((df.iloc[reln, list(range(13,23))] / N), np.linspace(0,0.9,10), '*-')
    plt.xlabel('Fraction neurons')
    plt.ylabel('Correlation minimum (pos) or maximum (neg) threshold')

def export_statistic(df, column,  norm = 1000, save = 0, name = 'defaultname'): # export some statistic, saved in a column, to a vector 
    N_all = len(SpikesPerNeuron)
    Stat = np.zeros([N_all]) # create zeros for non-specified neurons
    Stat[df['neuron']] = df[column] / norm# fetch column
    if save == 1: # save as seperate file in current directory
        np.save(name, Stat) # Save as .npy file, load with "test = np.load('defaultname.npy')"
    return Stat
    
def seed_neuron_corr(df, n, CorrMat, saves = 0, cname = 'correlation_n'): # correlation w.r.t. seed neuron
    CorrMat = CorrMat[:,n] # get column of neuron n
    ## how to handle negative values (=0, abs, ..?)
#    df_neur[0][cname] = CorrMat # save in df
    df_neur[cname] = CorrMat # save in df
#    export_statistic(df_neur[0], cname, norm = 1, save = saves, name = cname) # save as seperate file
    return CorrMat

def seed_neuron_pncorr(df, n, CorrMat, save = 0, cname = 'seedpn', th_p = 0.3, th_n = -0.2): # to highlight high corr and anti-corr w.r.t a seed neuron
    CorrMat = CorrMat[:,n] # Get column of neuron n
    seed_corrmat = np.zeros(len(CorrMat))
    ## Save two distinct values that can be fishualized:
    seed_corrmat[(CorrMat > th_p)] = 0.8
    seed_corrmat[(CorrMat < th_n)] = 0.3
    df[cname] = seed_corrmat # save in df
#    export_statistic(df, cname, norm = 1, save =1, name = cname) # save as seperate file
    return seed_corrmat  
#try_HBO_neuron = seed_neuron_pncorr(df = df_neur, n = 19050, CorrMat = C_neur, save = 1, cname ='HBOseed_65151', th_p = 0.3, th_n = -0.3)
####### 19991 19050 
def get_indices_nonzero_pncorrseed(pncorr, save  = 0): # Get the indices of nonzero positive or negative correlation w.r.t. seed neuron
    ## Named HBO because that is what I tried this on first
    HBO_ind = np.where(pncorr) # find non zeros
    HBO_ind = HBO_ind[0]
    HBO_ind = list(HBO_ind) # put in list
    HBO_neurons = df_neur['neuron'][HBO_ind] # get absolute indices
    if save == 1:
        np.save('HBOneuronslist', HBO_neurons) # save
    return HBO_neurons

## Plot correlation histograms per threshold:
#plot_hist_corr_pos(df_neur) # plot the histograms
#plot_hist_corr_neg(df_neur)

## Plot histogram of all correlation and distance pairwise values:
#plot_total_corr_dist_hist(C_neur, D_neur, nbins= 40) 



## Part for getting pairwise combinations in a new matrix plus some plotting 
def dist_corr2(dist, corr, NOIindp): # two matrices and neuron indices ( for Pandas)
    if np.shape(dist) != np.shape(corr): # must be same size
        pass
    else:
        N_neurons = len(dist) # number of neurons
        neurons_ind = np.triu_indices(N_neurons, 1) # get indices of upper triangle of matrix in tuple form: (irows, icolumns), the 1 denotes leaving out diaganol (auto distance/corr)
        
        # Put everything together in a pandas dataframe:
        df = pd.DataFrame({'neuron1': [NOIindp[n] for n in neurons_ind[0]], 'neuron2': [NOIindp[n] for n in neurons_ind[1]], 'distance': dist[neurons_ind], 'correlation': corr[neurons_ind]})
        return df        

def run_all(coords, values, NOIind): # get distance, correlation and put in pandas dataframe
    Distance = comp_dist(coords)
    Correlation = comp_corr(values)
    DistCorr_comb = dist_corr2(Distance, Correlation, NOIindp = NOIind)
    return DistCorr_comb

def run_regionthresh(thresh = 5, RorN = 'N', ROI = 274, NOI_input = [0,1,2]): # set spiking threshold and ROI (can be list)
    ## Set RorN to either 'R' for ROI selection, or 'N' for direct neuron indices input
    if RorN == 'R': # select based on ROI and thresh
        NOI = select_neurons(ROIs = ROI, threshold = thresh)
    elif RorN == 'N': #use given list of neuron ids, threshold cna also be taken into account
        NOI = NOI_input
    print("Number of neurons: ", format(len(NOI)))
    result = run_all(Data['coords'][NOI,:], Data['values'][NOI,:], NOIind = NOI) # compute pairwise things       
    result['absolute_corr'] = np.abs(result['correlation']) # add absolute correlation
    return result 

def add_ranges(df, nbins= 40): # divide distance into bins
    bin_max = np.max(df['distance'])
    bins = np.linspace(0, bin_max, nbins) #NB: currently distance == 0 is not taken into account
    distance_range = pd.cut(df['distance'], bins)
    df['dist_range'] = distance_range # add to df, does not need to be returned
#    return bins # return bin-array for xlabeling

def plotting_pn(df): # pos and neg divided
    ## boxplots are nice, violinplots can be nice for density but have annoying tails (can maybe be turned off?)
    plt.close(1)
    plt.figure(1)
#    sns.violinplot(x = 'dist_range',y = 'correlation', data = df[ ( df['correlation'] > 0)] )   
    hpb = sns.boxplot(x = 'dist_range',y = 'correlation', data = df[ ( df['correlation'] > 0)] )
    plt.title('Positive correlation coef. only')
    hpb.xaxis.set_major_locator(mpl.ticker.MultipleLocator(7))
#    hpb.xaxis.set_major_formatter(mpl.ticker.ScalarFormatter())
    plt.xlabel('Distance bin')
    plt.show()
    
    plt.close(2)
    plt.figure(2)
#    sns.violinplot(x = 'dist_range',y = 'correlation', data = df[ ( df['correlation'] < 0)] )
    hnb = sns.boxplot(x = 'dist_range',y = 'correlation', data = df[ ( df['correlation'] < 0)] )
    plt.title('Negative correlation coef. only')
    hnb.xaxis.set_major_locator(mpl.ticker.MultipleLocator(7))
#    hnb.xaxis.set_major_formatter(mpl.ticker.ScalarFormatter())
    plt.xlabel('Distance bin')
    plt.show()
    
def plotting_few(df):
    plt.close(6)
    plt.figure(6)
    sns.jointplot(x = 'distance', y = 'correlation', data = df)#, kind = 'kde')    
#    sns.jointplot(x = 'distance', y ='correlation', data = df[ (df['distance'] > 0 ) & (df['distance'] < 0.00535) ]) 
#    plt.scatter(x = 'distance', y = 'correlation', data = df)
    

def get_uniqueneurons(df): # from df with columns named  neuron1 and neuron2
    new_neurons = pd.concat([df['neuron1'], df['neuron2']]) # combine both columns
    new_neurons = [int(x) for x in new_neurons] # set indices to int
    new_neurons = np.unique(new_neurons) # remove doubles
    return new_neurons

def two_neurons_plot_cc(n1 = 24157, n2 = 72447): # Plot traces of two neurons and crosscorrelation
#    kn = 0
#    n1 = subsample.iat[kn, 2] # neuron 1
#    n2 = subsample.iat[kn, 3] # neuron 2
    print(n1,  n2)
    plt.close(3)
    plt.figure(3)
    plt.figure
    times = np.arange(1,3551) # time axis
    plt.subplot(3,1,1) # neuron 1
    plt.plot(times, get_train(n1), times, Data['values'][n1])
    print(Data['labels'][n1])
    plt.subplot(3,1,2) # neuron 2
    plt.plot(times, get_train(n2), times, Data['values'][n2])
    print(Data['labels'][n2])
    plt.subplot(3,1,3) # plot behavior
    plt.plot(times, Data['behavior'][:,0],'r', times, Data['behavior'][:,1],'k', times, Data['behavior'][:,2], 'b')
    
    ## Normalize cross correlation?
    crosscorr = [scipy.signal.correlate(Data['values'][n1], Data['values'][n2]),
                 scipy.signal.correlate(get_train(n1), get_train(n2))] # cross correlation, does not normalize tails.?
    maxcc = len(crosscorr[0])
    xlabelscc = np.linspace(-1 * ((maxcc -1) / 2), (maxcc - 1)/2, maxcc) # for plotting (relative to zero)
    plt.close(4)
    plt.figure(4)
    plt.plot(xlabelscc, crosscorr[0], xlabelscc, crosscorr[1])
    plt.legend('calcium', 'spikes')

#df_pairwise_combs = run_regionthresh(ROI = 113, thresh = 5) # forebrain
##%prun run_regionthresh(ROI = 0, thresh = 0) # get time profile to find bottleneck
#add_ranges(df_pairwise_combs, nbins = 25) # bin
#plotting_pn(df_pairwise_combs) # plot
##plotting_few(df_ROIthresh)
#subsample = df_pairwise_combs[ (df_pairwise_combs['correlation'] < -0.7) & (df_pairwise_combs['distance'] > 0.01) & (df_pairwise_combs['neuron1'] != df_pairwise_combs['neuron2'])]
#two_neurons_plot_cc(n1 =  subsample.iat[0, 2], n2 =  subsample.iat[0, 3])


#%% Ridge regression (version 1)
def ridge_1neuron(AllNeurons = Data['values'].T, neuron_k = 1000, alpha_r = 0.1, lag = 'no'):
    if lag == 'no':
        Data_k = AllNeurons[:,neuron_k]
    elif lag == 'yes':
        Data_k = AllNeurons[1::, neuron_k]
        AllNeurons = AllNeurons[0:-1, :]
#    AllNeurons = np.delete(AllNeurons, (neuron_k), axis = 1) # lose itself as regressor
    neuron_ridge = Ridge(alpha = alpha_r)
    neuron_ridge.fit(AllNeurons, Data_k)
    return neuron_ridge.coef_

def lasso_1neuron(AllNeurons = Data['values'].T, neuron_k = 1000, alpha_l = 1.0, lag = 'no'):
    if lag == 'no':
        Data_k = AllNeurons[:,neuron_k]
    elif lag == 'yes':
        Data_k = AllNeurons[0:-1, neuron_k]
        AllNeurons = AllNeurons[1::, :]
    neuron_lasso = Lasso(alpha = alpha_l)
    neuron_lasso.fit(AllNeurons, Data_k)
    return neuron_lasso.coef_ # also _sparse_coef possible

## transpose Data here?
def test_alpha_val():
    alphas = np.logspace(-10, 9, 20)
    test = np.zeros([LengthData, 20])
    nn = 0
    for nn in range(10,20):
        n_alpha = alphas[nn]
        test[:,nn] = ridge_1neuron(neuron_k = 31458, alpha_r = n_alpha)
    #    nn += 1
        print(nn)   

def regress_many(method = 'confined',  spike_thresh = 50, regr_type = 'lasso', lagtmp = 'no'):
#    method = 'confined' # or 'all' or 'confined'
    neurons = list(np.where(SpikesPerNeuron > spike_thresh)[0])
    N_neurons = len(neurons)
    print("Number of neurons: {}".format(N_neurons))
    if method == 'confined': # use only the neurons > threshold as regressors
        neurons = [24237, 24243, 27835, 81506, 84485, 87001, 87140] 
        neurons = [ 4522,  9080,  9438, 50335, 50369, 54209, 57854, 59844, 61095,
       64476, 70144, 79961, 81173] 
        N_neurons = len(neurons)
        NewData = Data['values'][neurons,:]
        NewData = NewData.T # transpose for regression input format
        RegrMat = np.zeros([N_neurons, N_neurons])
    elif method == 'all':
        NewData = Data['values'].T
        RegrMat = np.zeros([N_neurons, LengthData])
        
    if regr_type == 'lasso':        
        for n in range(N_neurons):
            RegrMat[n, :] = lasso_1neuron(AllNeurons = NewData, neuron_k = n, alpha_l = 1.0, lag = lagtmp)
            if n % 10 == 0:
                print(n / N_neurons)
    elif regr_type == 'ridge': 
        for n in range(N_neurons):
            RegrMat[n, :] = ridge_1neuron(AllNeurons = NewData, neuron_k = n, alpha_r = 0.1, lag = lagtmp)
            if n % 10 == 0:
                print(n / N_neurons)
    return RegrMat

#test_all = regress_many()
#test_all_allregr_lasso = regress_many(method = 'all', regr_type = 'lasso')
#test3 = ridge_1neuron(neuron_k = 232, alpha_r = 0.1)
#plt.hist(test3, 100)

def ridgeCV_1neuron(AllNeurons = Data['values'].T, neuron_k = 1000, alphas_r = (0.1, 1.0, 10.0)):
    Data_k = AllNeurons[:,neuron_k] # neuron being regressed
    neuron_ridge = RidgeCV(alphas = alphas_r, cv = 5)
    neuron_ridge.fit(AllNeurons, Data_k)
    return neuron_ridge.coef_

#testCV = ridgeCV_1neuron(neuron_k = 1000)
