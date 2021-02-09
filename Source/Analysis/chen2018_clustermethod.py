"""

The goal is to replicate the clustering algorithm from Chen et al. (2018)
Link: https://www.biorxiv.org/content/early/2018/03/27/289413
See supp. mat., page 23

I have put their pseudo algorithm below and in code at the approriate locations.
Sentences preceded by a digit (e.g. 1. Divide all [..]) denote an objective, which 
is then acquired by the substeps preceded by a letter (e.g. a. Perform [..])
I have built my code in the same order. The downside of this is that step 4 is a bit 
messy - I now simply put steps 2/3 in a function and executed that again.

------------------------------------------------
ORIGINAL
------------------------------------------------

Automated Functional Clustering algorithm
This algorithm was custom developed to suit this dataset, and the code is available as part of the GUI. We outline the
algorithm below:
    
1. Divide all cells into “functional voxels” (~10 cells each)
    a. Perform k-means clustering on all cells (k=20)
    b. Perform k-means clustering on outputs of (a) (k = ~400)
    c. Discard any cells whose correlation with the voxel average activity is less than $THRESH
    d. Discard any voxels with fewer than 5 cells
    
2. Merge voxels into clusters based on density in functional space
    a. for each pair of voxels ij (starting from most correlated):
        if the correlation between voxel i and j is greater than $THRESH,
        and the correlation between the the voxel j and the centroid (average) of the cluster
        containing voxel i is greater than $THRESH:
        then group voxel j in the same cluster as i.
    b. discard any clusters with fewer than 10 cells
    
3. Clean up clusters using regression to cluster centroids
    a. for each cell k:
        if the correlation to the closest cluster’s centroid is greater than $THRESH:
        include cell k in that cluster
    b. Discard any clusters with fewer than 10 cells
    
4. Iterate merge and cleanup steps
    a. Perform step 2 and 3 once more, using clusters as input voxels.
    
This clustering algorithm can either be applied to all cells in the brain or a chosen subset of interest, and the correlation
threshold determining clustering stringency ($THRESH) can be adjusted to trade-off completeness and accuracy (see
Results and Fig. S4-1d). For most analysis in the text, the value of $THRESH was 0.7. 

                                            
--------------------------------------------------------
                                            
"""

import numpy as np
import matplotlib.pyplot as plt
import h5py
import scipy
from scipy import sparse
import seaborn as sns; sns.set()
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.linear_model import Lasso
from sklearn.mixture import GaussianMixture
import scipy.spatial
from scipy.optimize import linear_sum_assignment
import sklearn.preprocessing
import pandas as pd
import xarray as xr
from datetime import datetime
import dask
from sklearn.neighbors.kde import KernelDensity
from mpl_toolkits import mplot3d
#%% Parameters
data_origin = 'auditory' # I implemented different data loading functions (exclude non-spiking (misha) or exclude neuropil (georges))
thresh = 0.7 # parameter $THRESH from pseudo-algorithm
n_rough_clusters = 20 # number of clusters for 1a
if data_origin == 'misha': # number of clusters for 1b
    data_name_import = 'SampleData_all.h5'
    n_fine_clusters = 150 # due to n_neurons = 27k
elif data_origin == 'georges':
    data_name_import = 'Data_20180131Run03.h5' # h5 data file to import (calcium/coords/times)
    n_fine_clusters = 300 # due to n_neurons = 62k
elif data_origin == 'volker':
    data_name_import = 'dff_rev.h5'
    n_fine_clusters = 300;
elif data_origin == 'auditory':
    data_name_import = 'auditory_20150303Run04.h5'
    n_fine_clusters = 100
discard_thresh_n_items = 10 #  parameter of step 3.b) I have skipped step 2.b) because they are cleaned in 3b anyway (it is only for speed-up). Then I also skipped 3b because I only encountered clusters >=10 cells
export_pdclusters = True # if true; write to pd.Dataframe hdf5 file so it can be loaded in Fishualizer (as static data set)
rescale = True
step1 = 'kmeans' # 'GMM' or 'kmeans'

#%% Load data
if data_origin == 'georges':
    
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

if data_origin == 'auditory':
    def load_data(): # load the SampleData, labels and names 
        F = h5py.File(data_name_import,'r')
        data_temp = {'values': np.transpose(F['Data']['Values'].value,(1,0)), 
                'coords': np.transpose(F['Data']['Coordinates'].value,(1,0)),
                'ref_coords': np.transpose(F['Data']['RefCoordinates'].value,(1,0)),
                'times': np.transpose(F['Data']['Times'].value,(1,0))}
        return data_temp
   
    data = load_data() # load into Data    
    
    length_data = len(data['values'])  
    data_use = xr.DataArray(data = data['values'],
                           coords={'neurons': np.arange(len(data['values'])),
                                   'times': np.squeeze(data['times'])},
                                   dims=['neurons', 'times'])
    data_use.set_index(np.arange(len(data_use)))

if data_origin == 'misha':
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

if data_origin == 'volker':
    def load_data():
        F = h5py.File(data_name_import,'r')
        data_temp = {'values': np.transpose(F['Data']['Values'].value,(1,0)), 
                'coords': np.transpose(F['Data']['Coordinates'].value,(1,0))}
        return data_temp
    data = load_data()
    length_data = len(data['values'])
    data['times'] = np.arange(data['values'].shape[1])
    data_use = xr.DataArray(data=data['values'], 
                            coords={'neurons': np.arange(length_data), 'times': np.arange(data['values'].shape[1])},
                            dims=['neurons', 'times'])
    data_use.set_index(np.arange(len(data_use)))
    
if rescale == True:
    for feature in range(data_use.shape[1]): # loop through columns/time points/features
        data_use[:, feature] = sklearn.preprocessing.scale(data_use[:, feature]) # scale to zero mean, unit variance
        
    
#%% Functions
def correlation(sig_funvox, sigs_neuron): # correlation function from Rémi (local.py)
    """
    Pearson coefficient of correlation between the calcium signals of two neurons
    Calculated manually to be faster in a 1 vector x 1 matrix
    TODO: Better measure?

    Parameters
    ----------
    sig_funvox    1 signal
    sigs_neurons  multiple signals

    Returns
    -------
    r: float
        Coefficient of correlation
    """
    df1 = sig_funvox
    df2 = sigs_neuron.transpose()
    cov = np.dot(df1 - df1.mean(), df2 - df2.mean(axis=0)) / (df2.shape[0] - 1)
    # ddof=1 necessary because covariance estimate is unbiased (divided by n-1)
    p_var = np.sqrt(np.var(df1, ddof=1) * np.var(df2, axis=0, ddof=1))
    r = cov / p_var
    return r     

#%% 1. Divide all cells into “functional voxels” (~10 cells each)
fun_vox_neurons = {}
fun_vox_count = 0
#    a. Perform k-means clustering on all cells (k=20)
if step1 == 'kmeans':
    #rough_kmeans = KMeans(n_clusters=n_rough_clusters, max_iter=100, precompute_distances=True, 
    #                      verbose=1, n_jobs=-2) # regular KMeans
    rough_kmeans = MiniBatchKMeans(n_clusters=n_rough_clusters, verbose=1)## Use MiniBatchKmeans instead for speed-up (is that sound?)
    rough_kmeans.fit(data_use)
    rough_labels = rough_kmeans.labels_
    print('Rough kmeans completed')
elif step1 == 'GMM':
    rough_GMM = GaussianMixture(n_components=n_rough_clusters, verbose=1, covariance_type='diag') # covariance_type?
    rough_GMM.fit(data_use)
    rough_labels = rough_GMM.predict(data_use)
    print('Rough GMM completed')
    
#   b. Perform k-means clustering on outputs of (a) (k = ~400)
for rough_cluster in range(n_rough_clusters): 
    print(f'rough cluster id: {rough_cluster}')
    rough_c_neurons = np.where(rough_labels == rough_cluster)[0]
    if len(rough_c_neurons) < n_fine_clusters: # discard?
        print(f'Only {len(rough_c_neurons)} in cluster, discarded')
        continue
    rough_c_neurons = np.squeeze(rough_c_neurons) # do not squeeze before if len(rough_c_neurons) in case of len  == 1 TypeError
   
    if step1 == 'kmeans':
        fine_kmeans = KMeans(n_clusters=n_fine_clusters, n_init=1, max_iter=100, 
                             precompute_distances=True, verbose=0, n_jobs=-2)
        data_fit_temp = np.squeeze(np.array(data_use[list(rough_c_neurons), :]))
        fine_kmeans.fit(data_fit_temp)
        fine_labels = fine_kmeans.labels_
        print('Fine kmeans completed')
    elif step1 == 'GMM':
        fine_GMM = GaussianMixture(n_components=n_fine_clusters, covariance_type='diag')
        data_fit_temp = np.squeeze(np.array(data_use[list(rough_c_neurons), :]))
        fine_GMM.fit(data_fit_temp)
        fine_labels = fine_GMM.predict(data_fit_temp)
        print('Fine GMM completed')
        
    for fine_cluster in range(n_fine_clusters):
         #   c. Discard any cells whose correlation with the voxel average activity is less than $THRESH
        fine_c_neurons = np.where(fine_labels == fine_cluster)[0]
        fun_vox_act = data_use[rough_c_neurons[fine_c_neurons], :].mean(dim='neurons') # mean activity
        correlation_ew = correlation(sig_funvox=fun_vox_act, sigs_neuron=data_use[rough_c_neurons[fine_c_neurons], :]) # correlate with voxel average
        condition_corr = np.array(correlation_ew >= thresh)
         #  d. Discard any voxels with fewer than 5 cells
        if sum(condition_corr) < 5:
            continue
        
        else: # only use cells that have above threshold correlation
            fun_vox_neurons[fun_vox_count] = data_use[rough_c_neurons[fine_c_neurons[condition_corr]], :] # only use > thresh corr
            fun_vox_count = fun_vox_count + 1
            
print(f'Number of functional voxels: {fun_vox_count}')
fun_voxels = np.zeros((fun_vox_count, len(np.squeeze(data['times'])))) # create definite functional voxels
fun_voxel_coords = np.zeros((fun_vox_count, 3))
for x in range(fun_vox_count):
    fun_voxels[x] = fun_vox_neurons[x].mean(dim='neurons') # take the av. act.
    fun_voxel_coords[x] = np.mean(data['coords'][fun_vox_neurons[x].neurons, :], axis=0) # mean of coords
fun_voxels = xr.DataArray(data=fun_voxels, coords={'fun_voxels': np.arange(fun_vox_count),
                                                   'times': np.squeeze(data['times'])},
                          dims=['fun_voxels', 'times']) # put in data array

#%% 2. Merge voxels into clusters based on density in functional space
 """
    a. for each pair of voxels ij (starting from most correlated):
        if the correlation between voxel i and j is greater than $THRESH,
        and the correlation between the the voxel j and the centroid (average) of the cluster
        containing voxel i is greater than $THRESH:
        then group voxel j in the same cluster as i.
    """
def step_two_three(fun_voxels_f, fun_vox_neurons_f):
    """
    I put the entire thing in 1 function, because it has to be repeated in step 4
    The structure of the code is therefore not particularly pleasing..
    """
    corrmat = np.corrcoef(fun_voxels_f)  # pairwise correlation matrix
    corrmat_cond = corrmat > thresh   # bool matrix of all pairwise correlations > thresh
    ind_uptriu = np.triu_indices_from(corrmat, k=1) # indices of right upper triangle of matrix
    high_connections = np.where(corrmat_cond[ind_uptriu])[0] # pairwise correlations > thresh in 1D indexing format
    hc_rows = ind_uptriu[0][high_connections] # row of pairwise corr
    hc_cols = ind_uptriu[1][high_connections] # column of pairwise corr
    hc_values = corrmat[hc_rows, hc_cols] # value of pairwise corr
    hcv_sorted = np.flipud(np.sort(hc_values)) # sorted values of pairwise corr (high to low)
    hcv_argsorted = np.flipud(np.argsort(hc_values)) # argsorted values of pairwise corr (high to low)
        
    clusters = {0: set([hc_rows[hcv_argsorted[0]], hc_cols[hcv_argsorted[0]]])}# start with the highest corr
    new_cluster_count = 1
    for ind_n, n in enumerate(hcv_argsorted): # loop through pairwise combinations with pairwise corr > thresh
        if ind_n % 1000 == 0:
            print(f'Progress {ind_n} of {len(hcv_argsorted)} pairwise combinations')
        make_new = True # enable possibility for new cluster (default)
        for c, vals in clusters.items(): # loop through all existing clusters
            if hc_rows[n] in vals: # if row in a cluster, check if column can be added
                if correlation(fun_voxels_f[hc_cols[n]], fun_voxels_f[[nc for nc in clusters[c]], :].mean(dim='fun_voxels')) > thresh:
                    clusters[c].add(hc_cols[n])
                    make_new = False # if added, disable possibility to make a new cluster
                    break
                    
            elif hc_cols[n] in vals: # if column in a cluster, check if row can be added
                if correlation(fun_voxels_f[hc_rows[n]], fun_voxels_f[[nc for nc in clusters[c]], :].mean(dim='fun_voxels')) > thresh:
                    clusters[c].add(hc_rows[n])
                    make_new = False # if added, disable possibility to make a new cluster
                    break
                
        if make_new == True: # if pairwise combination n was not added to an existing cluster, make a new one
            clusters[new_cluster_count] = set([hc_rows[n], hc_cols[n]]) # add row and column
            new_cluster_count = new_cluster_count + 1
    
            
    #  b. discard any clusters with fewer than 10 cells
    clustered_funvoxs = set()
    clustered_cells = set()
    n_clustered_funvoxs = 0
    n_clustered_cells = 0
    cluster_sizes = {}
    cluster_coords = np.zeros((len(clusters), 3))
    cluster_centroid_activity = {}
    clusters_cells = {} # now define clusters in cells, not in funct. voxels.
    for c, vals in clusters.items():
        clusters_cells[c] = set() # set of neuron indices
        for fv in vals: # loop through functional voxels
             clusters_cells[c].update(np.array(fun_vox_neurons_f[fv].coords['neurons'])) # add cells of fv to cluster
        clustered_cells.update(clusters_cells[c])
        clustered_funvoxs.update(vals)
        n_clustered_cells = n_clustered_cells + len(clusters_cells[c])
        n_clustered_funvoxs = n_clustered_funvoxs + len(vals) # count the number of clustered functional voxels
        cluster_sizes[c] = len(clusters_cells[c]) # count cluster size in cells
        cluster_coords[c] = np.mean(data['coords'][list(clusters_cells[c]), :], axis=0) # get mean coordinates of mean 
        cluster_centroid_activity[c] = np.mean(data['values'][list(clusters_cells[c]), :], axis=0) # get mean activity
        
        ### clusters are not discarded.
        #  this also happen sin step 3, so it is only done here for speed-up
        
#    plt.figure(1)
#    plt.hist([cluster_sizes[x] for x in cluster_sizes], 30) # plot cluster sizes (of functional voxels!)
#    plt.xlabel('cluster size')
#    plt.ylabel('number of clusters')
    
    print(f'{len(clustered_cells)} unique cells out of {len(data_use)} have been clustered ({n_clustered_cells} clustered in total)')
    print(f'{len(clustered_funvoxs)} unique functional voxels out of {len(fun_voxels_f)} have been clustered ({n_clustered_funvoxs} clustered in total)')
    
    #% 3. Clean up clusters using regression to cluster centroids
      #  a. for each cell k:
       #     if the correlation to the closest cluster’s centroid is greater than $THRESH:
        #    include cell k in that cluster
        
    # I think they mean to use all cells, and that it is possible to include a new neuron
    # in multiple clusters. That would mean it is embedded in multiple circuits or so
    
    clean_count = 0
    def clean_cell(n):
        coords_cell = data['coords'][n,:]
        distances = [np.linalg.norm(coords_cell - cc) for cc in cluster_coords] # distances to cluster centers
        nearest_cluster = np.argmin(distances) # closest cluster   
        corr_nc = correlation(data['values'][n, :], cluster_centroid_activity[nearest_cluster]) # correlation between activities
        if corr_nc > thresh:
            clusters_cells[nearest_cluster].add(n)
            cluster_coords[c] = np.mean(data['coords'][list(clusters_cells[nearest_cluster]), :], axis=0) # recompute coord mean
            cluster_centroid_activity[c] = np.mean(data['values'][list(clusters_cells[nearest_cluster]), :], axis=0) # recompute activity mean
#            print(f'Neuron {n} added to cluster {nearest_cluster} with corr {np.round(corr_nc, 3)}')
#        if clean_count % 5000 == 0:
#            print(f'iteration number {clean_count}')
#        clean_count += 1
        
    ## This line below is expensive, (also due to printing..?) -> dask parallelize this.
    # Also it seems like some cluster are extremely big and add very many neurons in this step.
    #run = [clean_cell(int(nx)) for nx in data_use.coords['neurons']]   # I decided to use data_use instead of (all) data['values'], because
    values = [dask.delayed(clean_cell)(int(nx)) for nx in data_use.coords['neurons']]
    run = dask.compute(values)  
    
        # b. Discard any clusters with fewer than 10 cells
        # This has not happened to me so far (clusters with <10 cells)
        # otherwise use cluster_above_thresh to select.
    cluster_sizes_3 = [len(vals) for c, vals in clusters_cells.items()]
    cluster_above_thresh = [(x >= discard_thresh_n_items) for x in cluster_sizes_3]
    print(f'{sum(cluster_above_thresh)} out of {len(cluster_above_thresh)} clusters have more than {discard_thresh_n_items} cells')
#    plt.figure(2)
#    plt.hist([cluster_sizes_3[x] for x in cluster_sizes], 30)
#    plt.xlabel('cluster size')
#    plt.ylabel('number of clusters')
#    
    ## go to the correct format to put back into function step_two_three()
    clusters_centroid_activity_RF = np.zeros((len(cluster_centroid_activity), len(np.squeeze(data['times']))))
    cluster_neurons = {}
    data_use_f = xr.DataArray(data=data['values'], 
                            coords={'neurons': np.arange(len(data['values'])), 'times': np.squeeze(data['times'])},
                            dims=['neurons', 'times']) # this is quite tedious and wastes memory (large matrix). 
    # It is a quickfix for redefining fun_vox_neurons, now called cluster_neurons
    # to re-use in step 2-3. 
    for c in clusters_cells:
        clusters_centroid_activity_RF[c, :] = cluster_centroid_activity[c]
        cluster_neurons[c] = data_use_f[list(clusters_cells[c]), :]
    del data_use_f # free memory again    
    clusters_centroid_activity_RF = xr.DataArray(data=clusters_centroid_activity_RF, 
                                             coords={'fun_voxels': np.arange(len(clusters_centroid_activity_RF)), 
                                                     'times': np.squeeze(data['times'])},
                                             dims=['fun_voxels', 'times'])
    
    return clusters_cells, clusters_centroid_activity_RF, cluster_coords, cluster_neurons
        
    #%%    4. Iterate merge and cleanup steps
    #a. Perform step 2 and 3 once more, using clusters as input voxels.
    
# first time (step 2/3)
clusters_cells_1, clusters_centroid_activity_RF_1, cluster_coords_1, cluster_neurons_1 = step_two_three(fun_voxels_f=fun_voxels, fun_vox_neurons_f=fun_vox_neurons)
cluster_sizes_1 = [len(vals) for c, vals in clusters_cells_1.items()]


        
# second time (step 4)
clusters_cells_2, clusters_centroid_activity_RF_2, cluster_coords_2, cluster_neurons_2 = step_two_three(fun_voxels_f=clusters_centroid_activity_RF_1, fun_vox_neurons_f=cluster_neurons_1)
cluster_sizes_2 = [len(vals) for c, vals in clusters_cells_2.items()]




#%% Export
"""
Sparse matrices might be nice for plt.spy() inspection
Pandas matrices contain CORRELATION values to cluster mean as values -> quality of individual clustering.
TODO: also do this for sparse matrices?
Pandas matrices can be fishualized. 
"""

n_cells = len(data['values'])
#cluster_sparse_matrix_1 = scipy.sparse.lil_matrix(np.zeros((n_cells, len(cluster_sizes_1))))
#cluster_pd_matrix_1 = pd.DataFrame({})
#for cl in clusters_cells_1:
#    cells = list(clusters_cells_1[cl])
#    cluster_sparse_matrix_1[cells, cl] = 1
#    col_name = 'cluster {}, n_neurons {}'.format(cl, cluster_sizes_1[cl])
#    cluster_pd_matrix_1[col_name] = np.zeros(n_cells)
#    cluster_pd_matrix_1.loc[cells, cl] = 1

cluster_sparse_matrix_2 = scipy.sparse.lil_matrix(np.zeros((n_cells, len(cluster_sizes_2)))) # put clusters in sparse format
cluster_pd_matrix_2 = pd.DataFrame({}) # put clusters in pandas format for exporting
for cl in clusters_cells_2: # cluster_2 is the end product.
    cells = list(clusters_cells_2[cl]) # cells (i.e. neurons) in cluster
    cluster_sparse_matrix_2[cells, cl] = 1
    col_name = 'cluster {}, n_neurons {}'.format(cl, cluster_sizes_2[cl])
    cluster_pd_matrix_2[col_name] = np.zeros(n_cells)
    ## Below: use absolute value for now, needs change (in fishualizer) to be able to get negative corre to cluster mean
    cluster_pd_matrix_2.loc[cells, col_name] = np.abs(correlation(np.array(clusters_centroid_activity_RF_2[cl]), data['values'][cells, :]))

if export_pdclusters == True:
    ename = f"chen2018_clusters_{datetime.now().strftime('%Y-%m-%d_%H_%M')}.h5"
    store_df = pd.HDFStore(ename)
    store_df['cluster_pd_matrix_2'] = cluster_pd_matrix_2
    store_df.close()
    
    np.save('clusters_cells', clusters_cells_2) # heat map of correlation coefs between cluster means
    np.save('clusters_centroid_act_RF', clusters_centroid_activity_RF_2)
    np.save('cluster_coords', cluster_coords_2)

load_clusters = False
if load_clusters:
    clusters_cells_2 = np.load('clusters_cells.npy')
    clusters_centroid_activity_RF_2 = np.load('clusters_centroid_act_RF.npy')
    cluster_coords_2 = np.load('cluster_coords.npy')
    clusters_cells_2 = clusters_cells_2[()]
    
    clusters_cells_other = np.load('clusters_cells.npy')
    clusters_cells_other = clusters_cells_other[()]
#%% Plotting
    
plt.figure(4) # spy matrix
plt.subplot(1,2,1)
plt.spy(cluster_sparse_matrix_2, aspect='auto', markersize=.5)

plt.subplot(1,2,2) # cluster sizes of all clusters
plt.plot(cluster_sizes_2)

plt.figure(5) # heat map of correlation coefs between cluster means
sns.heatmap(np.corrcoef(clusters_centroid_activity_RF_2))

ind_start = 93# cluster start S
plt.close(2) 
plt.figure(2)
plt.subplot(2,1,1)
time_trace = np.arange(len(clusters_centroid_activity_RF_2[0,:]))
for k in range(ind_start, ind_start + 5): # inspect some clusters
    plt.plot(time_trace, clusters_centroid_activity_RF_2[k, :])

plt.subplot(2,1,2)
n_plot_cells = 10 # inspect some cells
plot_cells = np.random.choice(np.array(list(clusters_cells_2[ind_start])), n_plot_cells)
for k in range(n_plot_cells):
    plt.plot(time_trace, data['values'][plot_cells[k], :])


plt.close(3)
plt.figure(3) # inspect a cluster with std 
std_sig = np.std(data['values'][np.array(list(clusters_cells_2[ind_start])), :], axis=0)
mean_sig = np.mean(data['values'][np.array(list(clusters_cells_2[ind_start])), :], axis=0)
plt.plot(time_trace, mean_sig - std_sig, 'r', time_trace, mean_sig, 'b', time_trace, mean_sig + std_sig, 'r')

#%% Some analysis
def mse(x, y):
        mse_val = ((x - y) ** 2).mean()
        return mse_val
    
def rescale_substract(cl=0, n=0): # subtract a rescaling of cluster mean of each neuron
    scalings = [0.1, 0.25, 0.4, 0.6, 0.8, 0.9, 1, 1.1, 1.3, 1.5, 2, 3, 4] # rescaling to consider
    c_sig = clusters_centroid_activity_RF_2[cl]
    n_sig = data['values'][n, :]
    best = [None, np.inf]
    for sc in scalings:
        new_mse = mse(n_sig, (sc * c_sig))
        if new_mse < best[1]:
            best[0] = sc
            best[1] = new_mse
        
        new_mse = mse(n_sig, (-1* sc * c_sig))
        if new_mse < best[1]:
            best[0] = -1 * sc
            best[1] = new_sme
    
    best_rescale = n_sig - (best[0] * c_sig)
    return best_rescale

def cluster_substract(cls=0):
    cl_neurons = list(clusters_cells_2[cls])
    cl_neurons_substracted = np.zeros_like(data['values'][cl_neurons, :])
    for iloop, neuron in enumerate(cl_neurons):
        cl_neurons_substracted[iloop, :] = rescale_substract(cl=cls, n=neuron)
    return cl_neurons_substracted

testcluster=22
test_sub = cluster_substract(cls=testcluster)

def plot_var(cl_sub_data, cl):
    variance_original = data['values'][list(clusters_cells_2[cl]), :].var(axis=1)
    variance_substracted = cl_sub_data.var(axis=1)
    plt.figure(12)
    plt.hist((variance_substracted / variance_original), 50)
    plt.xlabel('Variance substracted / variance original')
    plt.show()
plot_var(test_sub, testcluster)
    
def regress_cl(cl_sub_data, cl):
    def lasso_all(reg_data, n_rel, n_neurons, alpha_val=0.005):
        # use transposed data as input
        mask = np.ones(n_neurons, dtype = 'bool')
        mask[n_rel] = False
        model = Lasso(alpha=alpha_val)
        model.fit(X=reg_data[:, mask], y=reg_data[:, n_rel])
        coefficients = np.zeros(n_neurons)
        coefficients[mask] = model.coef_
        return coefficients#, model.score(X=data_use[:, mask], y=data_use[:, n])
    
    data_original = data['values'][list(clusters_cells_2[cl]), :].transpose()
    n_neurons_cluster = len(clusters_cells_2[cl])
    data_substracted = cl_sub_data.transpose()
    
    prelim_original = [dask.delayed(lasso_all)(data_original, iloop, n_neurons_cluster) for iloop in range(n_neurons_cluster)]
    coef_mat_original = dask.compute(prelim_original)
    prelim_substracted = [dask.delayed(lasso_all)(data_substracted, iloop, n_neurons_cluster) for iloop in range(n_neurons_cluster)]
    coef_mat_substracted = dask.compute(prelim_substracted)
    
    coef_mat_original = scipy.sparse.csr_matrix(np.squeeze(coef_mat_original))
    coef_mat_substracted = scipy.sparse.csr_matrix(np.squeeze(coef_mat_substracted))
    return coef_mat_original, coef_mat_substracted

testcm_or, testcm_sub = regress_cl(test_sub, testcluster)

plt.close(43)
plt.figure(43)
plt.title('coefficient matrices')
plt.subplot(1,2,1)
plt.title('original')
#sns.heatmap(testcm_or[0])
plt.spy(testcm_or, markersize=2, aspect='auto')

plt.subplot(1,2,2)
plt.title('subtracted')
#sns.heatmap(testcm_sub[0])
plt.spy(testcm_sub, markersize=2, aspect='auto')
plt.show()


def plot_1neuron_twice(or_sig, sub_sig, cl):
    plt.close(44)
    plt.figure(44)
    plt.subplot(2,1,1)
    plt.title('original (x) versus subtracted (y)')
    plt.plot(or_sig, sub_sig, '.')
    
    plt.subplot(2,1,2)
    plt.title('time vs or and sub')
    time_trace = np.arange(len(or_sig))
    plt.plot(time_trace, or_sig, time_trace, sub_sig)
    plt.legend(['original', 'subtracted'])
    
    plt.close(45)
    plt.figure(45)
    plt.title('cluster mean activity')
    plt.plot(time_trace, clusters_centroid_activity_RF_2[cl])
    
    plt.close(46)
    plt.figure(46)
    plt.title('difference or and sub')
    plt.plot(time_trace, or_sig - sub_sig)

nplot=4
plot_1neuron_twice(or_sig=data['values'][list(clusters_cells_2[15])[nplot], :], sub_sig=test_sub[nplot, :], cl=testcluster)






#%% CLUSTER COMPARISON

#%% Match clusters of SAME FISH (i.e. same set of neurons)
def count_similarity_set(set1, set2):
    """
    Return fraction of common elements as defined below
    """
    common_elements = set1.intersection(set2)
    fraction_common = 2 * len(common_elements) / (len(set1) + len(set2))
    return fraction_common

def make_fraction_matrix_dict(dict1, dict2):
    """
    Create matrix with all pairwise fractions of common elements
    """
    n_clusters1 = len(dict1)
    n_clusters2 = len(dict2)
    fraction_mat = np.zeros((n_clusters1, n_clusters2))
    
    for i1 in range(n_clusters1):
        for i2 in range(n_clusters2):
            if (len(dict1[i1]) > 0) and (len(dict2[i2]) > 0): # if elements in clusters
                ## assuming value of dict is set of neurons in cluster key
                fraction_mat[i1, i2] = count_similarity_set(dict1[i1], dict2[i2])
    
    return fraction_mat

def find_mapping(fraction_matrix):
    """
    Apply hungarian algorithm to find minimal mapping between two clusterizations
    scipy.optimization.linear_sum_assignment solves for minimal dimensions (i.e. lowest x lowest in case of lxh or hxl)
    """
    cost_mat = 1 - fraction_matrix # define cost as 1 - fraction of common elements
    rows, cols = scipy.optimize.linear_sum_assignment(cost_mat)
    return rows, cols # sorted order
    
def sort_heatmap(fraction_matrix, rows, cols):
    """
    Plot sorted heatmap
    Not very useful at the moment because you have to look up the original indices
    by hand
    """
    min_shape = np.min(fraction_matrix.shape)
    newmat = np.zeros((min_shape, min_shape)) # min size!!
    for irow in range(min_shape):
        for icol in range(min_shape):
            newmat[irow, icol] = fraction_matrix[rt[irow], ct[icol]]
    mean_fract = sum([newmat[k, k] for k in range(min_shape)]) / min_shape

    plt.figure(2)
    plt.title('heatmap of cluster comparison fractions - sorted \n'
              f'mean fraction is {np.round(mean_fract, 3)}')
    sns.heatmap(newmat)
    return newmat

fract_mat = make_fraction_matrix_dict(clusters_cells_2, clusters_cells_other) # fraction of common elements matrix
plt.figure(3)
plt.title('heatmap of cluster comparison fractions')
sns.heatmap(fract_mat) # plot full fraction matrix
rt, ct = find_mapping(fract_mat) # sort to find minimal mapping
final_fractions = fract_mat[rt, ct]
sorted_fract_mat = sort_heatmap(fract_mat, rt, ct) # plot minimal mapping
#sum_fracts = sum([sorted_fract_mat[k, k] for k in range(len(sorted_fract_mat))]) / len(sorted_fract_mat)


######
#%% Match clusters of DIFFERENT FISH (i.e. different neurons)
"""
Needed
 - Coordinates of fish in same 'reference' space - > the reference coordinates
 - clusters in format as used above (i.e. dictionary with cluster ids as keys, and set of neuron id belonging to cluster as values)
 
Method
Create rectangular grid in reference space
For all clusters;
 - Kernel density approximation (Gaussian) of reference coords of neurons in cluster
 - Compute sample scores of rectulangar grid of this KDE
TODO: - Take element-wise dot product of scores between two fish & Normalize

"""

## Import previously saved clusters:
data_name_import = 'auditory_20150303Run04.h5'
data_run4 = load_data()

clusters_cells_run4 = np.load('clusters_cells.npy')
clusters_cells_run4 = clusters_cells_run4[()]

data_name_import = 'auditory_20150303Run07.h5'
data_run7 = load_data()

clusters_cells_run7 = np.load('clusters_cells.npy')
clusters_cells_run7 = clusters_cells_run7[()]


def make_ref_coords(coords1, coords2, n_voxels_approx = 100000):
    """
    Create equally spaced rectangular grid of ref coords
    """
 # use ref coords!
    min1 = coords1.min(axis=0)
    min2 = coords2.min(axis=0)
    max1 = coords1.max(axis=0)
    max2 = coords2.max(axis=0)
    min_gen = np.zeros(3)
    max_gen = np.zeros(3)
    axis_arrays = {}
    for i_dim in range(3): # get absolute min and max
        min_gen[i_dim] = np.minimum(min1[i_dim], min2[i_dim])
        max_gen[i_dim] = np.maximum(max1[i_dim], max2[i_dim])

    volume = (max_gen[0] - min_gen[0]) * (max_gen[1] - min_gen[1]) * (max_gen[2] - min_gen[2])
    scaling_factor = np.cbrt(volume/n_voxels_approx) # because then V = N * scale_factor^3
    for i_dim in range(3):
        axis_arrays[i_dim] = np.arange(min_gen[i_dim], max_gen[i_dim], scaling_factor) # create axis array
            
    gridx, gridy, gridz = np.meshgrid(axis_arrays[0], axis_arrays[1], axis_arrays[2], indexing='ij')
    n_voxels = gridx.shape[0] * gridy.shape[1] * gridz.shape[2]
    grid = np.zeros((n_voxels, 3))
    grid[:, 0] = np.squeeze(gridx.reshape((n_voxels, 1))) # reshape to common format
    grid[:, 1] = np.squeeze(gridy.reshape((n_voxels, 1)))
    grid[:, 2] = np.squeeze(gridz.reshape((n_voxels, 1)))
    return grid

def cluster_density(coords, refcoords=reference_coords, bw = 0.004): 
    # bandwith now hand tuned, depends on rectangular grid resolution. 
    # bandwith also depends on user; how important are 'lonely' neurons in the clusters?
    # scipy KDE approximation
    kde = KernelDensity(kernel='gaussian', bandwidth=bw).fit(coords)
#    den = np.exp(kde.score_samples(refcoords))
    den = kde.score_samples(refcoords) # return log scores
    return den

def transfer_clusters(cl_cells, local_coords, new_coords):
    """
    Transfer clusters from one fish to rectangular grid in reference space
    """
    n_clusters = len(cl_cells) # assuming cl_cells is a dict
    n_cells_ref = new_coords.shape[0]
    cl_new_neurons = pd.DataFrame(np.zeros((n_cells_ref, n_clusters)), columns=[f'cluster_{x}' for x in range(n_clusters)]) # output format of grid units
    for i_cl in range(n_clusters):
        coords_cl_neurons = local_coords[np.array(list(cl_cells[i_cl])), :]
        densities = cluster_density(coords=coords_cl_neurons, refcoords=new_coords)
        cl_name = f'cluster_{i_cl}'
        cl_new_neurons[cl_name] = densities # use soft assignment 
    return cl_new_neurons

reference_coords = make_ref_coords(data_run4['ref_coords'], data_run7['ref_coords'], n_voxels_approx=400000) # create rectangular grid

clusters_run4_coords_ref = transfer_clusters(cl_cells=clusters_cells_run4, local_coords=data_run4['ref_coords'], new_coords=reference_coords)


###### Validation:


## Store to view in FIshualizer: (compare rectangular grid to actual (ground truth) fish)
store_df = pd.HDFStore('KDE_cl_run4.h5') 
store_df['KDE_cl_run4'] = clusters_run4_coords_ref # save clusters
store_df.close()

fake_values = np.zeros((clusters_run4_coords_ref.shape[0], 100))
fake_values = np.random.rand(clusters_run4_coords_ref.shape[0], 100)
hf = h5py.File('grid_run4.h5', 'w') # create a hdf5 set to plot, coords and dff needed
hg = hf.create_group('Data')
hg.create_dataset('coords', data=reference_coords.transpose())
hg.create_dataset('dff', data=fake_values.transpose())
hf.close()

## compare as fraction matrix:

clusters_run7_coords_ref = transfer_clusters(cl_cells=clusters_cells_run7, local_coords=data_run7['ref_coords'], new_coords=reference_coords) # create two mappings to reference grid

cl_run4_coords_ref_dict = {}
cl_run7_coords_ref_dict = {}
for i_cl in range(clusters_run7_coords_ref.shape[1]):
    cl_name = f'cluster_{i_cl}'
    cl_run7_coords_ref_dict[i_cl] = set(np.where(clusters_run7_coords_ref[cl_name])[0])
for i_cl in range(clusters_run4_coords_ref.shape[1]):
    cl_name = f'cluster_{i_cl}'
    cl_run4_coords_ref_dict[i_cl] = set(np.where(clusters_run4_coords_ref[cl_name])[0])
    
fract_mat_4_7 = make_fraction_matrix_dict(cl_run4_coords_ref_dict, cl_run7_coords_ref_dict)
rt, ct = find_mapping(fract_mat_4_7)
final_fractions_4_7 = fract_mat_4_7[rt, ct]
sorted_fract_mat_4_7 = sort_heatmap(fract_mat_4_7, rt, ct)
plt.figure()
sns.heatmap(fract_mat_4_7)
print(fract_mat_4_7.argmax(axis=0))
print('otherside')
print(fract_mat_4_7.argmax(axis=1))






