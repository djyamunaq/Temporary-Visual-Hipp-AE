import numpy as np
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
import seaborn as sb


def ratemaps(embeddings, position, n_bins=50, filter_width=2, occupancy_map=[], n_bins_padding=0):
    '''
    Creates smooth ratemaps from latent embeddings (activity) and spatial position through time.

    Args:
        embeddings (2D numpy array): 2D matrix latent embeddings through time, with shape (n_samples, n_latent).
        position (2D numpy array): 2D matrix containing the (x,y) spatial position through time, with shape (n_samples, 2).
        n_bins (int; default=50): resolution of the (x,y) discretization of space from which the ratemaps will be computed.
        filter_width (float; default=2): standard deviation of the Gaussian filter to be applied (in 'pixel' or bin units).
        occupancy_map (2D numpy array; default=[]): 2D matrix reflecting the occupancy time across the space, with shape (n_bins+2*n_bins_padding, n_bins+2*n_bins_padding).
        n_bins_padding (int; default=0): the number of extra pixels with 0 value that are added to every side of the arena.

    Returns:
        ratemaps (3D numpy array): 3D matrix containing the ratemaps associated to all embedding units, with 
                                   shape (n_latent, n_bins, n_bins).
    '''
    # Normalize position with respect to grid resolution to convert position to ratemap indices.
    pos_imgs_norm = np.copy(position)

    if np.min(pos_imgs_norm[:,0]) < 0:
        pos_imgs_norm[:,0] = pos_imgs_norm[:,0] + np.abs(np.min(pos_imgs_norm[:,0]))
    else:
        pos_imgs_norm[:,0] = pos_imgs_norm[:,0] - np.min(pos_imgs_norm[:,0])

    if np.min(pos_imgs_norm[:,1]) < 0:
        pos_imgs_norm[:,1] = pos_imgs_norm[:,1] + np.abs(np.min(pos_imgs_norm[:,1]))
    else:
        pos_imgs_norm[:,1] = pos_imgs_norm[:,1] - np.min(pos_imgs_norm[:,1])

    max_ = np.max(pos_imgs_norm)
    pos_imgs_norm[:,0] = pos_imgs_norm[:,0]/max_
    pos_imgs_norm[:,1] = pos_imgs_norm[:,1]/max_

    pos_imgs_norm *= n_bins-1
    pos_imgs_norm = pos_imgs_norm.round(0).astype(int)

    occ_prob = occupancy_map/np.sum(occupancy_map)

    # Add activation values to each cell in the ratemap and adds Gaussian smoothing.
    n_latent = embeddings.shape[1]
    ratemaps = np.zeros((n_latent, int(n_bins+2*n_bins_padding), int(n_bins+2*n_bins_padding)))
    for i in range(n_latent):
        ratemap_ = np.zeros((n_bins, n_bins))
        for ii, c in enumerate(embeddings[:,i]):
            indx_x = pos_imgs_norm[ii,0]
            indx_y = pos_imgs_norm[ii,1]
            #ratemaps[i, indx_x, indx_y] += c
            ratemap_[indx_x, indx_y] += c
        ratemaps[i] = np.pad(ratemap_, ((n_bins_padding, n_bins_padding), (n_bins_padding, n_bins_padding)), mode='constant', constant_values=0)
        if np.any(ratemaps[i]):
            ratemaps[i] = np.abs(ratemaps[i])
            ratemaps[i] = ratemaps[i]/np.max(ratemaps[i])
            ratemaps[i] = gaussian_filter(ratemaps[i], filter_width) 
            ratemaps[i] = ratemaps[i]/np.max(ratemaps[i])
            ratemaps[i] = ratemaps[i].T
            if len(occupancy_map) > 0:
                ratemaps[i] = ratemaps[i]/occ_prob
                ratemaps[i] = ratemaps[i]/np.max(ratemaps[i])
        
    return ratemaps


def stats_place_fields(ratemaps, peak_as_centroid=True, min_pix_cluster=0.02, max_pix_cluster=0.5, active_threshold=0.2):
    '''
    Runs a simple clustering algorithm to identify place fields, and compute their number, centroids, and sizes, for all ratemaps.

    Args:
        ratemaps (3D numpy array): 3D matrix containing the ratemaps associated to all embedding units, with shape (n_latent, n_bins, n_bins).
        peak_as_centroid (bool; default=True): if True, the centroid will be taken as the peak of the place field; if False, it will take the 'center of mass'.
        min_pix_cluster (bool; default=0.02): minimum proportion of the total pixels that need to be active within a region to be considered a place field, with a range [0,1].
        max_pix_cluster (bool; default=0.5): maximum proportion of the total pixels that need to be active within a region to be considered a place field, with a range [0,1].
        active_threshold (float; default=0.2): percentage over the maximum activity from which pixels are considered to be active, otherwise they become 0; within a range [0,1].

    Returns:
        all_num_fields (1D numpy array): array with the number of place fields per embedding unit, with shape (n_latent,).
        all_centroids (2D numpy array): array with (x,y) position of all place field centroids, with shape (total_n_place_fields, 2).
        all_sizes (1D numpy array): array with the sizes of all place fields across embedding units, with shape (total_n_place_fields,).
    '''
    all_num_fields = []
    all_centroids = []
    all_sizes = []
    for r in ratemaps:

        ratemap = r.copy()
        
        ## Params.
        total_area = ratemap.shape[0]*ratemaps.shape[1]
        cluster_min = total_area*min_pix_cluster  #50
        cluster_max = total_area*max_pix_cluster #1250
        
        ## Clustering.
        ratemap[ratemap <  ratemap.max()*active_threshold] = 0
        ratemap[ratemap >= ratemap.max()*active_threshold] = 1

        visited_matrix  = np.zeros_like(ratemap)

        # First pass of clustering.
        clusterd_matrix = np.zeros_like(ratemap)
        current_cluster = 1

        # go through every bin in the ratemap.
        for yy in range(1,ratemap.shape[0]-1):
            for xx in range(1,ratemap.shape[1]-1):
                if ratemap[  yy, xx ] == 1:
                    # go through every bin around this bin.
                    for ty in range(-1,2):
                        for tx in range(-1,2):
                            if clusterd_matrix[ yy+ty, xx+tx ] != 0:
                                clusterd_matrix[ yy,xx ] = clusterd_matrix[ yy+ty, xx+tx ]

                    if clusterd_matrix[ yy, xx ] == 0:
                        clusterd_matrix[ yy, xx ] = current_cluster
                        current_cluster += 1
                        
        # Refine clustering: neighbour bins to same cluster number.
        for yy in range(1,clusterd_matrix.shape[0]-1):
            for xx in range(1,clusterd_matrix.shape[1]-1):
                if clusterd_matrix[  yy, xx ] != 0:
                    # go through every bin around this bin.
                    for ty in range(-1,2):
                        for tx in range(-1,2):
                            if clusterd_matrix[ yy+ty, xx+tx ] != 0:
                                if clusterd_matrix[ yy+ty, xx+tx ] != clusterd_matrix[  yy, xx ]:
                                    clusterd_matrix[ yy+ty, xx+tx ] = clusterd_matrix[  yy, xx ]
                  
        ## Quantify number of place fields.
        clusters_labels = np.delete(np.unique(clusterd_matrix), np.where(  np.unique(clusterd_matrix) == 0 ) )
        n_place_fields_counter = 0
        clusterd_matrix_ = np.copy(clusterd_matrix)
        clusters_labels_ = np.copy(clusters_labels)
        for k in range(clusters_labels.size):
            n_bins = np.where(clusterd_matrix == clusters_labels[k])[0].size
            if cluster_min <= n_bins <= cluster_max:
                n_place_fields_counter += 1
            else:
                clusterd_matrix_[np.where(clusterd_matrix_==clusters_labels[k])] = 0
                clusters_labels_ = np.delete(clusters_labels_, np.where(clusters_labels_ == clusters_labels[k]) )

        all_num_fields.append(n_place_fields_counter)
        
        ## Compute centroids.
        centroids = []
        for k in clusters_labels_:
            if peak_as_centroid:  # compute centroid as the peak of the place field.
                x, y = np.unravel_index(np.argmax( r * (clusterd_matrix_==k) ), r.shape)
                #x = np.argmax(  r * (clusterd_matrix_==k) ) 
                #y = np.argmax(  r * (clusterd_matrix_==k) )
            else:  # compute the centroid as weighted sum ('center of mass').
                w_x = r[np.where(clusterd_matrix_==k)[0], :].sum(axis=1)
                w_x = w_x/w_x.sum()
                x = np.sum(w_x * np.where(clusterd_matrix_==k)[0])
                
                w_y = r[:, np.where(clusterd_matrix_==k)[1]].sum(axis=0)
                w_y = w_y/w_y.sum()
                y = np.sum(w_y * np.where(clusterd_matrix_==k)[1])
            centroids.append([x,y])

        all_centroids += centroids
        
        ## Compute sizes of place fields.
        sizes = []
        for k in clusters_labels_:
            n_bins = np.where(clusterd_matrix_ == k)[0].size
            sizes.append(n_bins)

        all_sizes += sizes
    
    return np.array(all_num_fields), np.array(all_centroids), np.array(all_sizes)



def plot_single_ratemap_density(r, unit, all_num_fields, sizes_per_field, centroids_per_field, plot_path, figsize=(3,3), save=False):
    print('Number of place fields = ' + str(all_num_fields[unit]))
    print('Size of place fields = ' + str(sizes_per_field[unit]))
    print('YX position of place fields = ' + str(centroids_per_field[unit]))

    fig = plt.figure(figsize=figsize)
    plt.imshow(r[unit], cmap='hot', origin='lower')
    if centroids_per_field[unit] != [[[0, 0]]]:
        for i in range(len(centroids_per_field[unit])):
            plt.scatter(centroids_per_field[unit][i][1], centroids_per_field[unit][i][0], color='green', marker='x', s=30)
    if save:
        fig.savefig(plot_path + '/Example_place_field.pdf', format='pdf', bbox_inches='tight')
        fig.savefig(plot_path + '/Example_place_field.png', format='png')
    plt.show()
    
    
def plot_place_field_hist(num_fields, plot_path, save=False):
    '''
    TO DO.
    '''
    place_field_counts = np.histogram(num_fields, bins=np.max(num_fields)+1, density=True)[0]
    plt.figure(figsize=(5,4))
    plt.bar(np.arange(np.max(num_fields)+1), place_field_counts, width=1, color='black', alpha=1, edgecolor='white')
    plt.xlabel('# place fields', fontsize=20)
    plt.ylabel('prob.', fontsize=20)
    plt.yticks(np.linspace(0,1,6), np.linspace(0,1,6).round(1), fontsize=18)
    plt.xticks(np.linspace(0, np.max(num_fields), np.max(num_fields)+1, dtype=int), np.linspace(0, np.max(num_fields), np.max(num_fields)+1, dtype=int), fontsize=18)
    plt.ylim(0,1)
    sb.despine()
    plt.tight_layout()
    if save:
        plt.savefig(plot_path + '/prob_place_field_histogram.pdf', format='pdf', bbox_inches='tight')
        plt.savefig(plot_path + '/prob_place_field_histogram.png', format='png')
    plt.show()
    
    
def format_centroids(all_num_fields, centroids, sizes):
    centroids_per_field = []
    sizes_per_field = []
    centroid_index = 0
    
    for i in range(len(all_num_fields)):

        if all_num_fields[i] != 0:
            centroids_append = centroids[centroid_index:centroid_index+all_num_fields[i]].tolist()
            sizes_append = sizes[centroid_index:centroid_index+all_num_fields[i]].tolist()
            centroid_index += all_num_fields[i]
        else:
            centroids_append = [[0, 0]]
            sizes_append = [[0, 0]]


        centroids_per_field.append(centroids_append)
        sizes_per_field.append(sizes_append)
    
    return centroids_per_field, sizes_per_field


def plot_ratemaps(r, plot_path, save=False):
    '''
    TO DO.
    '''
    plt.figure(figsize=(20,20), dpi=600)
    for i in range(100):
        plt.subplot(10, 10, i+1)
        plt.imshow(r[i], cmap='hot', origin='lower')
        plt.axis('off')
    plt.tight_layout()
    if save:
        plt.savefig(plot_path + '/100ratemaps.pdf', format='pdf', bbox_inches='tight')
        plt.savefig(plot_path + '/100units.png', format='png')
    plt.show()
    
