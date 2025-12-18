import pandas as pd
import numpy as np

"""Algorithm of DPLloyd introduced by 
[1] A. Blum, C. Dwork, F. McSherry, and K. Nissim, 
“Practical privacy: the SuLQ framework,” 
in Proc. ACM Symp. Prin. Database Syst. (PODS). New York, NY, USA: ACM, 
Jun. 2005, pp. 128–138. doi: 10.1145/1065167.1065184.

We follow the pseudocode (Algorithm 1) of 
[2] D. Su, J. Cao, N. Li, E. Bertino, M. Lyu, and H. Jin, 
“Differentially private k-means clustering and a hybrid approach to private optimization,” 
ACM Trans. Priv. Secur., vol. 20, no. 4, p. 16:1-16:33, Oct. 2017, doi: 10.1145/3133201."""


def Laplace_noise(epsilon: float=1, sensitivity=1):
    if epsilon<=0:
        return float('nan')
    else: 
        return np.random.laplace(0, scale=sensitivity/epsilon)

def norm_calculation(row1,row2):
    square_sum = 0 
    for col in range(len(row1)):
        square_sum = square_sum + (row1[col]-row2[col])**2
    return np.sqrt(square_sum)

def RandomCentroidGeneration(number_clusters,number_dimensions,normalized_range_value): 
    centroid_list = [ [] for i in range(number_clusters)]
    for i in range(number_clusters):
        for j in range(number_dimensions):
            centroid_list[i].append(np.random.uniform(-normalized_range_value,normalized_range_value))

    return centroid_list

def GenerateClusters(database,centroids):
    cols = list(range(len(database[0]))) 
    number_clusters=len(centroids)

    clusters_list = [ [] for i in range(number_clusters)]
    for record in database:
        closest_centroid = clusters_list[0]
        closest_centroid_distance = 0
        for col in cols:
            closest_centroid_distance += (record[col] - centroids[0][col]) ** 2
        for cluster, centroid in zip(clusters_list[1:], centroids[1:]):
            distance = 0
            for col in cols:
                distance += (record[col] - centroid[col]) ** 2
            if distance < closest_centroid_distance:
                closest_centroid = cluster
                closest_centroid_distance = distance
        
        #Append elemenet to cluster c
        closest_centroid.append(record)

    return clusters_list

def NoisyCentroidUpdate(clusters, number_dimensions, epsilon, normalized_range_value):
    number_centroids = len(clusters)
    centroid_list = [ [] for i in range(number_centroids)]
    for i in range(number_centroids):
        noisy_count = len(clusters[i]) + Laplace_noise(epsilon=epsilon/2, sensitivity=1)
        for col in range(number_dimensions):
            sum_value=0
            for n in range(len(clusters[i])):
                sum_value = sum_value + clusters[i][n][col]
            noisy_sum = sum_value + Laplace_noise(epsilon=epsilon/2, sensitivity=1) 
            centroid_coordinate = noisy_sum/noisy_count
            
            if centroid_coordinate<-normalized_range_value:
                centroid_coordinate = -normalized_range_value
            elif centroid_coordinate>normalized_range_value:
                centroid_coordinate = normalized_range_value

            centroid_list[i].append(centroid_coordinate)

    return centroid_list

def DPLloyd(database, number_dimensions, number_clusters, normalized_range_value, epsilon, number_iterations_DPLloyd, initial_set_centroids=None):
    ##If the epsilon is an impossible value (or 0), we avoid the computations
    if(np.isnan(epsilon) or epsilon==0):
        return [ [float('nan') for i in range(number_dimensions)] for j in range(number_clusters)]

    if initial_set_centroids==None:
        centroids = RandomCentroidGeneration(number_clusters=number_clusters,number_dimensions=number_dimensions,normalized_range_value=normalized_range_value)
    else:
        centroids = initial_set_centroids

    epsilon_instance = epsilon/((number_dimensions*number_clusters+1)*number_iterations_DPLloyd)
    for i in range(number_iterations_DPLloyd):
        ##We compute all the clusters and centroids (note that Su et al.'s algorithm computes cluster and then the centroid for each dimension, instead of computing all clusters and then all centroids)
        clusters = GenerateClusters(database=database,centroids=centroids) 
        centroids = NoisyCentroidUpdate(clusters=clusters,number_dimensions=number_dimensions,epsilon=epsilon_instance,normalized_range_value=normalized_range_value)

    return centroids