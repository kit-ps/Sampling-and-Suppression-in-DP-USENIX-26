import pandas as pd
import numpy as np
import random

"""Algorithm of the k-Median algorithm (Algorithm 2) in
[1] A. Gupta, K. Ligett, F. McSherry, A. Roth, and K. Talwar, 
“Differentially private combinatorial optimization,” 
in Proc. ACM-SIAM Symp. Discr. Alg. (SODA), Jan. 2010, pp. 1106–1125. doi: 10.1137/1.9781611973075.90."""

def distance(record1,record2):
    square_sum = 0 
    for col1,col2 in zip(record1,record2):
        square_sum = square_sum + (col1-col2)**2
    return np.sqrt(square_sum)

def kmedian(database, data_domain, number_dimensions, number_clusters, value_n, epsilon, sensitivity=1): #Sensitivity is 1 as we have normalized
    ##Turn dataframe into array
    database = np.array(database)

    ##If the epsilon is an impossible value (or 0), we avoid the computations
    if(np.isnan(epsilon) or epsilon==0):
        return float('nan')
        #return [ [float('nan') for i in range(number_dimensions)] for j in range(number_clusters)] #Returns empty cluster

    ##The following is value T in original code
    number_iterations_kmedian = int(6*number_clusters*np.log(value_n))

    ##Compute matrix of distances
    distances_matrix = [ [distance(record1,record2) for record2 in database] for record1 in data_domain]

    centroid_list = []
    ##Create F1 (first random sample)
    centroid0_index = random.sample(range(len(data_domain)),k=number_clusters)
    centroid_list.append(centroid0_index)

    epsilon_instance = epsilon/(2*sensitivity*(number_iterations_kmedian+1))

    exponential_final_weights_list = [] #For the last exponential mechanism in Step 6 

    ##Create F(i+1)=centroid[i+1]
    for i in range(number_iterations_kmedian):

        #Step 4 of Algorithm 2
        exponential_weights_list = []
        nextF_list = []

        for x_index in centroid_list[i]:
            ##Create candidate F (it is more efficient outside the y loop)
            candidate_F_minus_x = centroid_list[i].copy()
            
            ##Remove x (we later add y)
            candidate_F_minus_x.remove(x_index)

            ##Vectors of minimums over F minus x. Computed now to reduce complexity
            vector_of_minimums_over_F_minus_x=[]
            columns_F_minus_x = [distances_matrix[j] for j in candidate_F_minus_x]
            for value_index in range(len(database)):
                distance_minimum_over_F_minus_x = min([col[value_index] for col in columns_F_minus_x])
                vector_of_minimums_over_F_minus_x.append(distance_minimum_over_F_minus_x)

            for y_index in range(len(data_domain)): 
                if y_index in centroid_list[i]: #Items in data_domain but not in centroid_list[i]
                    continue

                ##Remove y from candidate F
                candidate_F = candidate_F_minus_x.copy() 
                candidate_F.append(y_index)

                ##Compute costF as sum of all minimum distances
                costF=0

                for distance_y,minimum_over_F_minus_x in zip(distances_matrix[y_index],vector_of_minimums_over_F_minus_x):
                    ##Distance d(record,F-x+y)
                    minimum_over_F_minus_x_plus_y = min(distance_y,minimum_over_F_minus_x)

                    ##Compute cost
                    costF = costF + minimum_over_F_minus_x_plus_y


                nextF_list.append(candidate_F)
                exponential_weights_list.append(np.exp(-epsilon_instance*costF))


        ##Next F chosen by the exponential mechanism 
        nextF = random.choices(nextF_list, weights=exponential_weights_list)

        ##Step 5 in Algorithm 2
        centroid_list.append(nextF[0])  #nextF is a list that contains the next F

        ##Generate the exponential weights for the final exponential mechanism (step 6)
        columns_F = [distances_matrix[j] for j in nextF[0]]
        costF_final = 0
        for value_index in range(len(database)):
            distance_minimum_over_F = min([col[value_index] for col in columns_F])
            costF_final = costF_final + distance_minimum_over_F


        exponential_final_weights_list.append(np.exp(-epsilon_instance*costF_final))

    ##Step 6--7 in Algorithm 2
    ##Compute last weight
    columns_F = [distances_matrix[j] for j in centroid_list[number_iterations_kmedian]]
    costF_final = 0
    for value_index in range(len(database)):
        distance_minimum_over_F = min([col[value_index] for col in columns_F])
        costF_final = costF_final + distance_minimum_over_F

    exponential_final_weights_list.append(np.exp(-epsilon_instance*costF_final))

    final_cluster = random.choices(range(len(centroid_list)), weights=exponential_final_weights_list)[0] #random.choices returns list that contains item
    return exponential_final_weights_list[final_cluster]/len(database) #Returns final average cost, which is our utility metric
    #return centroid_list[final_cluster] #Returns cluster