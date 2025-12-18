import pandas as pd
import numpy as np
import os
import itertools

def norm_calculation(row1,row2):
    square_sum = 0 
    for i in range(len(row1)):
        square_sum = square_sum + (row1[i]-row2[i])**2
    return np.sqrt(square_sum)

## Generates a database following the specifications of the paper
def generate_database(database_file_name, data_domain_file_name, size_database, col_names, col_dimension, number_clusters, variance_normal, clusters=None, weight_clusters=None):

	number_columns = len(col_names)

	if clusters==None:
		clusters = []
		for cluster in range(number_clusters):
			cluster = []
			for col in col_dimension:
				#For every column, generate a random value in the dimension (excluding the first and last 10% of interval). We round to nearest integer
				cluster.append(round(col[0]*0.1 + (col[1]-col[0])*0.8*np.random.random()))
			clusters.append(cluster)

	if weight_clusters==None:
		weight_clusters = [1/number_clusters for i in range(number_clusters)]
	else:
		#Normalize cluster weights (just in case)
		weight_clusters = weight_clusters/number_clusters

	record_list=[]

	for i in range(size_database):
		#We select the mean for the random value generator. Since np.random.choice only allows 1D arrays, we need to do the following:
		mean_index = np.random.choice(range(number_clusters),p=weight_clusters)
		mean = clusters[mean_index]

		record = mean + np.random.normal(0,variance_normal,number_columns)

		for i_col in range(number_columns):
			#Ensure that every column is an integer that is not outside the domain
			record[i_col] = min(col_dimension[i_col][1],max(col_dimension[i_col][0],round(record[i_col])))

		record_list.append(record)

	df = pd.DataFrame(record_list,columns=col_names)

	#We normalize the values
	max_norm_value = norm_calculation([col_dimension[j][0] for j in range(number_columns)],[col_dimension[j][1] for j in range(number_columns)])

	df = df/max_norm_value

	df.to_csv(database_file_name, index=False)

	#Generate list of all possible column values
	column_domain = []
	for col in col_dimension:
		column_domain.append([i/max_norm_value for i in range(col[0],col[1])])


	#Generate domain (cartesian product of all possible column values)
	data_domain = itertools.product(*column_domain)

	df_domain = pd.DataFrame(data_domain,columns=col_names)
	df_domain.to_csv(os.path.join(main_folder_name,data_domain_file_name), index=False)

generate_database(database_file_name = "database1.csv", data_domain_file_name = "database1_domain.csv", size_database = 100, col_names = ["row1","row2"], col_dimension = [[0,100],[0,100]], number_clusters = 4, variance_normal = 10)
