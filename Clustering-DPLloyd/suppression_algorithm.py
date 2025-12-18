import pandas as pd
import numpy as np
from multiprocessing import Pool
from suppression_privacy_parameters import *
from generate_average_distance_list import *
from DPLloyd import *
os.environ['TF_XLA_FLAGS']= '--tf_xla_enable_xla_devices'

"""From a list of values, generate the list of pairs (m,M) such that m and M are elements of the list and m<=M"""
def generate_triangular_list_m_M(list_input: list=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])-> list:
    m_and_M=[]

    for m in list_input:
        for M in list_input:
            if m<=M:
                m_and_M.append([m,M])

    return m_and_M

"""Given a dataset, it returns a suppressed dataset with the records suppressed 
according to the outlier scores in the probabilities file"""
def suppressed_dataset(probabilities, dataset):
    new_dataset=[]

    for (_,record),probability in zip(dataset.iterrows(),probabilities):
        x=np.random.random(1)
        if (x<=probability):
            new_dataset.append(record)
    return new_dataset

"""Compute NICV (normalized intracluser distance)"""
def compute_NICV(database, centroids):
    total_element=database.shape[0]
    
    number_centroids=len(centroids)

    error_sum = 0
    for record in database:
        closest_centroid_distance = norm_calculation(record,centroids[0])
        for c in range(1,number_centroids):
            distance = norm_calculation(record,centroids[c])
            if distance<closest_centroid_distance:
                closest_centroid_distance = distance

        error_sum=error_sum+(closest_centroid_distance)**2

    return error_sum/total_element

"""Iteration of MoS"""
def iteration_suppression(arg):
    m, M, epsilon, probability_of_being_sampled_list, df, number_dimensions, number_clusters, number_iterations_DPLloyd, normalized_range_value = arg
    
    np.random.seed(int.from_bytes(os.urandom(4),"big"))

    database = suppressed_dataset(probabilities=probability_of_being_sampled_list, dataset=df)
    database_array = np.array(database)
    
    centroids=DPLloyd(database=database_array, number_dimensions=number_dimensions, number_clusters=number_clusters, normalized_range_value=normalized_range_value, epsilon=epsilon, number_iterations_DPLloyd=number_iterations_DPLloyd)

    NICV=compute_NICV(database=database_array, centroids=centroids)

    return [m, M, epsilon, NICV]

"""Compute MoS for Clustering"""
def MoS_Clustering(output_file_name, df, columns, path_average_distances, m_and_M, number_clusters, epsilon, normalized_range_value, number_iterations_DPLloyd, EpsDeltaChange, numberofrepeat):
    number_dimensions = len(columns)

    header=["m", "M", "epsilon_of_M", "NICV"]
    element=[]

    average_distance_df = pd.read_csv(path_average_distances).iloc[:,0]

    jobs = []
    for m, M in m_and_M:
        probability_of_being_deleted_list = m+(M-m)*average_distance_df
        probability_of_being_sampled_list = 1-probability_of_being_deleted_list

        if EpsDeltaChange==True:
            epsilon_of_M = calculate_eps_suppression_inverse(m=m,M=M,eps=epsilon)
        else:
            epsilon_of_M = epsilon

        for k in range(numberofrepeat):
            jobs.append((m, M, epsilon_of_M, probability_of_being_sampled_list, df, number_dimensions, number_clusters, number_iterations_DPLloyd, normalized_range_value))
    
    with Pool(64) as pool:
        element.extend(pool.map(iteration_suppression, jobs))

    df=pd.DataFrame(element, columns=header)
    df.to_csv(output_file_name, index=False)

    compute_average_file(df=df,file_name=output_file_name,m_and_M=m_and_M)
    compute_variance_file(df=df,file_name=output_file_name,m_and_M=m_and_M)

"""Iteration of M"""
def iteration_without_suppression(arg):
    m, M, epsilon, database_array, number_dimensions, number_clusters, number_iterations_DPLloyd, normalized_range_value = arg

    np.random.seed(int.from_bytes(os.urandom(4),"big"))

    centroids=DPLloyd(database=database_array, number_dimensions=number_dimensions, number_clusters=number_clusters, normalized_range_value=normalized_range_value, epsilon=epsilon, number_iterations_DPLloyd=number_iterations_DPLloyd)

    NICV=compute_NICV(database=database_array,centroids=centroids)

    return [m, M, epsilon, NICV]


"""Compute M for Clustering"""
def M_Clustering(output_file_name, df, columns, m_and_M, epsilon, number_clusters, normalized_range_value, number_iterations_DPLloyd, EpsDeltaChange, numberofrepeat):
    database_array=np.array(df)
    number_dimensions = len(columns)

    header=["m", "M", "epsilon_of_M", "NICV"]
    element=[]

    jobs=[]
    for m, M in m_and_M:

        if EpsDeltaChange==True:
            epsilon_of_M = calculate_eps_suppression(m=m,M=M,eps=epsilon)
        else:
            epsilon_of_M = epsilon
    
        for iteration in range(numberofrepeat):
            jobs.append((m, M, epsilon_of_M, database_array, number_dimensions, number_clusters, number_iterations_DPLloyd, normalized_range_value))
    
    with Pool(64) as pool:
        element.extend(pool.map(iteration_without_suppression, jobs))

    new_df=pd.DataFrame(element, columns=header)
    new_df.to_csv(output_file_name, index=False)

    compute_average_file(df=new_df,file_name=output_file_name,m_and_M=m_and_M)
    compute_variance_file(df=new_df,file_name=output_file_name,m_and_M=m_and_M)

"""Group every entry with the same pair (m,M) and compute its average"""
def compute_average_file(df,file_name,m_and_M):
    header=["m", "M", "epsilon_of_M", "stat_NICV"]
    element=[]

    for m, M in m_and_M:
        df_m_and_M = df[(df["m"]==m) & (df["M"]==M)]
        mean_df=df_m_and_M.mean()
        epsilon_of_M=mean_df["epsilon_of_M"] #epsilon_of_M should be equal for all values we are computing the average over
        mean_NICV=mean_df["NICV"]
        element.append([m, M, epsilon_of_M, mean_NICV])
    output_file_name_stat = file_name.replace(".csv","_Average.csv")
    new_df=pd.DataFrame(element, columns=header)
    new_df.to_csv(output_file_name_stat, index=False)

"""Group every entry with the same pair (m,M) and compute its variance"""
def compute_variance_file(df,file_name,m_and_M):
    header=["m", "M", "epsilon_of_M", "stat_NICV"]
    element=[]

    for m, M in m_and_M:
        df_m_and_M = df[(df["m"]==m) & (df["M"]==M)]
        mean_df=df_m_and_M.mean()
        epsilon_of_M=mean_df["epsilon_of_M"] #epsilon_of_M should be equal for all values we are computing the average over
        var_df=df_m_and_M.var()
        var_NICV=var_df["NICV"]
        element.append([m, M, epsilon_of_M, var_NICV])
    output_file_name_stat = file_name.replace(".csv","_Variance.csv")
    new_df=pd.DataFrame(element, columns=header)
    new_df.to_csv(output_file_name_stat, index=False)

"""Compute difference between MoS and M three ways: 
* without changing privacy parameters, 
* changing those of M so it matches those of MoS, and 
* changing those of MoS so it matches those of M"""
def DifferenceBetweenMetrics(output_file_name, path_MoS_stat, path_MoS_ChangeEpsDelta_stat, path_M_stat, path_M_ChangeEpsDelta_stat, m_and_M):
    df_MoS=pd.read_csv(path_MoS_stat)
    df_MoS_ChangeEpsDelta=pd.read_csv(path_MoS_ChangeEpsDelta_stat)
    df_M=pd.read_csv(path_M_stat)
    df_M_ChangeEpsDelta=pd.read_csv(path_M_ChangeEpsDelta_stat)
    header=["m", "M",
            "difference_error_M_minus_MoS",
            "difference_error_M_minus_MoSChangeEpsDelta",
            "difference_error_MChangeEpsDelta_minus_MoS"]
    element=[]
    
    for m, M in m_and_M:
        df_MoS_instance=df_MoS[(df_MoS["m"]==m) & (df_MoS["M"]==M)]
        df_MoS_ChangeEpsDelta_instance=df_MoS_ChangeEpsDelta[(df_MoS_ChangeEpsDelta["m"]==m) & (df_MoS_ChangeEpsDelta["M"]==M)]
        df_M_instance=df_M[(df_M["m"]==m) & (df_M["M"]==M)]
        df_M_ChangeEpsDelta_instance=df_M_ChangeEpsDelta[(df_M_ChangeEpsDelta["m"]==m) & (df_M_ChangeEpsDelta["M"]==M)]

        difference_error_M_minus_MoS = df_M_instance["stat_NICV"] - df_MoS_instance["stat_NICV"]
        difference_error_M_minus_MoSChangeEpsDelta = df_M_instance["stat_NICV"] - df_MoS_ChangeEpsDelta_instance["stat_NICV"]
        difference_error_MChangeEpsDelta_minus_MoS = df_M_ChangeEpsDelta_instance["stat_NICV"] - df_MoS_instance["stat_NICV"]
        element.append([m, M,
                        difference_error_M_minus_MoS.iloc[0],
                        difference_error_M_minus_MoSChangeEpsDelta.iloc[0],
                        difference_error_MChangeEpsDelta_minus_MoS.iloc[0]])
    new_df=pd.DataFrame(element, columns=header)
    new_df.to_csv(output_file_name, index=False)