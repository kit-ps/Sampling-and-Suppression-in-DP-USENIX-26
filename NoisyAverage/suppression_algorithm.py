import pandas as pd
import numpy as np
from scipy.stats import norm
from multiprocessing import Pool
from suppression_privacy_parameters import *
from generate_average_distance_list import *
os.environ['TF_XLA_FLAGS']= '--tf_xla_enable_xla_devices'

"""From a list of values, generate the list of pairs (m,M) such that m and M are elements of the list and m<=M"""
def generate_triangular_list_m_M(list_input: list=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])-> list:
    m_and_M=[]

    for m in list_input:
        for M in list_input:
            if m<=M:
                m_and_M.append([m,M])

    return m_and_M

"""Noise to add in the Laplace mechanism"""
def Laplace_noise(epsilon: float=1, sensitivity=1):
    if epsilon<=0:
        return float('nan')
    else: 
        return np.random.laplace(0, scale=sensitivity/epsilon)

"""Noise to add in the Gaussian mechanism
Optimal Gaussian as described in
[1] B. Balle and Y.-X. Wang, 
“Improving the Gaussian mechanism for differential privacy: Analytical calibration and optimal denoising,” 
Int. Conf. Mach. Learn., May 2018. 
https://www.semanticscholar.org/paper/Improving-the-Gaussian-Mechanism-for-Differential-Balle-Wang/7d45ac8d29160103cd0bba76aa99b0f60f23a1cd
"""
@lru_cache
def sd_Gaussian(delta, epsilon=1, sensitivity=1):
    delta0 = 1/2 - np.exp(epsilon)*norm.cdf(-np.sqrt(2*epsilon)) #norm.cdf(0) = 1/2
    if delta >= delta0: 
        ##Maximize (place - since we are minimizing)
        def B_plus(theta):
            if theta[0]<0:
                return 100
            return -( norm.cdf( np.sqrt(epsilon*theta[0]) ) - np.exp(epsilon)*norm.cdf( -np.sqrt(epsilon*(theta[0]+2)) ) )

        minimize_output = scipy.optimize.minimize(B_plus, x0 = 0, tol=1E-6)
        minimum_value = -minimize_output['x'][0]
        alpha = np.sqrt(1+minimum_value/2)-np.sqrt(minimum_value/2)
    else: 
        ##Minimize
        def B_minus(theta):
            if theta[0]<0:
                return 100
            return norm.cdf( -np.sqrt(epsilon*theta[0]) ) - np.exp(epsilon)*norm.cdf( -np.sqrt(epsilon*(theta[0]+2)) )

        minimize_output = scipy.optimize.minimize(B_minus, x0 = 0, tol=1E-6)
        minimum_value = minimize_output['x'][0]
        alpha = np.sqrt(1+minimum_value/2)+np.sqrt(minimum_value/2)

    return alpha*sensitivity/np.sqrt(2*epsilon)

def Gaussian_noise(delta, epsilon=1, sensitivity=1):
    if(epsilon<=0):
        return float('nan')
    else:
        sd = sd_Gaussian(delta=delta, epsilon=epsilon, sensitivity=sensitivity)
        noise=np.random.normal(0, sd)
        return noise

"""Classic Gaussian mechanism formulation"""
#def Gaussian_noise(delta, epsilon=1, sensitivity=1):
#    if(epsilon<=0 or epsilon>=1):
#        return float('nan')
#    else:
#        variance=(2*np.power(sensitivity,2)*np.log(1.25/delta))/(np.power(epsilon, 2))
#        sd=np.sqrt(variance)
#        noise=np.random.normal(0, sd)
#        return noise

"""Sensitivity of the sum query: largest value - smallest value"""
def sensitivitySummation(value_range):
    return np.abs(value_range[1] - value_range[0])

"""Given a dataset, it returns a suppressed dataset with the records suppressed 
according to the outlier scores in the probabilities file"""
def suppressed_dataset(probabilities, dataset):
    new_dataset=[]

    for record,probability in zip(dataset,probabilities):
        x=np.random.random(1)
        if (x<=probability):
            new_dataset.append(record)
    return new_dataset

"""Iteration of the base algorithm of MoS (compute suppression and average of suppressed database)"""
def iteration_suppression(arg):
    m, M, original_average, probability_of_being_sampled_list, df = arg

    np.random.seed(int.from_bytes(os.urandom(4),"big"))

    data = suppressed_dataset(probabilities=probability_of_being_sampled_list, dataset=df)
    data = np.array(data)
    suppressed_database_sum = data.sum()
    suppressed_database_number_of_elements = data.shape[0]
    suppressed_database_average = suppressed_database_sum/suppressed_database_number_of_elements
    return [m, M, original_average, suppressed_database_average, suppressed_database_sum, suppressed_database_number_of_elements]

"""Generate numberofrepeat iterations of suppressed databases for every pair (m,M) and computes its average (only needed parameter)"""
def generate_iterations_suppressed_database(output_file_name, df, path_average_distances, m_and_M, numberofrepeat):
    header=["m", "M","original_average", "suppressed_database_average", "suppressed_database_sum", "suppressed_database_number_of_elements"]
    element=[]
    original_average=df.mean()

    average_distance_df = pd.read_csv(path_average_distances).iloc[:,0]

    jobs=[]
    for m,M in m_and_M:
        probability_of_being_deleted_list = m+(M-m)*average_distance_df
        probability_of_being_sampled_list = 1-probability_of_being_deleted_list

        for k in range(numberofrepeat):
            jobs.append((m, M, original_average, probability_of_being_sampled_list, df))
    
    with Pool(64) as pool:
        element.extend(pool.map(iteration_suppression, jobs))

    df=pd.DataFrame(element, columns=header)
    df.to_csv(output_file_name, index=False)

"""Compute MoS for Laplace and Gaussian average"""
def MoS_Laplace_and_Gaussian(output_file_name, base_file, m_and_M, value_range, epsilon, delta, EpsDeltaChange):
    suppressed_df=pd.read_csv(base_file)

    epsilon_of_M_list=[]
    delta_of_M_list=[]
    average_laplace_list=[]
    average_gaussian_list=[]
    AE_laplace_list=[]
    AE_gaussian_list=[]

    for _, row in suppressed_df.iterrows():
        suppressed_database_sum=row["suppressed_database_sum"]
        suppressed_database_number_of_elements=row["suppressed_database_number_of_elements"]
        original_average=row["original_average"]

        if EpsDeltaChange==True:
            m = row["m"]
            M = row["M"]

            epsilon_of_M = calculate_eps_suppression_inverse(m=m,M=M,eps=epsilon)
            delta_of_M = calculate_delta_suppression_inverse(m=m,delta=delta)
        else:
            epsilon_of_M = epsilon
            delta_of_M = delta

        epsilon_of_M_list.append(epsilon_of_M)
        delta_of_M_list.append(delta_of_M)

        #NoisyAverage with Laplace mechanism
        sumtotal_laplace=suppressed_database_sum+Laplace_noise(epsilon=epsilon_of_M/2, sensitivity=sensitivitySummation(value_range))
        total_element_laplace=suppressed_database_number_of_elements +Laplace_noise(epsilon=epsilon_of_M/2, sensitivity=1)
        average_laplace=sumtotal_laplace/total_element_laplace
        #Difference between Laplace and average real
        AE_laplace_suppression=np.abs(original_average-average_laplace)
        
        average_laplace_list.append(average_laplace)
        AE_laplace_list.append(AE_laplace_suppression)
        
        #NoisyAverage with Gaussian mechanism
        sumtotal_gaussian=suppressed_database_sum+ Gaussian_noise(delta=delta_of_M/2, epsilon=epsilon_of_M/2, sensitivity=sensitivitySummation(value_range)) 
        total_element_gaussian=suppressed_database_number_of_elements + Gaussian_noise(delta=delta_of_M/2, epsilon=epsilon_of_M/2, sensitivity=1)
        average_gaussian=sumtotal_gaussian/total_element_gaussian
        #Difference between Gaussian and average real
        AE_gaussian_suppression=np.abs(original_average-average_gaussian)

        average_gaussian_list.append(average_gaussian)
        AE_gaussian_list.append(AE_gaussian_suppression)

    suppressed_df["epsilon_of_M"]=epsilon_of_M_list
    suppressed_df["delta_of_M"]=delta_of_M_list
    suppressed_df["average_laplace"]=average_laplace_list
    suppressed_df["average_gaussian"]=average_gaussian_list
    suppressed_df["AE_laplace"]=AE_laplace_list
    suppressed_df["AE_gaussian"]=AE_gaussian_list
    suppressed_df.to_csv(output_file_name, index=False)

    compute_average_file(df=suppressed_df, file_name=output_file_name, m_and_M=m_and_M)
    compute_variance_file(df=suppressed_df, file_name=output_file_name, m_and_M=m_and_M)


"""Compute M for Laplace and Gaussian average"""
def M_Laplace_and_Gaussian(output_file_name, df, m_and_M, epsilon, delta, value_range, EpsDeltaChange, numberofrepeat):
    sumtotal = df.sum()
    original_average = df.mean()
    total_element = df.shape[0]

    header=["m", "M", "epsilon_of_M", "delta_of_M", "original_average", "average_laplace", "average_gaussian", "AE_laplace", "AE_gaussian"]
    element=[]
    for m, M in m_and_M:

        if EpsDeltaChange==True:
            epsilon_of_M = calculate_eps_suppression(m=m,M=M,eps=epsilon)
            delta_of_M = calculate_delta_suppression(m=m,delta=delta)
        else:
            epsilon_of_M = epsilon
            delta_of_M = delta
    
        for iteration in range(numberofrepeat):
            #NoisyAverage with Laplace mechanism
            sumtotal_laplace=sumtotal+Laplace_noise(epsilon=epsilon_of_M/2, sensitivity=sensitivitySummation(value_range))
            total_element_laplace=total_element +Laplace_noise(epsilon=epsilon_of_M/2, sensitivity=1)
            average_laplace=sumtotal_laplace/total_element_laplace
            AE_laplace=np.abs((sumtotal/total_element)-average_laplace)

            #NoisyAverage with Gaussian mechanism
            sumtotal_gaussian=sumtotal+ Gaussian_noise(delta=delta_of_M/2, epsilon=epsilon_of_M/2, sensitivity=sensitivitySummation(value_range)) 
            total_element_gaussian=total_element + Gaussian_noise(delta=delta_of_M/2, epsilon=epsilon_of_M/2, sensitivity=1)
            average_gaussian=sumtotal_gaussian/total_element_gaussian
            AE_gaussian=np.abs((sumtotal/total_element)-average_gaussian)

            element.append([m, M, epsilon_of_M, delta_of_M, original_average, average_laplace, average_gaussian, AE_laplace, AE_gaussian])
    new_df=pd.DataFrame(element, columns=header)
    new_df.to_csv(output_file_name, index=False)

    compute_average_file(df=new_df, file_name=output_file_name, m_and_M=m_and_M)
    compute_variance_file(df=new_df, file_name=output_file_name, m_and_M=m_and_M)

"""Group every entry with the same pair (m,M) and compute its average"""
def compute_average_file(df, file_name, m_and_M):
    header=["m", "M", "epsilon_of_M", "delta_of_M", "original_average", "stat_AE_laplace", "stat_AE_gaussian"]
    element=[]

    original_average = df["original_average"].iloc[0] #Value is equal for all rows, so we just select the first one

    for m, M in m_and_M:
        df_m_and_M = df[(df["m"]==m) & (df["M"]==M)]
        mean_df = df_m_and_M.mean()
        epsilon_of_M = mean_df["epsilon_of_M"] #epsilon_of_M should be equal for all values we are computing the average over
        delta_of_M = mean_df["delta_of_M"] #delta_of_M should be equal for all values we are computing the average over
        mean_AE_laplace = mean_df["AE_laplace"]
        mean_AE_gaussian = mean_df["AE_gaussian"]
        element.append([m, M, epsilon_of_M, delta_of_M, original_average, mean_AE_laplace, mean_AE_gaussian])
    output_file_name_stat = file_name.replace(".csv","_Average.csv")
    new_df=pd.DataFrame(element, columns=header)
    new_df.to_csv(output_file_name_stat, index=False)

"""Group every entry with the same pair (m,M) and compute its variance"""
def compute_variance_file(df, file_name, m_and_M):
    header=["m", "M", "epsilon_of_M", "delta_of_M", "original_average", "stat_AE_laplace", "stat_AE_gaussian"]
    element=[]

    original_average = df["original_average"].iloc[0]

    for m, M in m_and_M:
        df_m_and_M = df[(df["m"]==m) & (df["M"]==M)]
        mean_df = df_m_and_M.mean()
        epsilon_of_M = mean_df["epsilon_of_M"]
        delta_of_M = mean_df["delta_of_M"]
        var_df = df_m_and_M.var()
        var_AE_laplace = var_df["AE_laplace"]
        var_AE_gaussian = var_df["AE_gaussian"]
        element.append([m, M, epsilon_of_M, delta_of_M, original_average, var_AE_laplace, var_AE_gaussian])
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
    header=["m", "M", "original_average", 
            "difference_laplace_M_minus_MoS", "difference_gaussian_M_minus_MoS", 
            "difference_laplace_M_minus_MoSChangeEpsDelta", "difference_gaussian_M_minus_MoSChangeEpsDelta", 
            "difference_laplace_MChangeEpsDelta_minus_MoS", "difference_gaussian_MChangeEpsDelta_minus_MoS"]
    element=[]
    
    for m, M in m_and_M:
        df_MoS_instance=df_MoS[(df_MoS["m"]==m) & (df_MoS["M"]==M)]
        df_MoS_ChangeEpsDelta_instance=df_MoS_ChangeEpsDelta[(df_MoS_ChangeEpsDelta["m"]==m) & (df_MoS_ChangeEpsDelta["M"]==M)]
        df_M_instance=df_M[(df_M["m"]==m) & (df_M["M"]==M)]
        df_M_ChangeEpsDelta_instance=df_M_ChangeEpsDelta[(df_M_ChangeEpsDelta["m"]==m) & (df_M_ChangeEpsDelta["M"]==M)]

        original_average = df_MoS_instance["original_average"]
        difference_laplace_M_minus_MoS = df_M_instance["stat_AE_laplace"] - df_MoS_instance["stat_AE_laplace"]
        difference_gaussian_M_minus_MoS = df_M_instance["stat_AE_gaussian"] - df_MoS_instance["stat_AE_gaussian"]
        difference_laplace_M_minus_MoSChangeEpsDelta = df_M_instance["stat_AE_laplace"] - df_MoS_ChangeEpsDelta_instance["stat_AE_laplace"]
        difference_gaussian_M_minus_MoSChangeEpsDelta = df_M_instance["stat_AE_gaussian"] - df_MoS_ChangeEpsDelta_instance["stat_AE_gaussian"]
        difference_laplace_MChangeEpsDelta_minus_MoS = df_M_ChangeEpsDelta_instance["stat_AE_laplace"] - df_MoS_instance["stat_AE_laplace"]
        difference_gaussian_MChangeEpsDelta_minus_MoS = df_M_ChangeEpsDelta_instance["stat_AE_gaussian"] - df_MoS_instance["stat_AE_gaussian"]
        element.append([m, M, original_average.iloc[0],
                        difference_laplace_M_minus_MoS.iloc[0], difference_gaussian_M_minus_MoS.iloc[0], 
                        difference_laplace_M_minus_MoSChangeEpsDelta.iloc[0], difference_gaussian_M_minus_MoSChangeEpsDelta.iloc[0],
                        difference_laplace_MChangeEpsDelta_minus_MoS.iloc[0], difference_gaussian_MChangeEpsDelta_minus_MoS.iloc[0]])
    new_df=pd.DataFrame(element, columns=header)
    new_df.to_csv(output_file_name, index=False)