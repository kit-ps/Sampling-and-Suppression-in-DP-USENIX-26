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

"""Exponential noise"""
def exponential_noise(epsilon=1, sensitivity=1):
    if epsilon<=0:
        return float('nan')
    else:
        return np.random.exponential((2*sensitivity)/epsilon)

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

""" Run RNM with Laplace and exponential noise, the version with Gaussian noise and the exponential mechanism. 
Return whether the mechanism returns correct maximum"""
def check_if_RNM_returns_real_maximum(counts, epsilon, delta, value_range, real_maximum_index):  
    ##If the epsilon is an impossible value (or 0), we avoid the computations (also avoids an error with np.random.choice)
    if(np.isnan(epsilon) or epsilon==0):
        return [float('nan'), float('nan'), float('nan'), float('nan')]

    noisy_counts_laplace_list = []
    noisy_counts_gaussian_list = []
    noisy_counts_exponential_list = []
    weight_exponential_mechanism_list = []

    #We compute the maximum count (necessary for exponential mechanism)
    value_count_maximum = counts[counts.index[0]]

    #Add noise to every count (including the zero ones). Compute the weights for the exponential mechanism.
    for i in range(value_range[0],value_range[1]+1):
        if any(x == i for x in counts.index.to_list()):
            value_count = counts[i]
        else:
            value_count = 0

        noisy_count_laplace = value_count + Laplace_noise(epsilon=epsilon, sensitivity=1)
        noisy_count_gaussian = value_count + Gaussian_noise(epsilon=epsilon, delta=delta, sensitivity=1)
        noisy_count_exponential = value_count + exponential_noise(epsilon=epsilon, sensitivity=1)
        sensitivity_exp_mech=1
        weight_exponential_mechanism = np.exp(epsilon/(2*sensitivity_exp_mech)*(value_count-value_count_maximum))

        noisy_counts_laplace_list.append(noisy_count_laplace)
        noisy_counts_gaussian_list.append(noisy_count_gaussian)
        noisy_counts_exponential_list.append(noisy_count_exponential)
        weight_exponential_mechanism_list.append(weight_exponential_mechanism)

    #Compute result through exponential mechanism
    index_noisy_max_exponential_mechanism = value_range[0] + np.random.choice(range(value_range[0],value_range[1]+1), p=weight_exponential_mechanism_list/sum(weight_exponential_mechanism_list))
    
    ##Find maximum value:
    index_noisy_max_laplace = value_range[0] + noisy_counts_laplace_list.index(max(noisy_counts_laplace_list))
    index_noisy_max_gaussian = value_range[0] + noisy_counts_gaussian_list.index(max(noisy_counts_gaussian_list))
    index_noisy_max_exponential = value_range[0] + noisy_counts_exponential_list.index(max(noisy_counts_exponential_list))

    ##Check whether each index is the correct index or not
    index_laplace_is_correct = int(real_maximum_index==index_noisy_max_laplace)
    index_gaussian_is_correct = int(real_maximum_index==index_noisy_max_gaussian)
    index_exponential_is_correct = int(real_maximum_index==index_noisy_max_exponential)
    index_exponential_mechanism_is_correct = int(real_maximum_index==index_noisy_max_exponential_mechanism)

    ##This is only necessary when using the original Gaussian mechanism, which is only defined for epsilon<1
    #If epsilon>=1, then Gaussian is not defined and we need redefine it as 'nan'
    #if epsilon>=1:
    #    index_gaussian_is_correct = float('nan')

    return [index_laplace_is_correct, index_gaussian_is_correct, index_exponential_is_correct, index_exponential_mechanism_is_correct]

"""Iteration of MoS"""
def iteration_suppression(arg):
    m, M, epsilon, delta, probability_of_being_sampled_list, df, value_range, real_maximum_index = arg

    np.random.seed(int.from_bytes(os.urandom(4),"big"))
    
    database = suppressed_dataset(probabilities=probability_of_being_sampled_list, dataset=df)
    element=pd.DataFrame(database,columns=["col"])
    counts=element["col"].value_counts()

    indices_check = check_if_RNM_returns_real_maximum(counts=counts, epsilon=epsilon, delta=delta, value_range=value_range, real_maximum_index=real_maximum_index)

    return [m, M, epsilon, delta] + indices_check

"""Compute MoS for RNM"""
def MoS_RNM(output_file_name, df, path_average_distances, m_and_M, value_range, epsilon, delta, EpsDeltaChange, numberofrepeat):
    counts=df.value_counts()
    real_maximum_index = counts.index[0] ##Maximum will be the first item of counts

    header=["m", "M", "epsilon_of_M", "delta_of_M", "correct_laplace", "correct_gaussian", "correct_exponential", "correct_exponential_mechanism"]
    element=[]

    average_distance_df = pd.read_csv(path_average_distances).iloc[:,0]

    jobs = []
    for m, M in m_and_M:

        probability_of_being_deleted_list = m+(M-m)*average_distance_df
        probability_of_being_sampled_list = 1-probability_of_being_deleted_list

        if EpsDeltaChange==True:
            epsilon_of_M = calculate_eps_suppression_inverse(m=m,M=M,eps=epsilon)
            delta_of_M = calculate_delta_suppression_inverse(m=m,delta=delta)
        else:
            epsilon_of_M = epsilon
            delta_of_M = delta

        for k in range(numberofrepeat):
            jobs.append((m, M, epsilon_of_M, delta_of_M, probability_of_being_sampled_list, df, value_range, real_maximum_index))
    
    with Pool(64) as pool:
        element.extend(pool.map(iteration_suppression, jobs))

    suppressed_df=pd.DataFrame(element, columns=header)
    suppressed_df.to_csv(output_file_name, index=False)

    compute_empirical_probability_file(df=suppressed_df, file_name=output_file_name, m_and_M=m_and_M)

"""Iteration of M"""
def iteration_without_suppression(arg):
    m, M, epsilon, delta, counts, value_range, real_maximum_index = arg

    np.random.seed(int.from_bytes(os.urandom(4),"big"))

    indices_check = check_if_RNM_returns_real_maximum(counts=counts, epsilon=epsilon, delta=delta, value_range=value_range, real_maximum_index=real_maximum_index)
    
    return [m, M, epsilon, delta] + indices_check

"""Compute M for Clustering"""
def M_RNM(output_file_name, df, m_and_M, value_range, epsilon, delta, EpsDeltaChange, numberofrepeat):
    counts=df.value_counts()
    real_maximum_index = counts.index[0] ##Maximum will be the first item of counts

    header=["m", "M", "epsilon_of_M", "delta_of_M", "correct_laplace", "correct_gaussian", "correct_exponential", "correct_exponential_mechanism"]
    element=[]

    jobs=[]
    for m, M in m_and_M:

        if EpsDeltaChange==True:
            epsilon_of_M = calculate_eps_suppression(m=m,M=M,eps=epsilon)
            delta_of_M = calculate_delta_suppression(m=m,delta=delta)
        else:
            epsilon_of_M = epsilon
            delta_of_M = delta

        for k in range(numberofrepeat):
            jobs.append((m, M, epsilon_of_M, delta_of_M, counts, value_range, real_maximum_index))
    
    with Pool(64) as pool:
        element.extend(pool.map(iteration_without_suppression, jobs))
    new_df=pd.DataFrame(element, columns=header)
    new_df.to_csv(output_file_name, index=False)

    compute_empirical_probability_file(df=new_df, file_name=output_file_name, m_and_M=m_and_M)

"""Group every entry with the same pair (m,M) and compute the empirical probability of correctly outputting the maximum"""
def compute_empirical_probability_file(df, file_name, m_and_M):
    header=["m", "M", "epsilon_of_M", "delta_of_M", "correct_empirical_probability_laplace", "correct_empirical_probability_gaussian", "correct_empirical_probability_exponential", "correct_empirical_probability_exponential_mechanism"]
    element=[]

    for m, M in m_and_M:
        mean_df=df[(df["m"]==m) & (df["M"]==M)].mean()
        epsilon_of_M=mean_df["epsilon_of_M"] #epsilon_of_M should be equal for all values we are computing the average over
        delta_of_M=mean_df["delta_of_M"] #delta_of_M should be equal for all values we are computing the average over
        correct_empirical_probability_laplace=mean_df["correct_laplace"]
        correct_empirical_probability_gaussian=mean_df["correct_gaussian"]
        correct_empirical_probability_exponential=mean_df["correct_exponential"]
        correct_empirical_probability_exponential_mechanism=mean_df["correct_exponential_mechanism"]
        element.append([m, M, epsilon_of_M, delta_of_M, correct_empirical_probability_laplace, correct_empirical_probability_gaussian, correct_empirical_probability_exponential, correct_empirical_probability_exponential_mechanism])
    output_file_name_stat = file_name.replace(".csv","_Emp_Prob.csv")
    new_df=pd.DataFrame(element, columns=header)
    new_df.to_csv(output_file_name_stat, index=False)

def DifferenceBetweenMetrics(output_file_name,path_MoS_stat, path_MoS_ChangeEpsDelta_stat, path_M_stat, path_M_ChangeEpsDelta_stat, m_and_M):
    df_MoS=pd.read_csv(path_MoS_stat)
    df_MoS_ChangeEpsDelta=pd.read_csv(path_MoS_ChangeEpsDelta_stat)
    df_M=pd.read_csv(path_M_stat)
    df_M_ChangeEpsDelta=pd.read_csv(path_M_ChangeEpsDelta_stat)
    header=["m", "M",
            "difference_laplace_error_M_minus_MoS", "difference_gaussian_error_M_minus_MoS", "difference_exponential_error_M_minus_MoS", "difference_exponential_mechanism_error_M_minus_MoS",
            "difference_laplace_error_M_minus_MoSChangeEpsDelta", "difference_gaussian_error_M_minus_MoSChangeEpsDelta", "difference_exponential_error_M_minus_MoSChangeEpsDelta", "difference_exponential_mechanism_error_M_minus_MoSChangeEpsDelta",
            "difference_laplace_error_MChangeEpsDelta_minus_MoS", "difference_gaussian_error_MChangeEpsDelta_minus_MoS", "difference_exponential_error_MChangeEpsDelta_minus_MoS", "difference_exponential_mechanism_error_MChangeEpsDelta_minus_MoS"]
    element=[]
    
    for m, M in m_and_M:
        df_MoS_instance=df_MoS[(df_MoS["m"]==m) & (df_MoS["M"]==M)]
        df_MoS_ChangeEpsDelta_instance=df_MoS_ChangeEpsDelta[(df_MoS_ChangeEpsDelta["m"]==m) & (df_MoS_ChangeEpsDelta["M"]==M)]
        df_M_instance=df_M[(df_M["m"]==m) & (df_M["M"]==M)]
        df_M_ChangeEpsDelta_instance=df_M_ChangeEpsDelta[(df_M_ChangeEpsDelta["m"]==m) & (df_M_ChangeEpsDelta["M"]==M)]

        ## We compute the empirical probability of being incorrect, and thus we add the -. This ensures that "larger values = larger utility". 

        difference_laplace_error_M_minus_MoS = -(df_M_instance["correct_empirical_probability_laplace"] - df_MoS_instance["correct_empirical_probability_laplace"])
        difference_laplace_error_M_minus_MoSChangeEpsDelta = -(df_M_instance["correct_empirical_probability_laplace"] - df_MoS_ChangeEpsDelta_instance["correct_empirical_probability_laplace"])
        difference_laplace_error_MChangeEpsDelta_minus_MoS = -(df_M_ChangeEpsDelta_instance["correct_empirical_probability_laplace"] - df_MoS_instance["correct_empirical_probability_laplace"])
        
        difference_gaussian_error_M_minus_MoS = -(df_M_instance["correct_empirical_probability_gaussian"] - df_MoS_instance["correct_empirical_probability_gaussian"])
        difference_gaussian_error_M_minus_MoSChangeEpsDelta = -(df_M_instance["correct_empirical_probability_gaussian"] - df_MoS_ChangeEpsDelta_instance["correct_empirical_probability_gaussian"])
        difference_gaussian_error_MChangeEpsDelta_minus_MoS = -(df_M_ChangeEpsDelta_instance["correct_empirical_probability_gaussian"] - df_MoS_instance["correct_empirical_probability_gaussian"])

        difference_exponential_error_M_minus_MoS = -(df_M_instance["correct_empirical_probability_exponential"] - df_MoS_instance["correct_empirical_probability_exponential"])
        difference_exponential_error_M_minus_MoSChangeEpsDelta = -(df_M_instance["correct_empirical_probability_exponential"] - df_MoS_ChangeEpsDelta_instance["correct_empirical_probability_exponential"])
        difference_exponential_error_MChangeEpsDelta_minus_MoS = -(df_M_ChangeEpsDelta_instance["correct_empirical_probability_exponential"] - df_MoS_instance["correct_empirical_probability_exponential"])

        difference_exponential_mechanism_error_M_minus_MoS = -(df_M_instance["correct_empirical_probability_exponential_mechanism"] - df_MoS_instance["correct_empirical_probability_exponential_mechanism"])
        difference_exponential_mechanism_error_M_minus_MoSChangeEpsDelta = -(df_M_instance["correct_empirical_probability_exponential_mechanism"] - df_MoS_ChangeEpsDelta_instance["correct_empirical_probability_exponential_mechanism"])
        difference_exponential_mechanism_error_MChangeEpsDelta_minus_MoS = -(df_M_ChangeEpsDelta_instance["correct_empirical_probability_exponential_mechanism"] - df_MoS_instance["correct_empirical_probability_exponential_mechanism"])

        element.append([m, M,
                        difference_laplace_error_M_minus_MoS.iloc[0], difference_gaussian_error_M_minus_MoS.iloc[0], difference_exponential_error_M_minus_MoS.iloc[0], difference_exponential_mechanism_error_M_minus_MoS.iloc[0],
                        difference_laplace_error_M_minus_MoSChangeEpsDelta.iloc[0], difference_gaussian_error_M_minus_MoSChangeEpsDelta.iloc[0], difference_exponential_error_M_minus_MoSChangeEpsDelta.iloc[0], difference_exponential_mechanism_error_M_minus_MoSChangeEpsDelta.iloc[0],
                        difference_laplace_error_MChangeEpsDelta_minus_MoS.iloc[0], difference_gaussian_error_MChangeEpsDelta_minus_MoS.iloc[0], difference_exponential_error_MChangeEpsDelta_minus_MoS.iloc[0], difference_exponential_mechanism_error_MChangeEpsDelta_minus_MoS.iloc[0]])
    new_df=pd.DataFrame(element, columns=header)
    new_df.to_csv(output_file_name, index=False)