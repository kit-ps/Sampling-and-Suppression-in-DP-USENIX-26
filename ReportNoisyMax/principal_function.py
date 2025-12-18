from suppression_algorithm import *
from graphic_generator import *

def generateFileandGraph(database_name, column_name, main_folder_name, value_range, list_epsilons=[0.25,0.5,1,2], delta=None, numberofrepeat: int = 500):
    path_CSVfiles = os.path.join(main_folder_name,"CSVfiles",column_name)
    # If folder does not exist, create the folder
    if not os.path.exists(path_CSVfiles):
        os.makedirs(path_CSVfiles)

    path_plots = os.path.join(main_folder_name,"Plots",column_name)
    # If folder does not exist, create the folder
    if not os.path.exists(path_plots):
        os.makedirs(path_plots)

    ##Read database and specify column
    df=pd.read_csv(database_name)[column_name]

    ##Compute delta
    if delta==None:
        total_element=df.shape[0]
        delta=np.power((1/total_element), 2)

    ##Generate list of (m,M)
    #m_and_M_large_scale = generate_triangular_list_m_M([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    #m_and_M_short_scale = generate_triangular_list_m_M([0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09])
    #m_and_M_equal = [[round(p,5),round(p,5)] for p in np.arange(0.01,1,0.01)]
    m_and_M_equal = [[round(p,5),round(p,5)] for p in np.arange(0,1,0.01)]
    m_and_M_large_scale = [[round(p,5),round(q,5)] for p in np.arange(0.1,1,0.1) for q in np.arange(p,1,0.1)]
    m_and_M_short_scale = [[round(p,5),round(q,5)] for p in np.arange(0.1,1,0.1) for q in np.arange(p,1,0.1)]
    ##We generate the combined list to simplify code. We remove repeats
    m_and_M_combined = []
    for value in (m_and_M_equal + m_and_M_large_scale + m_and_M_short_scale):
        if value not in m_and_M_combined:
            m_and_M_combined.append(value)

    ##Generate the list of average distances
    file_name_average_distance_list = os.path.join(main_folder_name,column_name+"distances.csv")
    generate_average_distance_list(file_name_output=file_name_average_distance_list, df=df)

    for eps in list_epsilons:
        file_name_start = os.path.join(path_CSVfiles, column_name + "_eps=" + str(eps) + "_delta=" + '%.3e' % delta)
        MoS_RNM(output_file_name = file_name_start + "_MoS.csv", df=df, path_average_distances=file_name_average_distance_list, m_and_M=m_and_M_combined, value_range=value_range, epsilon=eps, delta=delta, EpsDeltaChange=False, numberofrepeat=numberofrepeat)
        MoS_RNM(output_file_name = file_name_start + "_MoS_ChangeEpsDelta.csv", df=df, path_average_distances=file_name_average_distance_list, m_and_M=m_and_M_combined, value_range=value_range, epsilon=eps, delta=delta, EpsDeltaChange=True, numberofrepeat=numberofrepeat)
        M_RNM(output_file_name = file_name_start + "_M.csv", df=df, m_and_M=m_and_M_combined, value_range=value_range, epsilon=eps, delta=delta, EpsDeltaChange=False, numberofrepeat=numberofrepeat)
        M_RNM(output_file_name = file_name_start + "_M_ChangeEpsDelta.csv", df=df, m_and_M=m_and_M_combined, value_range=value_range, epsilon=eps, delta=delta, EpsDeltaChange=True, numberofrepeat=numberofrepeat)
        
        #Difference computation
        file_name_combined = file_name_start + "_combined_Emp_Prob.csv"
        DifferenceBetweenMetrics(output_file_name=file_name_combined,
                        path_MoS_stat=file_name_start + "_MoS_Emp_Prob.csv", 
                        path_MoS_ChangeEpsDelta_stat=file_name_start + "_MoS_ChangeEpsDelta_Emp_Prob.csv",
                        path_M_stat=file_name_start + "_M_Emp_Prob.csv",
                        path_M_ChangeEpsDelta_stat=file_name_start + "_M_ChangeEpsDelta_Emp_Prob.csv",
                        m_and_M=m_and_M_combined)
    
        #Plots
        plot_name_start = os.path.join(path_plots, column_name + "_eps=" + str(eps) + "_delta=" + '%.3e' % delta)

        string_possibilities = ["difference_laplace_error_M_minus_MoS", "difference_gaussian_error_M_minus_MoS", "difference_exponential_error_M_minus_MoS", "difference_exponential_mechanism_error_M_minus_MoS",
            "difference_laplace_error_M_minus_MoSChangeEpsDelta", "difference_gaussian_error_M_minus_MoSChangeEpsDelta", "difference_exponential_error_M_minus_MoSChangeEpsDelta", "difference_exponential_mechanism_error_M_minus_MoSChangeEpsDelta",
            "difference_laplace_error_MChangeEpsDelta_minus_MoS", "difference_gaussian_error_MChangeEpsDelta_minus_MoS", "difference_exponential_error_MChangeEpsDelta_minus_MoS", "difference_exponential_mechanism_error_MChangeEpsDelta_minus_MoS"]

        for string in string_possibilities:
            for list_m_and_M, file_name_m_and_M, plots_limits in zip([m_and_M_large_scale,m_and_M_short_scale],["10--90","1--9"],[ [[0,1],[0,1]], [[0,0.1],[0,0.1]] ]):
                generate_plot_suppression(plot_path_start=plot_name_start, csv_path=file_name_combined, plot_values=string, epsilon=eps, list_m_and_M=list_m_and_M, file_name_m_and_M=file_name_m_and_M, plots_limits=plots_limits)

    ##Plots for the uniform Poisson sampling case
    csv_path_list_M = [os.path.join(path_CSVfiles, column_name + "_eps=" + str(eps) + "_delta=" + '%.3e' % delta + "_M_Emp_Prob.csv") for eps in list_epsilons]
    csv_path_list_MoSChange = [os.path.join(path_CSVfiles, column_name + "_eps=" + str(eps) + "_delta=" + '%.3e' % delta + "_MoS_ChangeEpsDelta_Emp_Prob.csv") for eps in list_epsilons]

    for plot_type in ["EmpProb", "EmpProb+SD"]:
        for noise_name in ["laplace","gaussian","exponential","exponential_mechanism"]:
            plot_name_start = os.path.join(path_plots, column_name + "_uniform_Poisson_sampling_" + noise_name)
            generate_plot_uniform_Poisson_sampling(plot_path_start=plot_name_start, csv_path_list_M=csv_path_list_M, csv_path_list_MoSChange=csv_path_list_MoSChange, epsilon_list=list_epsilons, plot_type=plot_type, noise_name=noise_name, numberofrepeat=numberofrepeat)
