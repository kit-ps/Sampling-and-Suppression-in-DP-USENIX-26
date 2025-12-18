from suppression_algorithm import *
from graphic_generator import *

def generateFileandGraph(database_name, columns, data_domain_name, main_folder_name, number_clusters, list_epsilons=[0.25,0.5,1,2], numberofrepeat: int = 20):
    path_CSVfiles = os.path.join(main_folder_name,"CSVfiles","_".join(columns))
    # If folder does not exist, create the folder
    if not os.path.exists(path_CSVfiles):
        os.makedirs(path_CSVfiles)

    path_plots = os.path.join(main_folder_name,"Plots","_".join(columns))
    # If folder does not exist, create the folder
    if not os.path.exists(path_plots):
        os.makedirs(path_plots)

    ##Read database
    df = pd.read_csv(database_name)

    ##Read domain
    data_domain = list(np.array(pd.read_csv(data_domain_name)))

    ##Value n for the kmedian algorithm. The value needs to be predefined and we select the size of the database as in the original paper
    value_n = len(df)

    ##Generate list of (m,M)
    #m_and_M_large_scale = generate_triangular_list_m_M([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    #m_and_M_equal = [[round(p,5),round(p,5)] for p in np.arange(0.01,1,0.01)]
    m_and_M_equal = [[round(p,5),round(p,5)] for p in np.arange(0,1,0.01)]
    m_and_M_large_scale = [[round(p,5),round(q,5)] for p in np.arange(0.1,1,0.1) for q in np.arange(p,1,0.1)]
    ##We generate the combined list to simplify code. We remove repeats
    m_and_M_combined = []
    for value in (m_and_M_equal + m_and_M_large_scale):
        if value not in m_and_M_combined:
            m_and_M_combined.append(value)

    ##Generate the list of average distances
    file_name_average_distance_list = os.path.join(main_folder_name,"_".join(columns)+"_distances.csv")
    generate_average_distance_list(file_name_output=file_name_average_distance_list, df=df, columns=columns)

    for eps in list_epsilons:
        file_name_start = os.path.join(path_CSVfiles, "eps=" + str(eps))
        MoS_Clustering(output_file_name = file_name_start + "_MoS.csv", df=df, columns=columns, data_domain=data_domain, path_average_distances=file_name_average_distance_list, m_and_M=m_and_M_combined, epsilon=eps, number_clusters=number_clusters, value_n=value_n, EpsDeltaChange=False,numberofrepeat=numberofrepeat)
        MoS_Clustering(output_file_name = file_name_start + "_MoS_ChangeEpsDelta.csv", df=df, columns=columns, data_domain=data_domain, m_and_M=m_and_M_combined, path_average_distances=file_name_average_distance_list, epsilon=eps, number_clusters=number_clusters, value_n=value_n, EpsDeltaChange=True, numberofrepeat=numberofrepeat)
        M_Clustering(output_file_name = file_name_start + "_M.csv", df=df, columns=columns, m_and_M=m_and_M_combined, data_domain=data_domain, epsilon=eps, number_clusters=number_clusters, value_n=value_n, EpsDeltaChange=False, numberofrepeat=numberofrepeat)
        M_Clustering(output_file_name = file_name_start + "_M_ChangeEpsDelta.csv", df=df, columns=columns, m_and_M=m_and_M_combined, data_domain=data_domain, epsilon=eps, number_clusters=number_clusters, value_n=value_n, EpsDeltaChange=True, numberofrepeat=numberofrepeat)

        #Difference computation
        for statistic in ["Average", "Variance"]:
            file_name_combined = file_name_start + "_combined_" + statistic + ".csv"
            DifferenceBetweenMetrics(output_file_name=file_name_combined,
                            path_MoS_stat=file_name_start + "_MoS_" + statistic + ".csv", 
                            path_MoS_ChangeEpsDelta_stat=file_name_start + "_MoS_ChangeEpsDelta_" + statistic + ".csv",
                            path_M_stat=file_name_start + "_M_" + statistic + ".csv",
                            path_M_ChangeEpsDelta_stat=file_name_start + "_M_ChangeEpsDelta_" + statistic + ".csv",
                            m_and_M=m_and_M_combined)
        
            #Plots
            plot_name_start = os.path.join(path_plots, "eps=" + str(eps))
            for string in ["difference_error_M_minus_MoS", "difference_error_M_minus_MoSChangeEpsDelta", "difference_error_MChangeEpsDelta_minus_MoS"]:
                list_m_and_M=m_and_M_large_scale
                file_name_m_and_M="10--90" 
                plots_limits=[[0,1],[0,1]]
                generate_plot_suppression(plot_path_start=plot_name_start, csv_path=file_name_combined, plot_values=string, list_m_and_M=list_m_and_M, file_name_m_and_M=file_name_m_and_M, plots_limits=plots_limits, statistic=statistic, epsilon=eps)

    ##Plots for the uniform Poisson sampling case
    for plot_type in ["Average", "Average+SD", "Variance"]:  
        csv_path_list_M_Average = [os.path.join(path_CSVfiles, "eps=" + str(eps) + "_M_Average.csv") for eps in list_epsilons]
        csv_path_list_M_Variance = [os.path.join(path_CSVfiles, "eps=" + str(eps) + "_M_Variance.csv") for eps in list_epsilons]
        csv_path_list_MoSChange_Average = [os.path.join(path_CSVfiles, "eps=" + str(eps) + "_MoS_ChangeEpsDelta_Average.csv") for eps in list_epsilons]
        csv_path_list_MoSChange_Variance = [os.path.join(path_CSVfiles, "eps=" + str(eps) + "_MoS_ChangeEpsDelta_Variance.csv") for eps in list_epsilons]

        plot_name_start = os.path.join(path_plots, "_".join(columns) + "_uniform_Poisson_sampling")
        generate_plot_uniform_Poisson_sampling(plot_path_start=plot_name_start, 
                            csv_path_list_M_Average=csv_path_list_M_Average, csv_path_list_M_Variance=csv_path_list_M_Variance, 
                            csv_path_list_MoSChange_Average=csv_path_list_MoSChange_Average, csv_path_list_MoSChange_Variance=csv_path_list_MoSChange_Variance, 
                            epsilon_list=list_epsilons, plot_type=plot_type, numberofrepeat=numberofrepeat)