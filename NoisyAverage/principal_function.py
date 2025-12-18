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
    #Otherwise, the selected delta is used

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
    generate_average_distance_list(file_name_output=file_name_average_distance_list, df=df, value_range=value_range)

    ##Generate base of suppressed database for MoS (simplifies some computations)
    file_name_base = os.path.join(path_CSVfiles, column_name + "_base.csv")
    generate_iterations_suppressed_database(output_file_name=file_name_base, df=df, path_average_distances=file_name_average_distance_list, m_and_M=m_and_M_combined, numberofrepeat=numberofrepeat)

    for eps in list_epsilons:
        file_name_start = os.path.join(path_CSVfiles, column_name + "_eps=" + str(eps) + "_delta=" + '%.3e' % delta)
        MoS_Laplace_and_Gaussian(output_file_name = file_name_start + "_MoS.csv", base_file=file_name_base,  m_and_M=m_and_M_combined, value_range=value_range, epsilon=eps, delta=delta, EpsDeltaChange=False)
        MoS_Laplace_and_Gaussian(output_file_name = file_name_start + "_MoS_ChangeEpsDelta.csv", base_file=file_name_base, m_and_M=m_and_M_combined, value_range=value_range, epsilon=eps, delta=delta, EpsDeltaChange=True)
        M_Laplace_and_Gaussian(output_file_name = file_name_start + "_M.csv", df=df, m_and_M=m_and_M_combined, value_range=value_range, epsilon=eps, delta=delta, EpsDeltaChange=False, numberofrepeat=numberofrepeat)
        M_Laplace_and_Gaussian(output_file_name = file_name_start + "_M_ChangeEpsDelta.csv", df=df, m_and_M=m_and_M_combined, value_range=value_range, epsilon=eps, delta=delta, EpsDeltaChange=True, numberofrepeat=numberofrepeat)

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
            plot_name_start = os.path.join(path_plots, column_name + "_eps=" + str(eps) + "_delta=" + '%.3e' % delta)
            if statistic=="Average":
                error_type="PE" ##Percent error
            else:
                error_type="RE" ##Relative error

            for string in ["difference_laplace_M_minus_MoS", "difference_gaussian_M_minus_MoS", "difference_laplace_M_minus_MoSChangeEpsDelta", "difference_gaussian_M_minus_MoSChangeEpsDelta", "difference_laplace_MChangeEpsDelta_minus_MoS", "difference_gaussian_MChangeEpsDelta_minus_MoS"]:
                for list_m_and_M, file_name_m_and_M, plots_limits in zip([m_and_M_large_scale,m_and_M_short_scale],["10--90","1--9"],[ [[0,1],[0,1]], [[0,0.1],[0,0.1]] ]):
                    generate_plot_suppression(plot_path_start=plot_name_start, csv_path=file_name_combined, plot_values=string, statistic=statistic, error_type=error_type, epsilon=eps, list_m_and_M=list_m_and_M, file_name_m_and_M=file_name_m_and_M, plots_limits=plots_limits)

    ##Plots for the uniform Poisson sampling case
    for plot_type in ["Average", "Average+SD", "Variance"]:  
        if plot_type=="Average" or plot_type=="Average+SD":
            error_type="PE" ##Percent error
        else:
            error_type="RE" ##Relative error 

        csv_path_list_M_Average = [os.path.join(path_CSVfiles, column_name + "_eps=" + str(eps) + "_delta=" + '%.3e' % delta + "_M_Average.csv") for eps in list_epsilons]
        csv_path_list_M_Variance = [os.path.join(path_CSVfiles, column_name + "_eps=" + str(eps) + "_delta=" + '%.3e' % delta + "_M_Variance.csv") for eps in list_epsilons]
        csv_path_list_MoSChange_Average = [os.path.join(path_CSVfiles, column_name + "_eps=" + str(eps) + "_delta=" + '%.3e' % delta + "_MoS_ChangeEpsDelta_Average.csv") for eps in list_epsilons]
        csv_path_list_MoSChange_Variance = [os.path.join(path_CSVfiles, column_name + "_eps=" + str(eps) + "_delta=" + '%.3e' % delta + "_MoS_ChangeEpsDelta_Variance.csv") for eps in list_epsilons]

        for mechanism_name in ["laplace","gaussian"]:
            plot_name_start = os.path.join(path_plots, column_name + "_uniform_Poisson_sampling_" + mechanism_name)
            generate_plot_uniform_Poisson_sampling(plot_path_start=plot_name_start, 
                            csv_path_list_M_Average=csv_path_list_M_Average, csv_path_list_M_Variance=csv_path_list_M_Variance, 
                            csv_path_list_MoSChange_Average=csv_path_list_MoSChange_Average, csv_path_list_MoSChange_Variance=csv_path_list_MoSChange_Variance, 
                            epsilon_list=list_epsilons, plot_type=plot_type, error_type=error_type, mechanism_name=mechanism_name, numberofrepeat=numberofrepeat)