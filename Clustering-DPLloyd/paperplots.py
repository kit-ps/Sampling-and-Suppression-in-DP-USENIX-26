import os
from graphic_generator import *
from suppression_algorithm import *

## This function generates directly the precise plots used in our paper (requires the csv files to have been already generated)
def paper_plots(columns, main_folder_name, list_epsilons=[0.25,0.5,1,2], repetitions: int = 500):
    path_CSVfiles = os.path.join(main_folder_name,"CSVfiles","_".join(columns))

    path_plots = os.path.join("PaperPlots","_".join(columns))
    # If folder does not exist, create the folder
    if not os.path.exists(path_plots):
        os.makedirs(path_plots)

    m_and_M_large_scale = [[round(p,5),round(q,5)] for p in np.arange(0.1,1,0.1) for q in np.arange(p,1,0.1)]
   
    statistic="Average"
    for eps in list_epsilons:
        file_name_start = os.path.join(path_CSVfiles, "eps=" + str(eps))
        file_name_combined = file_name_start + "_combined_" + statistic + ".csv"

        #Plots
        plot_name_start = os.path.join(path_plots, "eps=" + str(eps))
        list_m_and_M=m_and_M_large_scale
        file_name_m_and_M="10--90" 
        plots_limits=[[0,1],[0,1]]    
        for string in ["difference_error_M_minus_MoS", "difference_error_M_minus_MoSChangeEpsDelta"]:
            generate_plot_suppression(plot_path_start=plot_name_start, csv_path=file_name_combined, plot_values=string, list_m_and_M=list_m_and_M, file_name_m_and_M=file_name_m_and_M, plots_limits=plots_limits, statistic=statistic, epsilon=eps, include_title=False)

    ## Uniform Poisson sampling plots
    for plot_type in ["Average+SD"]:  
        csv_path_list_M_Average = [os.path.join(path_CSVfiles, "eps=" + str(eps) + "_M_Average.csv") for eps in list_epsilons]
        csv_path_list_M_Variance = [os.path.join(path_CSVfiles, "eps=" + str(eps) + "_M_Variance.csv") for eps in list_epsilons]
        csv_path_list_MoSChange_Average = [os.path.join(path_CSVfiles, "eps=" + str(eps) + "_MoS_ChangeEpsDelta_Average.csv") for eps in list_epsilons]
        csv_path_list_MoSChange_Variance = [os.path.join(path_CSVfiles, "eps=" + str(eps) + "_MoS_ChangeEpsDelta_Variance.csv") for eps in list_epsilons]

        plot_name_start = os.path.join(path_plots, "_".join(columns) + "_uniform_Poisson_sampling")
        generate_plot_uniform_Poisson_sampling(plot_path_start=plot_name_start, 
                            csv_path_list_M_Average=csv_path_list_M_Average, csv_path_list_M_Variance=csv_path_list_M_Variance, 
                            csv_path_list_MoSChange_Average=csv_path_list_MoSChange_Average, csv_path_list_MoSChange_Variance=csv_path_list_MoSChange_Variance, 
                            epsilon_list=list_epsilons, plot_type=plot_type, repetitions=repetitions)

#paper_plots(["age", "fnlwgt", "education-num", "capital-gain", "capital-loss", "hours-per-week"], "Adult_clustering")