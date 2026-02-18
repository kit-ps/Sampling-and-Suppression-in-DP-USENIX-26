import os
from graphic_generator import *
from suppression_algorithm import *

## This function generates directly the precise plots used in our paper (requires the csv files to have been already generated)
def paper_plots(database_name, column_name, main_folder_name, list_epsilons=[0.25,0.5,1,2], delta=None, repetitions: int = 500):
    path_CSVfiles = os.path.join(main_folder_name,"CSVfiles",column_name)

    path_plots = os.path.join("PaperPlots",main_folder_name,column_name)
    # If folder does not exist, create the folder
    if not os.path.exists(path_plots):
        os.makedirs(path_plots)

    ##Compute delta if none provided
    if delta==None:
        ##Read database and specify column
        df=pd.read_csv(database_name)[column_name]

        total_element=df.shape[0]
        delta=np.power((1/total_element), 2)

    m_and_M_large_scale = [[round(p,5),round(q,5)] for p in np.arange(0.1,1,0.1) for q in np.arange(p,1,0.1)]

    statistic = "Average"
    error_type = "PE"

    ## Outlier-score suppression plots
    for eps in list_epsilons:
        file_name_start = os.path.join(path_CSVfiles, column_name + "_eps=" + str(eps) + "_delta=" + '%.3e' % delta)
        file_name_combined = file_name_start + "_combined_" + statistic + ".csv"
            
        #Plots
        plot_name_start = os.path.join(path_plots, column_name + "_eps=" + str(eps) + "_delta=" + '%.3e' % delta)

        for string in ["difference_laplace_M_minus_MoS", "difference_gaussian_M_minus_MoS", "difference_laplace_M_minus_MoSChangeEpsDelta", "difference_gaussian_M_minus_MoSChangeEpsDelta"]:
            for list_m_and_M, file_name_m_and_M, plots_limits in zip([m_and_M_large_scale],["10--90"],[ [[0,1],[0,1]] ]):
                generate_plot_suppression(plot_path_start=plot_name_start, csv_path=file_name_combined, plot_values=string, statistic=statistic, error_type=error_type, epsilon=eps, list_m_and_M=list_m_and_M, file_name_m_and_M=file_name_m_and_M, plots_limits=plots_limits, include_title=False)

    ## Uniform Poisson sampling plots
    plot_type="Average+SD"

    csv_path_list_M_Average = [os.path.join(path_CSVfiles, column_name + "_eps=" + str(eps) + "_delta=" + '%.3e' % delta + "_M_Average.csv") for eps in list_epsilons]
    csv_path_list_M_Variance = [os.path.join(path_CSVfiles, column_name + "_eps=" + str(eps) + "_delta=" + '%.3e' % delta + "_M_Variance.csv") for eps in list_epsilons]
    csv_path_list_MoSChange_Average = [os.path.join(path_CSVfiles, column_name + "_eps=" + str(eps) + "_delta=" + '%.3e' % delta + "_MoS_ChangeEpsDelta_Average.csv") for eps in list_epsilons]
    csv_path_list_MoSChange_Variance = [os.path.join(path_CSVfiles, column_name + "_eps=" + str(eps) + "_delta=" + '%.3e' % delta + "_MoS_ChangeEpsDelta_Variance.csv") for eps in list_epsilons]

    for mechanism_name in ["laplace","gaussian"]:
        plot_name_start = os.path.join(path_plots, column_name + "_uniform_Poisson_sampling_" + mechanism_name)
        generate_plot_uniform_Poisson_sampling(plot_path_start=plot_name_start, 
                        csv_path_list_M_Average=csv_path_list_M_Average, csv_path_list_M_Variance=csv_path_list_M_Variance, 
                        csv_path_list_MoSChange_Average=csv_path_list_MoSChange_Average, csv_path_list_MoSChange_Variance=csv_path_list_MoSChange_Variance, 
                        epsilon_list=list_epsilons, plot_type=plot_type, error_type=error_type, mechanism_name=mechanism_name, repetitions=500)

#paper_plots("adult_train.csv","age","Adult")
#paper_plots("adult_train.csv","hours-per-week","Adult")

#paper_plots("census.csv","FEDTAX","Census")
#paper_plots("census.csv","FICA","Census")

#paper_plots("irishn_train.csv","Age","Irishn")
#paper_plots("irishn_train.csv","HighestEducationCompleted","Irishn")