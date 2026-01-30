import os
from graphic_generator import *
from suppression_algorithm import *

## This function generates directly the precise plots used in our paper (requires the csv files to have been already generated)
def paper_plots(database_name, column_name, main_folder_name, list_epsilons=[0.25,0.5,1,2], delta=None, numberofrepeat: int = 2000):
    path_CSVfiles = os.path.join(main_folder_name,"CSVfiles",column_name)

    path_plots = os.path.join("PaperPlots",main_folder_name,column_name)
    # If folder does not exist, create the folder
    if not os.path.exists(path_plots):
        os.makedirs(path_plots)

    ##Compute delta
    if delta==None:
        ##Read database and specify column
        df=pd.read_csv(database_name)[column_name]

        total_element=df.shape[0]
        delta=np.power((1/total_element), 2)

    m_and_M_large_scale = [[round(p,5),round(q,5)] for p in np.arange(0.1,1,0.1) for q in np.arange(p,1,0.1)]
    
    for eps in list_epsilons:
        file_name_start = os.path.join(path_CSVfiles, column_name + "_eps=" + str(eps) + "_delta=" + '%.3e' % delta)
        
        file_name_combined = file_name_start + "_combined_Emp_Prob.csv"

        #Plots
        plot_name_start = os.path.join(path_plots, column_name + "_eps=" + str(eps))

        string_possibilities = ["difference_laplace_error_M_minus_MoS", "difference_gaussian_error_M_minus_MoS", "difference_exponential_error_M_minus_MoS", "difference_exponential_mechanism_error_M_minus_MoS",
            "difference_laplace_error_M_minus_MoSChangeEpsDelta", "difference_gaussian_error_M_minus_MoSChangeEpsDelta", "difference_exponential_error_M_minus_MoSChangeEpsDelta", "difference_exponential_mechanism_error_M_minus_MoSChangeEpsDelta"]

        for string in string_possibilities:
            for list_m_and_M, file_name_m_and_M, plots_limits in zip([m_and_M_large_scale],["10--90"],[ [[0,1],[0,1]] ]):
                generate_plot_suppression(plot_path_start=plot_name_start, csv_path=file_name_combined, plot_values=string, epsilon=eps, list_m_and_M=list_m_and_M, file_name_m_and_M=file_name_m_and_M, plots_limits=plots_limits, include_title=False)

    ##Plots for the uniform Poisson sampling case
    csv_path_list_M = [os.path.join(path_CSVfiles, column_name + "_eps=" + str(eps) + "_delta=" + '%.3e' % delta + "_M_Emp_Prob.csv") for eps in list_epsilons]
    csv_path_list_MoSChange = [os.path.join(path_CSVfiles, column_name + "_eps=" + str(eps) + "_delta=" + '%.3e' % delta + "_MoS_ChangeEpsDelta_Emp_Prob.csv") for eps in list_epsilons]

    for noise_name in ["laplace","gaussian","exponential","exponential_mechanism"]:
        plot_name_start = os.path.join(path_plots, column_name + "_uniform_Poisson_sampling_" + noise_name)
        generate_plot_uniform_Poisson_sampling(plot_path_start=plot_name_start, csv_path_list_M=csv_path_list_M, csv_path_list_MoSChange=csv_path_list_MoSChange, 
                    epsilon_list=list_epsilons, plot_type="EmpProb+SD", noise_name=noise_name, numberofrepeat=numberofrepeat)

#paper_plots("adult_train.csv","age","Adult")
#paper_plots("adult_train.csv","hours-per-week","Adult")

#paper_plots("irishn_train.csv","Age","Irishn")
#paper_plots("irishn_train.csv","HighestEducationCompleted","Irishn")