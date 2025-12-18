import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy import stats
import warnings
warnings.filterwarnings("ignore", message="invalid value")
warnings.filterwarnings("ignore", message="divide by zero")

###Computation of epsilon^S
def calculate_V1(F,m,M):
    """By multiplying a1 by the following value, we will ensure that all values are 'nan' 
    if m==M or F==1, which avoids a division by 0. Function outputs 1 if values are valid
    and 'nan' otherwise."""
    ensure_no_division_by_zero = np.where(m==M,float('nan'),np.where(np.isclose(F,1),float('nan'),1))
    a1 = (F-1) * (M/m) * (np.power((M-m), 2))*ensure_no_division_by_zero
    b1 = -((M-m)/m) * ((np.power(m,2)-4*M*m +2*M)*(F-1) + F*M)
    c1 = ((1-m)/m)* ( (F-1) * (2* np.power(m, 2)-4*M*m-m) + (3*F-1)*M )
    d1 = -(1-m) *( (F-1) * (m-2) + (F/m) )

    D10 = np.power(b1,2)-3*a1*c1
    D11 = (2*np.power(b1,3)) - (9*a1*b1*c1) + (27*np.power(a1, 2)* d1)
    case = np.power(D11, 2)-4*np.power(D10, 3)
    R1 = np.where(D10>0,float('nan'),np.sqrt(np.power(D10,3)))
    V1 = np.where(case>0, 
        -1/(3*a1)*(b1+np.cbrt((D11+np.sqrt(case))/2)+np.cbrt((D11-np.sqrt(case))/2)),
        -1/(3*a1)*(b1+2*np.sqrt(D10)*np.cos(1/3*np.arccos(D10/(2*R1)))) )

    return V1

def calculate_V2(F,m,M):
    """By multiplying a1 by the following value, we will ensure that all values are 'nan' 
    if m==M or F==1, which avoids a division by 0. Function outputs 1 if values are valid
    and 'nan' otherwise."""
    ensure_no_division_by_zero = np.where(m==M,float('nan'),np.where(F==1,float('nan'),1))
    a2 = (F-(F-1)*m)/m*ensure_no_division_by_zero
    b2 = -(6*F-(F-1)*(M+5*m))/m
    c2 = (1/(m*(1-M))) * (m*((F-1)*(m+9*M-9)-F)+4*M*((F-1)*M-4*F+1)+12*F)
    d2 = -(2*F-(F-1)*(M+m))*((4-m-4*M)/(m*(1-M)))+2*(F-1)
    D20 = np.power(b2, 2)-3*a2*c2
    D21 = 2*np.power(b2, 3) - 9*a2*b2*c2+27*np.power(a2, 2)*d2
    R2 = np.sqrt(np.power(D20,3))
    V2 = -1/(3*a2)*(b2+2*np.sqrt(D20)*np.cos((1/3)*np.arccos(D21/(2*R2))))
    
    return V2

def theoretical_eps(eps,m,M):
    F=np.exp(eps)
    V1=np.where(eps==0,((1-m)/(M-m))-np.sqrt(M*m*(1-m)*(1-M))/(M*(M-m)),calculate_V1(F,m,M))
    p1=np.where(V1>1,1,np.where(V1<0,0,V1))
    V2=np.where(eps==0,2-(np.sqrt(m*(1-M)))/(1-M),calculate_V2(F,m,M))
    p2=np.where(V2>1,1,np.where(V2<0,0,V2))
          
    #Maximums of each individual function:
    L1 = np.log( F-(F-1)*(p1*M+ (1-p1)*m) ) + p1*(M/m) + (1-p1)*(1-m)/(1-(p1*M+(1-p1)*m))-1
    L2 = np.log( F-(F-1)*(p2*M+(1-p2)*(M+m-p2*M)/(2-p2)) ) + p2*(M/m) +(1-p2)*(1-((M+m-p2*M)/(2-p2)))/(1-M)-1
    L3 = -(np.log(1/F + (1-1/F)*M)) + (1- (1-M)/(1-m))

    return np.where(L1>=L2,np.where(L1>=L3,L1,L3),np.where(L2>=L3,L2,L3))

def theoretical_eps_all(eps,m,M): 
    F=np.exp(eps)
    return np.where(m>M,float('nan'),np.where(m==M,np.log(F-(F-1)*m), theoretical_eps(eps,m,M)))





### Plot of suppression
def generate_plot_suppression(plot_path_start, csv_path, plot_values, statistic, error_type, epsilon, list_m_and_M, file_name_m_and_M, plots_limits, include_title=True):
    ## Select points to plot
    df_all = pd.read_csv(csv_path)
    Points=[]

    ##We restrict the database to only those points (m,M) in list_m_and_M
    df_all["combined_m_and_M"] = df_all[["m","M"]].values.tolist() #Add column of (m,M) values into the dataframe
    df = df_all[df_all["combined_m_and_M"].isin(list_m_and_M)]
    ## Restrict database to the chosen plot_values
    plot_df = df.filter(["m","M",plot_values],axis=1)
    ##Delete all rows with a NaN value
    plot_df.dropna(inplace=True)

    ## To ensure that the correct error is printed
    if error_type=="AE":
        term = 1
    elif error_type=="RE":
        term = 1/df_all["original_average"].iloc[0]
    elif error_type=="PE":
        term = 100/df_all["original_average"].iloc[0]

    #If the statistic is variance, term must be squared to keep consistent with formula
    if statistic=="Variance":
        title_abbreviation = "Var"+error_type
        term = np.power(term,2)
    else:
        title_abbreviation = "M"+error_type

    ## Decide title and file name
    file_name_end = "_"+plot_values+"_"+title_abbreviation+"_"+file_name_m_and_M+".pdf"
    
    if(plot_values=="difference_laplace_M_minus_MoS"):
        title = "NoisyAverage with Laplace noise: Difference of " + title_abbreviation
        epsilon_text_M = "$\\mathcal{M}$ is $"+str(epsilon)+"$-DP"
        epsilon_text_MoS = "$\\mathcal{M}\\circ\\mathcal{S}$ is $(\\varepsilon^{\\mathcal{S}}("+str(epsilon)+",m,M))$-DP"
    elif(plot_values=="difference_gaussian_M_minus_MoS"):
        title = "NoisyAverage with Gaussian noise: Difference of " + title_abbreviation
        epsilon_text_M = "$\\mathcal{M}$ is $("+str(epsilon)+",\\delta)$-DP"
        epsilon_text_MoS = "$\\mathcal{M}\\circ\\mathcal{S}$ is $(\\varepsilon^{\\mathcal{S}}("+str(epsilon)+",m,M),\\delta^{\\mathcal{S}})$-DP"
    elif(plot_values=="difference_laplace_M_minus_MoSChangeEpsDelta"):
        title = "NoisyAverage with Laplace noise: Difference of " + title_abbreviation
        epsilon_text_M = "$\\mathcal{M}$ is $"+str(epsilon)+"$-DP"
        epsilon_text_MoS = "$\\mathcal{M}\\circ\\mathcal{S}$ is $"+str(epsilon)+"$-DP"  
    elif(plot_values=="difference_gaussian_M_minus_MoSChangeEpsDelta"):
        title = "NoisyAverage with Gaussian noise: Difference of " + title_abbreviation
        epsilon_text_M = "$\\mathcal{M}$ is $("+str(epsilon)+",\\delta)$-DP"
        epsilon_text_MoS = "$\\mathcal{M}\\circ\\mathcal{S}$ is $("+str(epsilon)+",\\delta)$-DP"  
    elif(plot_values=="difference_laplace_MChangeEpsDelta_minus_MoS"):
        title = "NoisyAverage with Laplace noise: Difference of " + title_abbreviation
        epsilon_text_M = "$\\mathcal{M}$ is $(\\varepsilon^{\\mathcal{S}}("+str(epsilon)+",m,M))$-DP"
        epsilon_text_MoS = "$\\mathcal{M}\\circ\\mathcal{S}$ is $(\\varepsilon^{\\mathcal{S}}("+str(epsilon)+",m,M))$-DP"   
    elif(plot_values=="difference_gaussian_MChangeEpsDelta_minus_MoS"):
        title = "NoisyAverage with Gaussian noise: Difference of " + title_abbreviation
        epsilon_text_M = "$\\mathcal{M}$ is $(\\varepsilon^{\\mathcal{S}}("+str(epsilon)+",m,M),\\delta^{\\mathcal{S}})$-DP"
        epsilon_text_MoS = "$\\mathcal{M}\\circ\\mathcal{S}$ is $(\\varepsilon^{\\mathcal{S}}("+str(epsilon)+",m,M),\\delta^{\\mathcal{S}})$-DP" 

    ## Generate list of plot points
    x = plot_df["m"]
    y = plot_df["M"]
    z = plot_df[plot_values]*term
    for i in range(plot_df.shape[0]):
        Points.append([x.iloc[i],y.iloc[i],z.iloc[i]])

    ## Ensure that there are at least three points, otherwise the plot cannot be generated
    if len(x)<3:
        print("Plot "+ title + " for epsilon=" + str(epsilon) +" cannot be generated as there are less than three support values")
        return 0

    ## Specify colors of plot: Ensure that 0 will be yellow, negative numbers will be red and positive numbers will be green
    norm = mpl.colors.TwoSlopeNorm(vmin=min(-1e-10,z.min()), vcenter=0, vmax=max(1e-10,z.max()))

    ## Start plot
    fig, ax = plt.subplots()
    graph=ax.tricontourf(x, y, z, norm=norm, levels=30, cmap="RdYlGn", antialiased=True)
    # Colorbar
    fig.colorbar(graph)
    # Add percentage note
    if error_type == "PE":
        ax.text(1.15, -0.06, '(in p.p.)', va='center', ha='center',fontsize=13,transform=ax.transAxes)
    # Set axis and axis labels
    ax.set_xlim(plots_limits[0])
    ax.set_ylim(plots_limits[1])
    ax.set(xlabel='$m$', ylabel='$M$')
    
    # Print title
    if include_title==True:
        ax.set_title(title)

    # Add the text showing privacy parameters of M and MoS
    plt.text(0.65,0.3,epsilon_text_M,ha="center",va="center",fontsize=13,transform=ax.transAxes)
    plt.text(0.65,0.2,epsilon_text_MoS,ha="center",va="center",fontsize=13,transform=ax.transAxes)

    ### Plot implicitly when eps^S(eps,m,M)=eps with respect to m and M
    # Granularity
    delta = 0.0025*min(plots_limits[0][1]-plots_limits[0][0],plots_limits[1][1]-plots_limits[1][0])
    x = np.arange(plots_limits[0][0]+delta, plots_limits[0][1]-delta, delta)
    y = np.arange(plots_limits[1][0]+delta, plots_limits[1][1]-delta, delta)
    p, q = np.meshgrid(x, y)

    eps_suppression = lambda n, x, y: theoretical_eps_all(eps=n,m=x,M=y)
    
    z=eps_suppression(epsilon,p,q)

    my_blue = "dodgerblue"
    CSblue_line = plt.contour(p, q, z, [epsilon], colors=my_blue)
    ax.clabel(CSblue_line, [epsilon], inline=1, fontsize=10, fmt = "$\\varepsilon^{\\mathcal{S}}=%g$")

    ### Add labels for points
    for i in range(len(Points)):
        plt.text(Points[i][0],Points[i][1],np.format_float_positional(float(Points[i][2]), precision=3),fontsize=7,ha="center",va="center")

    plt.savefig(plot_path_start+file_name_end,bbox_inches='tight')
    plt.close()




### Plot of uniform Poisson sampling
def generate_plot_uniform_Poisson_sampling(plot_path_start, csv_path_list_M_Average, csv_path_list_M_Variance, csv_path_list_MoSChange_Average, csv_path_list_MoSChange_Variance, epsilon_list, plot_type, error_type, mechanism_name, numberofrepeat, include_p1=True, include_title=True):
    fig, ax = plt.subplots()

    ##Filter the csv files that contain the statistic we consider (either Average or Variance)
    #csv_path_list_M = filter(lambda x: x.endswith(statistic + ".csv"), csv_path_list_M)
    #csv_path_list_MoSChange = filter(lambda x: x.endswith(statistic + ".csv"), csv_path_list_MoSChange)
    
    #colors = iter(mpl.colors.TABLEAU_COLORS.values())
    colors = iter(["#999999", "#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7"])

    for eps, csv_path_M_Average, csv_path_M_Variance, csv_path_MoSChange_Average, csv_path_MoSChange_Variance in zip(epsilon_list, csv_path_list_M_Average, csv_path_list_M_Variance, csv_path_list_MoSChange_Average, csv_path_list_MoSChange_Variance):
        df_M_all_Average = pd.read_csv(csv_path_M_Average)
        df_M_all_Variance = pd.read_csv(csv_path_M_Variance)
        df_MoSChange_all_Average = pd.read_csv(csv_path_MoSChange_Average)
        df_MoSChange_all_Variance = pd.read_csv(csv_path_MoSChange_Variance)
        Points_M=[]
        Points_MoSChange = []

        #To ensure that the correct error is printed
        if error_type=="AE":
            term = 1
            ylabel_error = "average error"
        elif error_type=="RE":
            term = 1/df_M_all_Average["original_average"].iloc[0]
            ylabel_error = "relative error"
        elif error_type=="PE":
            term = 100/df_M_all_Average["original_average"].iloc[0]
            ylabel_error = "percent error"

        ##We restrict the database to the points m==M
        df_M_Average = df_M_all_Average[df_M_all_Average["m"]==df_M_all_Average["M"]]
        df_M_Variance= df_M_all_Variance[df_M_all_Variance["m"]==df_M_all_Variance["M"]]
        df_MoSChange_Average = df_MoSChange_all_Average[df_MoSChange_all_Average["m"]==df_MoSChange_all_Average["M"]]
        df_MoSChange_Variance = df_MoSChange_all_Variance[df_MoSChange_all_Variance["m"]==df_MoSChange_all_Variance["M"]]

        df_M_plot_Average = df_M_Average.filter(["m","stat_AE_"+mechanism_name],axis=1).sort_values(by=["m"])
        df_M_plot_Variance = df_M_Variance.filter(["m","stat_AE_"+mechanism_name],axis=1).sort_values(by=["m"])
        df_MoSChange_plot_Average = df_MoSChange_Average.filter(["m","stat_AE_"+mechanism_name],axis=1).sort_values(by=["m"])
        df_MoSChange_plot_Variance = df_MoSChange_Variance.filter(["m","stat_AE_"+mechanism_name],axis=1).sort_values(by=["m"])

        ##Delete all rows with a NaN value
        df_M_plot_Average.dropna(inplace=True)
        df_M_plot_Variance.dropna(inplace=True)
        df_MoSChange_plot_Average.dropna(inplace=True)
        df_MoSChange_plot_Variance.dropna(inplace=True)
        if df_M_plot_Average.shape[0]==0:
            continue

        ##If include_p1 selected, include the plot for the sampling rate p=1 (m=0). 
        ##We select in this case the value of df_M_plot for m=0.01 (for both df)
        #if include_p1==True:
        #    value_p1_Average = df_M_plot_Average["stat_AE_"+mechanism_name].iloc[0]
        #    df_M_plot_Average = pd.concat([pd.DataFrame([[0,value_p1_Average]], columns=df_M_plot_Average.columns), df_M_plot_Average], ignore_index=True).sort_values(by=["m"])
        #    df_MoSChange_plot_Average = pd.concat([pd.DataFrame([[0,value_p1_Average]], columns=df_MoSChange_plot_Average.columns), df_MoSChange_plot_Average], ignore_index=True).sort_values(by=["m"])

        #    value_p1_Variance = df_M_plot_Variance["stat_AE_"+mechanism_name].iloc[0]
        #    df_M_plot_Variance = pd.concat([pd.DataFrame([[0,value_p1_Variance]], columns=df_M_plot_Variance.columns), df_M_plot_Variance], ignore_index=True).sort_values(by=["m"])
        #    df_MoSChange_plot_Variance = pd.concat([pd.DataFrame([[0,value_p1_Variance]], columns=df_MoSChange_plot_Variance.columns), df_MoSChange_plot_Variance], ignore_index=True).sort_values(by=["m"])


        if mechanism_name=="laplace":
            DP_label = str(eps)+"-DP"
        else:
            DP_label = "$("+str(eps)+",\\delta)$-DP"

        #97.5% percentile of the t-Student distribution (df=n-1) for the 95% confidence interval
        t_value = stats.t.ppf(0.975, numberofrepeat-1)

        plot_values_M_Average = df_M_plot_Average["stat_AE_"+mechanism_name]*term
        plot_values_M_SD = np.sqrt(df_M_plot_Variance["stat_AE_"+mechanism_name])*term*t_value/np.sqrt(numberofrepeat)
        plot_values_M_Variance = df_M_plot_Variance["stat_AE_"+mechanism_name]*np.power(term,2)
        plot_values_MoSChange_Average = df_MoSChange_plot_Average["stat_AE_"+mechanism_name]*term
        plot_values_MoSChange_SD = np.sqrt(df_MoSChange_plot_Variance["stat_AE_"+mechanism_name])*term*t_value/np.sqrt(numberofrepeat)
        plot_values_MoSChange_Variance = df_MoSChange_plot_Variance["stat_AE_"+mechanism_name]*np.power(term,2)
        #term is squared for the variance to keep the formula correctly scaled
        
        color = next(colors)
        if plot_type=="Average":
            plt.plot(1-df_M_plot_Average["m"],plot_values_M_Average, color=color, linestyle = "solid", label = "$\\mathcal{M}$ satisfying "+DP_label)
            plt.plot(1-df_MoSChange_plot_Average["m"],plot_values_MoSChange_Average, color=color, linestyle = "dotted", label = "$\\mathcal{M}\\circ\\mathcal{S}$ satisfying "+DP_label)
        elif plot_type=="Average+SD":
            plt.plot(1-df_M_plot_Average["m"],plot_values_M_Average, color=color, linestyle = "solid", label = "$\\mathcal{M}$ satisfying "+DP_label)
            plt.fill_between(1-df_M_plot_Average["m"], plot_values_M_Average - plot_values_M_SD, plot_values_M_Average + plot_values_M_SD, color=color, alpha=0.2)
            plt.plot(1-df_MoSChange_plot_Average["m"],plot_values_MoSChange_Average, color=color, linestyle = "dotted", label = "$\\mathcal{M}\\circ\\mathcal{S}$ satisfying "+DP_label)  
            plt.fill_between(1-df_MoSChange_plot_Average["m"], plot_values_MoSChange_Average - plot_values_MoSChange_SD, plot_values_MoSChange_Average + plot_values_MoSChange_SD, color=color, alpha=0.2)
        else: #"Variance"
            plt.plot(1-df_M_plot_Variance["m"],plot_values_M_Variance, color=color, linestyle = "solid", label = "$\\mathcal{M}$ satisfying "+DP_label)
            plt.plot(1-df_MoSChange_plot_Variance["m"],plot_values_MoSChange_Variance, color=color, linestyle = "dotted", label = "$\\mathcal{M}\\circ\\mathcal{S}$ satisfying "+DP_label)
        


    #Axis limits and legend
    ax.set_xlim([0,1])
    ax.set_yscale('log')
    ax.legend()
    if plot_type=="Variance":
        ylabel_text = "Variance of the "+ylabel_error + " (log scale)"
        file_name_abbreviation = "Var"+error_type
    elif plot_type=="Average":
        ylabel_text = "Mean "+ylabel_error + " (log scale)"
        file_name_abbreviation = "M"+error_type
    else:
        ylabel_text = "Mean "+ylabel_error + " (log scale)"
        file_name_abbreviation = "M"+error_type+"+Sd"
    ax.set(xlabel='Sampling rate',ylabel=ylabel_text)

    #Add "%" to plots if PE is selected
    if error_type=="PE":
        ax.yaxis.set_major_formatter('{x}%')

    #Print Title
    title = "NoisyAverage with "+ mechanism_name.capitalize() + " mechanisms"
    if include_title==True:
        ax.set_title(title)

    file_name_end = "_"+ file_name_abbreviation +".pdf"
    plt.savefig(plot_path_start+file_name_end,bbox_inches='tight')
    plt.close()