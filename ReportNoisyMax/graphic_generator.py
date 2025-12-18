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
def generate_plot_suppression(plot_path_start, csv_path, plot_values, epsilon, list_m_and_M, file_name_m_and_M, plots_limits, include_title=True):
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
    
    #Find title and tags for privacy parameters of M and MoS. Gaussian mechanism is the only approximate DP algorithms and follows different structure
    if("gaussian" in plot_values):
        title = "Gaussian RNM: Difference of empirical probability"
        if("M_minus_MoSChangeEpsDelta" in plot_values):
            epsilon_text_M = "$\\mathcal{M}$ is $("+str(epsilon)+",\\delta)$-DP"
            epsilon_text_MoS = "$\\mathcal{M}\\circ\\mathcal{S}$ is $("+str(epsilon)+",\\delta)$-DP" 
        elif("MChangeEpsDelta_minus_MoS" in plot_values): 
            epsilon_text_M = "$\\mathcal{M}$ is $(\\varepsilon^{\\mathcal{S}}("+str(epsilon)+",m,M),\\delta^{\\mathcal{S}})$-DP"
            epsilon_text_MoS = "$\\mathcal{M}\\circ\\mathcal{S}$ is $(\\varepsilon^{\\mathcal{S}}("+str(epsilon)+",m,M),\\delta^{\\mathcal{S}})$-DP" 
        elif("M_minus_MoS" in plot_values):
            epsilon_text_M = "$\\mathcal{M}$ is $("+str(epsilon)+",\\delta)$-DP"
            epsilon_text_MoS = "$\\mathcal{M}\\circ\\mathcal{S}$ is $(\\varepsilon^{\\mathcal{S}}("+str(epsilon)+",m,M),\\delta^{\\mathcal{S}})$-DP"
    else:
        if("laplace" in plot_values):
            title = "Laplace RNM: Difference of empirical probability"
        elif("exponential_mechanism" in plot_values):
            title = "Exponential mechanism: Difference of empirical probability"
        elif("exponential" in plot_values):
            title = "Exponential RNM: Difference of empirical probability"

        if("M_minus_MoSChangeEpsDelta" in plot_values):
            epsilon_text_M = "$\\mathcal{M}$ is $"+str(epsilon)+"$-DP"
            epsilon_text_MoS = "$\\mathcal{M}\\circ\\mathcal{S}$ is $"+str(epsilon)+"$-DP" 
        elif("MChangeEpsDelta_minus_MoS" in plot_values): 
            epsilon_text_M = "$\\mathcal{M}$ is $(\\varepsilon^{\\mathcal{S}}("+str(epsilon)+",m,M))$-DP"
            epsilon_text_MoS = "$\\mathcal{M}\\circ\\mathcal{S}$ is $(\\varepsilon^{\\mathcal{S}}("+str(epsilon)+",m,M))$-DP" 
        elif("M_minus_MoS" in plot_values):
            epsilon_text_M = "$\\mathcal{M}$ is $"+str(epsilon)+"$-DP"
            epsilon_text_MoS = "$\\mathcal{M}\\circ\\mathcal{S}$ is $(\\varepsilon^{\\mathcal{S}}("+str(epsilon)+",m,M))$-DP"

    ## Generate list of plot points
    x = plot_df["m"]
    y = plot_df["M"]
    z = plot_df[plot_values]
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

    ##Plot implicitly when eps^S(eps,m,M)=eps with respect to m and M
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

    ##Add labels for points
    for i in range(len(Points)):
        plt.text(Points[i][0],Points[i][1],np.format_float_positional(float(Points[i][2]), precision=3),fontsize=7,ha="center",va="center")

    file_name_end = "_" + plot_values + "_" + file_name_m_and_M + ".pdf"
    plt.savefig(plot_path_start+file_name_end,bbox_inches='tight')
    plt.close()





###Wilson confidence interval (95%)
def Wilson_CI(emp_prob,numberofrepeat):
    #97.5% percentile of the normal distribution for the 95% confidence interval
    normal_percentile = stats.norm.ppf(0.975)

    term1 = emp_prob + np.power(normal_percentile,2)/(2*numberofrepeat)
    term2 = normal_percentile/(2*numberofrepeat)*np.sqrt(4*numberofrepeat*emp_prob*(1-emp_prob)+np.power(normal_percentile,2))
    term3 = (1+np.power(normal_percentile,2)/numberofrepeat)

    return [(term1 - term2)/term3, (term1 + term2)/term3]

### Plot of uniform Poisson sampling
def generate_plot_uniform_Poisson_sampling(plot_path_start, csv_path_list_M, csv_path_list_MoSChange, epsilon_list, plot_type, noise_name, numberofrepeat, include_p1=True, include_title=True):
    fig, ax = plt.subplots()

    #colors = iter(mpl.colors.TABLEAU_COLORS.values())
    colors = iter(["#999999", "#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7"])

    for eps, csv_path_M, csv_path_MoSChange in zip(epsilon_list, csv_path_list_M, csv_path_list_MoSChange):
        df_M_all = pd.read_csv(csv_path_M)
        df_MoSChange_all = pd.read_csv(csv_path_MoSChange)
        Points_M=[]
        Points_MoSChange = []

        ##We restrict the database to the points m==M
        df_M = df_M_all[df_M_all["m"]==df_M_all["M"]]
        df_MoSChange = df_MoSChange_all[df_MoSChange_all["m"]==df_MoSChange_all["M"]]

        df_M_plot = df_M.filter(["m","correct_empirical_probability_"+noise_name],axis=1).sort_values(by=["m"])
        df_MoSChange_plot = df_MoSChange.filter(["m","correct_empirical_probability_"+noise_name],axis=1).sort_values(by=["m"])

        ##Delete all rows with a NaN value
        df_M_plot.dropna(inplace=True)
        df_MoSChange_plot.dropna(inplace=True)
        if df_M_plot.shape[0]==0:
            continue

        ##If include_p1 selected, include the plot for the sampling rate p=1 (m=0). 
        ##We select in this case the value of df_M_plot for m=0.01 (for both df)
        #if include_p1==True:
        #    value_p1 = df_M_plot["correct_empirical_probability_"+noise_name].iloc[0]
        #    df_M_plot = pd.concat([pd.DataFrame([[0,value_p1]], columns=df_M_plot.columns), df_M_plot], ignore_index=True).sort_values(by=["m"])
        #    df_MoSChange_plot = pd.concat([pd.DataFrame([[0,value_p1]], columns=df_MoSChange_plot.columns), df_MoSChange_plot], ignore_index=True).sort_values(by=["m"])

        if noise_name=="laplace":
            mechanism_name = "RNM with Laplace noise"
            DP_label = str(eps)+"-DP"
        elif noise_name=="gaussian":
            mechanism_name = "RNM-like variant with Gaussian noise"
            DP_label = "$("+str(eps)+",\\delta)$-DP"
        elif noise_name=="exponential":
            mechanism_name = "RNM with exponential noise"
            DP_label = str(eps)+"-DP"
        elif noise_name=="exponential_mechanism":
            mechanism_name = "Exponential mechanism"
            DP_label = str(eps)+"-DP"

        color = next(colors)
        
        plot_values_M_EmpProb = 1-df_M_plot["correct_empirical_probability_"+noise_name]
        plot_values_M_upper_CI_bound = [Wilson_CI(emp_prob=value,numberofrepeat=numberofrepeat)[1] for value in plot_values_M_EmpProb]
        plot_values_M_lower_CI_bound = [Wilson_CI(emp_prob=value,numberofrepeat=numberofrepeat)[0] for value in plot_values_M_EmpProb]
        plot_values_MoSChange_EmpProb = 1-df_MoSChange_plot["correct_empirical_probability_"+noise_name]
        plot_values_MoSChange_upper_CI_bound = [Wilson_CI(emp_prob=value,numberofrepeat=numberofrepeat)[1] for value in plot_values_MoSChange_EmpProb]
        plot_values_MoSChange_lower_CI_bound = [Wilson_CI(emp_prob=value,numberofrepeat=numberofrepeat)[0] for value in plot_values_MoSChange_EmpProb]
        #term is squared for the variance to keep the formula correctly scaled

        if plot_type=="EmpProb":
            plt.plot(1-df_M_plot["m"],plot_values_M_EmpProb, color=color, linestyle = "solid", label = "$\\mathcal{M}$ satisfying "+DP_label)
            plt.plot(1-df_MoSChange_plot["m"],plot_values_MoSChange_EmpProb, color=color, linestyle = "dotted", label = "$\\mathcal{M}\\circ\\mathcal{S}$ satisfying "+DP_label)
        else: #elif plot_type=="EmpProb+SD":
            plt.plot(1-df_M_plot["m"],plot_values_M_EmpProb, color=color, linestyle = "solid", label = "$\\mathcal{M}$ satisfying "+DP_label)
            plt.fill_between(1-df_M_plot["m"], plot_values_M_lower_CI_bound, plot_values_M_upper_CI_bound, color=color, alpha=0.2)
            plt.plot(1-df_MoSChange_plot["m"],plot_values_MoSChange_EmpProb, color=color, linestyle = "dotted", label = "$\\mathcal{M}\\circ\\mathcal{S}$ satisfying "+DP_label)
            plt.fill_between(1-df_MoSChange_plot["m"], plot_values_MoSChange_lower_CI_bound, plot_values_MoSChange_upper_CI_bound, color=color, alpha=0.2)
            

    #Axis limits and legend
    ax.set_xlim([0,1])
    ax.set_ylim([0,1])
    ax.legend()
    ax.set(xlabel='Sampling rate',ylabel='Empirical probability of not returning mode')

    #Print Title
    title = mechanism_name 
    if include_title==True:
        ax.set_title(title)

    file_name_end = "_"+ plot_type +".pdf"
    plt.savefig(plot_path_start+file_name_end, bbox_inches='tight')
    plt.close()