import matplotlib.pyplot as plt
import numpy as np
import warnings
warnings.filterwarnings("ignore", message="invalid value")
warnings.filterwarnings("ignore", message="divide by zero")

###Computation of epsilon^S
def calculate_V1(F,m,M):
    a1=(F-1) * (M/m) * (np.power((M-m), 2))
    b1=-((M-m)/m) * ((np.power(m,2)-4*M*m +2*M)*(F-1) + F*M)
    c1= ((1-m)/m)* ( (F-1) * (2* np.power(m, 2)-4*M*m-m) + (3*F-1)*M )
    d1=-(1-m) *( (F-1) * (m-2) + (F/m) )

    D10=np.power(b1,2)-3*a1*c1
    D11= (2*np.power(b1,3)) - (9*a1*b1*c1) + (27*np.power(a1, 2)* d1)
    R1=np.sqrt(np.power(D10, 3))
    case=np.power(D11, 2)-4*np.power(D10, 3)
    V1 = np.where(case>=0, 
        -1/(3*a1)*(b1+np.cbrt((D11+np.sqrt(case))/2)+np.cbrt((D11-np.sqrt(case))/2)),
        -1/(3*a1)*(b1+2*np.sqrt(D10)*np.cos(1/3*np.arccos(D11/(2*R1)))) )

    return V1

def calculate_V2(F,m,M):
    a2 = (F-(F-1)*m)/m
    b2 = -(6*F-(F-1)*(M+5*m))/m
    c2 = (1/(m*(1-M))) * (m*((F-1)*(m+9*M-9)-F)+4*M*((F-1)*M-4*F+1)+12*F)
    d2 = -(2*F-(F-1)*(M+m))*((4-m-4*M)/(m*(1-M)))+2*(F-1)
    D20 = np.power(b2, 2)-3*a2*c2
    D21 = 2*np.power(b2, 3) - 9*a2*b2*c2+27*np.power(a2, 2)*d2
    R2 = np.sqrt(np.power(D20,3))
    V2=-1/(3*a2)*(b2+2*np.sqrt(D20)*np.cos((1/3)*np.arccos(D21/(2*R2))))
    
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

##Function: returns if V1>=1 (or, equivalently, V2>=1)
def V_larger_1_all(eps,m,M):
    F=np.exp(eps)
    return np.where(m>M,float('nan'),np.where(m==M,float('nan'), np.where(eps==0,2-(np.sqrt(m*(1-M)))/(1-M),calculate_V2(F,m,M))))

##Function: returns V1 if L1 achieves maximum
def L1_achieves_max_returnV1(eps,m,M):
    F=np.exp(eps)
    V1=np.where(eps==0,((1-m)/(M-m))-np.sqrt(M*m*(1-m)*(1-M))/(M*(M-m)),calculate_V1(F,m,M))
    p1=np.where(V1>1,1,np.where(V1<0,0,V1))
    V2=np.where(eps==0,2-(np.sqrt(m*(1-M)))/(1-M),calculate_V2(F,m,M))
    p2=np.where(V2>1,1,np.where(V2<0,0,V2))
          
    #Maximums of each individual function:
    L1 = np.log( F-(F-1)*(p1*M+ (1-p1)*m) ) + p1*(M/m) + (1-p1)*(1-m)/(1-(p1*M+(1-p1)*m))-1
    L2 = np.log( F-(F-1)*(p2*M+(1-p2)*(M+m-p2*M)/(2-p2)) ) + p2*(M/m) +(1-p2)*(1-((M+m-p2*M)/(2-p2)))/(1-M)-1
    L3 = -(np.log(1/F + (1-1/F)*M)) + (1- (1-M)/(1-m))

    return np.where(L1>=L2,np.where(L1>=L3,V1,float('nan')),float('nan'))

def L1_achieves_max_returnV1_all(eps,m,M):
    F=np.exp(eps)
    return np.where(m>M,float('nan'),np.where(m==M,float('nan'), L1_achieves_max_returnV1(eps,m,M)))

##Function: returns V2 of L2 that achieves maximum
def L2_achieves_max_returnV2(eps,m,M):
    F=np.exp(eps)
    V1=np.where(eps==0,((1-m)/(M-m))-np.sqrt(M*m*(1-m)*(1-M))/(M*(M-m)),calculate_V1(F,m,M))
    p1=np.where(V1>1,1,np.where(V1<0,0,V1))
    V2=np.where(eps==0,2-(np.sqrt(m*(1-M)))/(1-M),calculate_V2(F,m,M))
    p2=np.where(V2>1,1,np.where(V2<0,0,V2))
          
    #Maximums of each individual function:
    L1 = np.log( F-(F-1)*(p1*M+ (1-p1)*m) ) + p1*(M/m) + (1-p1)*(1-m)/(1-(p1*M+(1-p1)*m))-1
    L2 = np.log( F-(F-1)*(p2*M+(1-p2)*(M+m-p2*M)/(2-p2)) ) + p2*(M/m) +(1-p2)*(1-((M+m-p2*M)/(2-p2)))/(1-M)-1
    L3 = -(np.log(1/F + (1-1/F)*M)) + (1- (1-M)/(1-m))

    return np.where(L1>=L2,float('nan'),np.where(L2>=L3,V2,float('nan')))

def L2_achieves_max_returnV2_all(eps,m,M):
    F=np.exp(eps)
    return np.where(m>M,float('nan'),np.where(m==M,float('nan'), L2_achieves_max_returnV2(eps,m,M)))

##Function: whether L3 achieves max: Positive value if so, negative otherwise
def L3_achieves_max(eps,m,M):
    F=np.exp(eps)
    V1=np.where(eps==0,((1-m)/(M-m))-np.sqrt(M*m*(1-m)*(1-M))/(M*(M-m)),calculate_V1(F,m,M))
    p1=np.where(V1>1,1,np.where(V1<0,0,V1))
    V2=np.where(eps==0,2-(np.sqrt(m*(1-M)))/(1-M),calculate_V2(F,m,M))
    p2=np.where(V2>1,1,np.where(V2<0,0,V2))
          
    #Maximums of each individual function:
    L1 = np.log( F-(F-1)*(p1*M+ (1-p1)*m) ) + p1*(M/m) + (1-p1)*(1-m)/(1-(p1*M+(1-p1)*m))-1
    L2 = np.log( F-(F-1)*(p2*M+(1-p2)*(M+m-p2*M)/(2-p2)) ) + p2*(M/m) +(1-p2)*(1-((M+m-p2*M)/(2-p2)))/(1-M)-1
    L3 = -(np.log(1/F + (1-1/F)*M)) + (1- (1-M)/(1-m))

    return L3-np.where(L1>=L2,L1,L2)

def L3_achieves_max_all(eps,m,M):
    F=np.exp(eps)
    return np.where(m>M,float('nan'),np.where(m==M,0, L3_achieves_max(eps,m,M)))




##### Contour plots of epsilon^S with respect to m and M for various epsilons
## Epsilon levels selected
countours_eps_suppression=[0,0.1,0.25,0.5,1,2,5,10]

## Colors
my_green = (15/255, 247/255, 48/255)
my_red = (206/255, 10/255, 0/255)
my_blue = "dodgerblue"
my_colors = []
length = len(countours_eps_suppression)
for i in range(length):
    temp_color = ((i*my_red[0]+(length-i)*my_green[0])/length, (i*my_red[1]+(length-i)*my_green[1])/length, (i*my_red[2]+(length-i)*my_green[2])/length)
    my_colors.append(temp_color)

##Granularity of evaluated plot
delta = 0.0025
x = np.arange(0+delta, 1-delta, delta)
y = np.arange(0+delta, 1-delta, delta)
p, q = np.meshgrid(x, y)

##Function of epsilon^S (function to plot)
eps_suppression = lambda n, x, y: theoretical_eps_all(eps=n,m=x,M=y)

for eps in [0,0.25,0.5,0.75,1,2]:
    z=eps_suppression(eps,p,q)

    fig, ax = plt.subplots()

    # Plot every area and line (do not plot the line corresponding to eps^S=eps, which is printed in a different color)
    CS_areas = plt.contourf(p, q, z, countours_eps_suppression, colors=my_colors, alpha=0.15, extend="max")
    CS_lines = plt.contour(p, q, z, [value for value in countours_eps_suppression if value is not eps], colors=my_colors)

    # Plot every area where eps^S<=eps and line eps^S=eps 
    if eps!=0:
        CSblue_area = plt.contourf(p, q, z, [0,eps], colors=my_blue, alpha=0.4)
    CSblue_line = plt.contour(p, q, z, [eps], colors=my_blue)

    # Axis limits
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])

    # Labels of contour lines
    ax.clabel(CS_lines, [value for value in countours_eps_suppression if value is not eps], inline=1, fontsize=10, fmt = "$\\varepsilon^{\\mathcal{S}}=%g$")
    ax.clabel(CSblue_line, [eps], inline=1, fontsize=10, fmt = "$\\varepsilon^{\\mathcal{S}}=%g$")

    # Label of selected epsilon
    plt.text(0.65,0.25,"$\\varepsilon="+str(eps)+"$",fontsize=25,ha="center",va="center")

    plt.savefig("plot_eps_suppression_"+str(eps)+".pdf",bbox_inches='tight')
    plt.close()


### Plot of areas with simplified expression (color coded)
# Functions
functionVlarger1 = lambda n, x, y: V_larger_1_all(eps=n,m=x,M=y)
functionV1 = lambda n, x, y: L1_achieves_max_returnV1_all(eps=n,m=x,M=y)
functionV2 = lambda n, x, y: L2_achieves_max_returnV2_all(eps=n,m=x,M=y)
functionL3 = lambda n, x, y: L3_achieves_max_all(eps=n,m=x,M=y)

fig, ax = plt.subplots()
for eps in [0,0.5,1,2,5]:
    ## Red region
    zVlarger1=functionVlarger1(eps,p,q)

    CSVlarger1_area = plt.contourf(p, q, zVlarger1, [1,float('inf')], colors=["red"], alpha=0.1)
    CSVlarger1_line = plt.contour(p, q, zVlarger1, [1], colors=["red"])

    zV1=functionV1(eps,p,q)

    ##Plot V1 is larger than 1 (Obsolete, CSVlarger1 considers this case)
    #CSV1_1_area = plt.contourf(p, q, zV1, [1,float('inf')], colors=["red"], alpha=0.4)
    #CSV1_1_line = plt.contour(p, q, zV1, [1], colors=["red"])

    #Plot V1 is smaller than 0
    CSV1_0_area = plt.contourf(p, q, zV1, [float('-inf'),0], colors=["orange"], alpha=0.1)
    CSV1_0_line = plt.contour(p, q, zV1, [0], colors=["orange"])

    ## Green region
    zV2=functionV2(eps,p,q)

    ##Plot V2 is larger than 1 (Obsolete, CSVlarger1 considers this case)
    #CSV2_1_area = plt.contourf(p, q, zV2, [1,float('inf')], colors=["red"], alpha=0.4)
    #CSV2_1_line = plt.contour(p, q, zV2, [1], colors=["red"])

    #Plot V2 is smaller than 0
    CSV2_0_area = plt.contourf(p, q, zV2, [float('-inf'),0], colors=["green"], alpha=0.1)
    CSV2_0_line = plt.contour(p, q, zV2, [0], colors=["green"])

    ## Purple region
    #Plot L3 is larger than L1 and L2
    zL3=functionL3(eps,p,q)

    CSL3_area = plt.contourf(p, q, zL3, [0,float('inf')], colors=["purple"], alpha=0.1)
    CSL3_line = plt.contour(p, q, zL3, [0], colors=["purple"])

    ## Axis limits
    ax.set_xlim([0,1])
    ax.set_ylim([0,1])

    ## Print out labels
    ax.clabel(CSVlarger1_line, [1], inline=1, fontsize=10, fmt = "$\\varepsilon="+str(eps)+"$")
    #ax.clabel(CSV1_1_line, [1], inline=1, fontsize=10, fmt = "$\\varepsilon="+str(eps)+"$")
    ax.clabel(CSV1_0_line, [0], inline=1, fontsize=10, fmt = "$\\varepsilon="+str(eps)+"$")
    #ax.clabel(CSV2_1_line, [1], inline=1, fontsize=10, fmt = "$\\varepsilon="+str(eps)+"$")
    ax.clabel(CSV2_0_line, [0], inline=1, fontsize=10, fmt = "$\\varepsilon="+str(eps)+"$")
    ax.clabel(CSL3_line, [0], inline=1, fontsize=10, fmt = "$\\varepsilon="+str(eps)+"$")


#Print legend
plt.text(0.7,0.35,"$\\ln(\\mathrm{e}^{\\varepsilon}-(\\mathrm{e}^{\\varepsilon}-1)M)+\\frac{M}{m}-1$",fontsize=13,ha="center",va="center",color="red")
plt.text(0.7,0.25,"$\\ln(\\mathrm{e}^{\\varepsilon}-(\\mathrm{e}^{\\varepsilon}-1)\\frac{M+m}{2})+\\frac{1-\\frac{M+m}{2}}{1-M}-1$",fontsize=13,ha="center",va="center",color="green")
plt.text(0.7,0.15,"$-\\ln(\\mathrm{e}^{-\\varepsilon}+(1-\\mathrm{e}^{-\\varepsilon})M)+1-\\frac{1-M}{1-m}$",fontsize=13,ha="center",va="center",color="purple")

plt.savefig("plot_simplified_areas.pdf",bbox_inches='tight')
plt.close()