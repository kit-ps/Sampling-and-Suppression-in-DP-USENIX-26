from sage.all import *
from scipy.optimize import rosen, differential_evolution
import pandas as pd
import numpy as np

N_max = 1E9 #Maximum database size to be checked
tol_diff_evol = 1E-6 #Tolerance of the differential evolution
maxiter_diff_evol = 2500 #Maximum number of iterations in differential evolution

### Computations for epsilon>0
##Empirical maximization of the function (maximization with respect to N, pJ, pk, c, and t).
#The maximum according to differential_evolution over m and M
def diff_evol(eps,m,M):
    F=exp(eps)
    
    ##Functions to maximize (differential_evolution searches for minimum, hence the -)
    #Edges
    def function_edges(theta):
        N=theta[0]
        pJ=theta[1]
        pk=theta[2]
        c=theta[3]
        t=theta[4]

        if N * (1 - pJ) - 1 < 0:  ##Limit in the case N=|J| (i.e., N-J<1)
            return -(ln(F - (F - 1) * (m + N * c) / (N + 1)) + N * ln((N + c / m) / (N + 1)))
        else:
            B = ((N - 1) * (M + m) - (pJ * N * c + (N * (1 - pJ) - 1) * (1 - pk) * m + t)) / (N + (N * (1 - pJ) - 1) * pk - 2)
            U = ((N - 1) * (M + m) - (pJ * N * c + (N * (1 - pJ) - 1) * (1 - pk) * m)) / (N + (N * (1 - pJ) - 1) * pk - 1)
            S = pJ * N * c + t + (N * (1 - pJ) - 1) * ((1 - pk) * m + pk * B)
            if t > U:
                return 0
            else:
                f1 = ln(F - (F - 1) * (m + S) / (N + 1))
                f2 = ln((N + (1 - t) / (1 - ((N - 2) * (t - m) + S) / N)) / (N + 1))
                f3 = ((N * (1 - pJ) - 1) * (1 - pk)) * ln((N + (1 - m) / (1 - S / N)) / (N + 1))
                f4 = ((N * (1 - pJ) - 1) * pk) * ln((N + (1 - B) / (1 - (m + M * (N - 1)) / N)) / (N + 1))
                f5 = (N * pJ) * ln((N + c / m) / (N + 1))
                return -(f1 + f2 + f3 + f4 + f5)

    #Diagonal
    def function_diagonal(theta):
        N=theta[0]
        pJ=theta[1]
        c=theta[2]
        t=theta[3]
        
        U=((N-1)*(M+m)-N*pJ*c)/(N*(2-pJ)-2)
        if t>U:
            return 0
        else:
            f1 = ln(F-(F-1)*(m+N*pJ*c+N*(1-pJ)*t)/(N+1))
            f2 = N*pJ*ln((N+c/m)/(N+1))
            f3 = N*(1-pJ)*ln((N+(1-t)/(1-((N*(2-pJ)-2)*t-(N-2)*m+N*pJ*c)/N))/(N+1))
            return -(f1+f2+f3)
    
    result_edges = differential_evolution(function_edges,[(3,N_max),(0,1),(0,1),(m,M),(m,M)],tol=tol_diff_evol,x0=[3.1,0.5,0.5,(m+M)/2,(m+M)/2],maxiter=maxiter_diff_evol)
    result_diagonal = differential_evolution(function_diagonal,[(3,N_max),(0,1),(m,M),(m,M)],tol=tol_diff_evol,x0=[3.1,0.5,(m+M)/2,(m+M)/2],maxiter=maxiter_diff_evol)

    ##Functions to maximize when N goes to infinity. We do this since we cannot check at infinity.
    #Edges 
    def function_edges_limit(theta):
        pJ=theta[0]
        pk=theta[1]
        c=theta[2]

        if np.isclose(pJ,1):  ##Limit in the case N=|J|
            return -(ln(F - (F - 1) * c) + (c / m - 1))
        else:
            B = ((M + m) - (pJ * c + (1 - pJ) * (1 - pk) * m)) / (1 + (1 - pJ) * pk)
            #U = ((M + m) - (pJ * c + (1 - pJ) * (1 - pk) * m)) / (1 + (1 - pJ) * pk)
            S = pJ * c + (1 - pJ)*((1 - pk) * m + pk * B)

            f1 = ln(F - (F - 1) * S)
            f3 = ((1 - pJ) * (1 - pk)) * ((1 - m) / (1 - S) - 1)
            f4 = ((1 - pJ) * pk) * ((1 - B) / (1 - M)-1)
            f5 = pJ * (c / m - 1)
            return -(f1 + f3 + f4 + f5)

    #Diagonal
    def function_diagonal_limit(theta):
        pJ=theta[0]
        c=theta[1]
        t=theta[2]
        
        U=((M+m)-pJ*c)/(2-pJ)
        if t>U:
            return 0
        else:
            f1 = ln(F-(F-1)*(pJ*c+(1-pJ)*t))
            f2 = pJ*(c/m - 1)
            f3 = (1-pJ)*((1-t)/(1-((2-pJ)*t-m+pJ*c)) - 1)
            return -(f1+f2+f3)
    
    result_edges_limit = differential_evolution(function_edges_limit,[(0,1),(0,1),(m,M)],tol=tol_diff_evol,x0=[0.5,0.5,(m+M)/2],maxiter=maxiter_diff_evol)
    result_diagonal_limit = differential_evolution(function_diagonal_limit,[(0,1),(m,M),(m,M)],tol=tol_diff_evol,x0=[0.5,(m+M)/2,(m+M)/2],maxiter=maxiter_diff_evol)

    ##Computation of degenerate cases
    ##N=1
    V11 = min(M,max(m,F/(F-1)-m))
    H11 = ln(F-(F-1)*(m+V11)/2) + ln((1+V11/m)/2)
    H10m = ln(F-(F-1)*m)
    H10M = ln(F-(F-1)*(m+M)/2)+ ln((1+(1-M)/(1-m))/2)

    ##N=2
    V22 = V11#min(M,max(m,F/(F-1)-m))
    H22 = ln(F-(F-1)*(m+2*V22)/3)+2*ln((2+V22/m)/3)
    V21a = min(M,max(m,3/2*F/(F-1)-2*m))
    H21a = ln(F-(F-1)*(2*m+V21a)/3)+ln((2+V21a/m)/3)
    def f21b(V21b):
        return ln(F-(F-1)*(2*m+V21b)/3) + ln((2+V21b/((V21b+m)/2))/3) + ln((2+(1-m)/(1-(V21b+m)/2))/3)
    H21b = max(find_local_maximum(f21b,m,M)[0],f21b(m),f21b(M))
    #H20m is same as H10m
    H20M = ln(F-(F-1)*(m+2*M)/3)+2*ln((2+(1-M)/(1-(M+m)/2))/3)
    
    DegenerateValues = max(H11,H10M,H22,H21a,H21b,H20M)

    ##Check if degenerate cases are larger or not
    if DegenerateValues >= max(-result_edges['fun'],-result_diagonal['fun'],-result_edges_limit['fun'],-result_diagonal_limit['fun']):
        print("Degenerate value are larger for (",eps,",",m,",",M,").",DegenerateValues) 

    #Return larger value
    if -result_edges['fun']>=max(-result_diagonal['fun'],-result_edges_limit['fun'],-result_diagonal_limit['fun']):
        return result_edges
    elif -result_diagonal['fun']>=max(-result_edges['fun'],-result_edges_limit['fun'],-result_diagonal_limit['fun']): 
        return result_diagonal
    elif -result_edges_limit['fun']>=max(-result_edges['fun'],-result_diagonal['fun'],-result_diagonal_limit['fun']):
        return result_edges_limit
    elif -result_diagonal_limit['fun']>=max(-result_edges['fun'],-result_diagonal['fun'],-result_edges_limit['fun']): 
        return result_diagonal_limit


##Theoretical maximum: N to infinity, B=M, pk=0 or pk=1, and pJ is solution to cubic polynomial
##Finding optimum of pJ with computed formula
def calculate_p1(F,m,M):
    a = (F-1)*M/m*(M-m)**2
    b = -(M-m)/m*((m**2-4*M*m+2*M)*(F-1)+F*M)
    c = (1-m)/m*((F-1)*(2*(m**2)-4*M*m-m)+(3*F-1)*M)
    d = -(1-m)*((F-1)*(m-2)+F/m)
    D0 = b**2-3*a*c
    D1 = 2*b**3-9*a*b*c+27*(a**2)*d
    R = sqrt(D0**3)
    case = D1**2-4*(D0**3)
    if case>=0:
        cubicroot1=real_nth_root((D1+sqrt(case))/2,3)
        cubicroot2=real_nth_root((D1-sqrt(case))/2,3)
        V = -1/(3*a)*(b+cubicroot1+cubicroot2)
    else:
        V = -1/(3*a)*(b+2*sqrt(D0)*cos(1/3*arccos(D1/(2*R))))

    if V<0:
        return 0
    elif V>1:
        return 1
    else:
        return V 

def calculate_p2(F,m,M):
    a = (F-(F-1)*m)/m
    b = -(6*F-(F-1)*(M+5*m))/m
    c = 1/(m*(1-M))*(m*((F-1)*(m+9*M-9)-F)+4*M*((F-1)*M-4*F+1)+12*F)
    d = -(2*F-(F-1)*(M+m))*((4-m-4*M)/(m*(1-M)))+2*(F-1)
    D0 = b**2-3*a*c
    D1 = 2*b**3-9*a*b*c+27*(a**2)*d
    R = sqrt(D0**3)
    V=-1/(3*a)*(b+2*sqrt(D0)*cos(1/3*arccos(D1/(2*R))))
    if V<0:
        return 0
    elif V>1:
        return 1
    else:
        return V

def theoretical_eps(eps,m,M): 
    F=exp(eps)

    p1=calculate_p1(F,m,M)
    p2=calculate_p2(F,m,M)
    
    #Maximums of each individual function:
    L1 = ln(F-(F-1)*(p1*M+(1-p1)*m))+p1*M/m+(1-p1)*(1-m)/(1-(p1*M+(1-p1)*m))-1
    L2 = ln(F-(F-1)*(p2*M+(1-p2)*(M+m-p2*M)/(2-p2)))+p2*M/m+(1-p2)*(1-((M+m-p2*M)/(2-p2)))/(1-M)-1
    
    if L1 >= L2:
        return [0,L1]
    else:
        return [1,L2]




###Computation for epsilon = 0
##Maximum of function F (maximization with respect to N, pJ, pk, c, and t). We do the maximum over the ln. 
#The maximum according to differential_evolution over m and M
def diff_evol_eps0(m,M):
    #Function to maximize (differential_evolution searches for minimum, hence the -)
    #Edges
    def function_edges(theta):
        N=theta[0]
        pJ=theta[1]
        pk=theta[2]
        c=theta[3]
        t=theta[4]

        if N*(1-pJ)-1<0: ##Limit in the case N=|J| (i.e., N-J<1)
            return -(N*ln((N+c/m)/(N+1)))
        else:
            B = ((N-1)*(M+m)-(pJ*N*c+(N*(1-pJ)-1)*(1-pk)*m+t))/(N+(N*(1-pJ)-1)*pk-2)
            U = ((N-1)*(M+m)-(pJ*N*c+(N*(1-pJ)-1)*(1-pk)*m))/(N+(N*(1-pJ)-1)*pk-1)
            S = pJ*N*c+t+(N*(1-pJ)-1)*((1-pk)*m+pk*B)
            if t>U:
                return 0
            else:
                f2 = ln((N+(1-t)/(1-((N-2)*(t-m)+S)/N))/(N+1))
                f3 = ((N*(1-pJ)-1)*(1-pk))*ln((N+(1-m)/(1-S/N))/(N+1))
                f4 = ((N*(1-pJ)-1)*pk)*ln((N+(1-B)/(1-(m+M*(N-1))/N))/(N+1))
                f5 = (N*pJ)*ln((N+c/m)/(N+1))
                return -(f2+f3+f4+f5)

    #Diagonal
    def function_diagonal(theta):
        N=theta[0]
        pJ=theta[1]
        c=theta[2]
        t=theta[3]
        
        U=((N-1)*(M+m)-N*pJ*c)/(N*(2-pJ)-2)
        if t>U:
            return 0
        else:
            f2 = N*pJ*ln((N+c/m)/(N+1))
            f3 = N*(1-pJ)*ln((N+(1-t)/(1-((N*(2-pJ)-2)*t-(N-2)*m+N*pJ*c)/N))/(N+1))
            return -(f2+f3)
    
    result_edges = differential_evolution(function_edges,[(3,N_max),(0,1),(0,1),(m,M),(m,M)],tol=tol_diff_evol,x0=[3,0.5,0.5,(m+M)/2,(m+M)/2],maxiter=maxiter_diff_evol)
    result_diagonal = differential_evolution(function_diagonal,[(3,N_max),(0,1),(m,M),(m,M)],tol=tol_diff_evol,x0=[3,0.5,(m+M)/2,(m+M)/2],maxiter=maxiter_diff_evol)

    ##Computation of degenerate cases
    ##N=1
    H1 = ln((1+M/m)/2)

    ##N=2
    H22 = 2*ln((2+M/m)/3)
    #H21a = ln((2+M/m)/3)
    H21b = ln((2+M/((M+m)/2))/3)+ln((2+(1-m)/(1-(M+m)/2))/3)
    #H20 = 0
    
    DegenerateValues = max(H1,H22,H21b)

    ##Check if degenerate cases are larger or not
    if DegenerateValues >= max(-result_edges['fun'],-result_diagonal['fun']):
        print("Degenerate value are larger for (0,",m,",",M,"): ",DegenerateValues) 

    #Return larger value
    if -result_edges['fun']>=-result_diagonal['fun']:
        return result_edges
    else: 
        return result_diagonal

##Theoretical maximum: N to infinity, B=M, pk=0 or pk=1, and pJ is solution to cubic polynomial
##Finding optimum of p with computed formula
#Theoretical formula for eps=0
def theoretical_eps0(m,M): 
    V1 = (1-m)/(M-m)-sqrt(M*m*(1-m)*(1-M))/(M*(M-m))
    p1 = max_symbolic(0,min_symbolic(1,V1))
    V2 = 2-sqrt(m*(1-M))/(1-M)
    p2 = max_symbolic(0,min_symbolic(1,V2))
     
    #Maximums of each individual function:
    L1 = p1*M/m+(1-p1)*(1-m)/(1-(p1*M+(1-p1)*m))-1
    L2 = p2*M/m+(1-p2)*(1-((M+m-p2*M)/(2-p2)))/(1-M)-1
    
    if L1 >= L2:
        return [0,L1]
    else:
        return [1,L2]


###Combination of both epsilons
def theoretical_eps_combined(eps,m,M):
    if eps==0:
        return theoretical_eps0(m,M)
    else: 
        return theoretical_eps(eps,m,M)




#### Iterations for epsilon>0
sens = 9E-6 #Rounding sensitivity 
error = 1E-3 #Maximum allowed error for alert message

def iteration(eps):

    np.random.seed(int.from_bytes(os.urandom(4),"big"))

    coleps = []
    colm = []
    colM = []
    colDiffEvol = []
    colHypValue = []
    colDiff = []
    ##We check for all m and M in 0.01, 0.02, ..., 0.98, 0.99 such that m<M
    for m100 in [m for m in range(1,99)]: #m runs from 0.01 to 0.98 (0.99 is not necessary)
        for M100 in [M for M in range(m100+1,100)]: #M runs from m+0.01 to 0.99
            m_it=m100/100
            M_it=M100/100
            coleps.append(eps)
            colm.append(m_it)
            colM.append(M_it)
            #time_start=time.time()
            diff_evol_result=diff_evol(eps,m_it,M_it)
            value=-diff_evol_result['fun']
            th_value=theoretical_eps(eps,m_it,M_it) ##Different function
            colDiffEvol.append(n(value))
            colHypValue.append(n(th_value[1]))
            colDiff.append(n(value-th_value[1]))

            if (value > th_value[1] + sens):
                print("\nLarger value at (", eps, ",", m_it, ",", M_it, "): ", value, "+", th_value, "diff:", n(value - th_value[1]), "\n", diff_evol_result, flush=True)

            if (abs(th_value[1] - value) > error):
                print("\nUntight value at (", eps, ",", m_it, ",", M_it, "): ", value, "+", th_value, "diff:", n(value - th_value[1]), "\n", diff_evol_result, flush=True)

    return coleps,colm,colM,colDiffEvol,colHypValue,colDiff