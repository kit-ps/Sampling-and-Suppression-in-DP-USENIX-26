import numpy as np
import pandas as pd
import scipy
import os
from functools import lru_cache 

def calculate_delta_suppression(delta, m):
    delta_suppression=delta*(1-m)
    return delta_suppression

def calculate_delta_suppression_inverse(delta, m):
    return delta/(1-m)

def calculate_L1(m: float, M: float, eps: float)-> float:
    F=np.exp(eps)
    if eps==0:
        V1=((1-m)/(M-m))-np.sqrt(M*m*(1-m)*(1-M))/(M*(M-m))
    else:
        a1=(F-1) * (M/m) * (np.power((M-m), 2))
        b1=-((M-m)/m) * ((np.power(m,2)-4*M*m +2*M)*(F-1) + F*M)
        c1= ((1-m)/m)* ( (F-1) * (2* np.power(m, 2)-4*M*m-m) + (3*F-1)*M )
        d1=-(1-m) *( (F-1) * (m-2) + (F/m) )
    
        D10=np.power(b1,2)-3*a1*c1
        D11= (2*np.power(b1,3)) - (9*a1*b1*c1) + (27*np.power(a1, 2)* d1)
        R1=np.sqrt(np.power(D10, 3))
       #Options D1
        case=np.power(D11, 2)-4*np.power(D10, 3)
        if case>=0:
            cubic_pos=(D11 + np.sqrt(case))/2
            cubic_neg=(D11 - np.sqrt(case))/2
            V1= -(1/(3*a1))*(b1+np.cbrt(cubic_pos)+np.cbrt(cubic_neg))  
        else:
            V1=-(1/(3*a1))*(b1+2*np.sqrt(D10)*np.cos( (1/3) * np.arccos(D11/(2*R1)) ))  
    maxV1=np.amax([V1, 0]) 
    p=np.amin([1, maxV1])              
    L1=np.log(  F-(F-1)*(p*M+ (1-p)*m)  ) + p*(M/m) + (1-p)*(1-m)/(1-(p*M+(1-p)*m))-1
    
    return L1  

def calculate_L2(m: float, M: float, eps: float)-> float:
    F=np.exp(eps)
    if eps==0:
        V2= 2-(np.sqrt(m*(1-M)))/(1-M)
    else:
        a2 = (F-(F-1)*m)/m
        b2 = -(6*F-(F-1)*(M+5*m))/m
        c2 = (1/(m*(1-M))) * (m*((F-1)*(m+9*M-9)-F)+4*M*((F-1)*M-4*F+1)+12*F)
        d2 = -(2*F-(F-1)*(M+m))*((4-m-4*M)/(m*(1-M)))+2*(F-1)
        D20 = np.power(b2, 2)-3*a2*c2
        D21 = 2*np.power(b2, 3) - 9*a2*b2*c2+27*np.power(a2, 2)*d2
        R2 = np.sqrt(np.power(D20,3))
        V2=-1/(3*a2)*(b2+2*np.sqrt(D20)*np.cos((1/3)*np.arccos(D21/(2*R2))))
    maxV2=np.amax([V2, 0])
    p=np.amin([1, maxV2])
    inside_of_log=F-(F-1)*(p*M+(1-p)*(M+m-p*M)/(2-p))
    L2=np.log(inside_of_log) + p*(M/m) +(1-p)*(1-((M+m-p*M)/(2-p)))/(1-M)-1
    return L2

def calculate_L3(m: float, M: float, eps: float)-> float:
    F=np.exp(eps)
    L3= -(np.log(1/F + (1-1/F)*M))  + (1- (1-M)/(1-m))
    return L3

@lru_cache
def calculate_eps_suppression(m: float, M: float, eps: float)-> float:
    if (m==M):
        F=np.exp(eps)
        A=F-(F-1)*m
        B=1/(1/F + (1-1/F)*M)
        max=np.amax([A, B])
        eps_suppression= np.log(max)
        return eps_suppression
    else: 
        L1=calculate_L1(m, M, eps)
        L2=calculate_L2(m, M, eps)
        L3=calculate_L3(m, M, eps)
        eps_suppression=np.amax([L1, L2, L3])
        return eps_suppression

@lru_cache
def calculate_eps_suppression_inverse(m: float, M: float, eps: float):
    eps_suppression_for_eps0 = calculate_eps_suppression(m,M,0)

    if eps<eps_suppression_for_eps0:
        return float("nan")

    if np.isclose(eps,eps_suppression_for_eps0):
        return 0

    if np.isclose(m,M):
        return np.log((np.exp(eps)-m)/(1-m))

    def epsilon_inverse(theta):
        if theta[0]<0:
            return 100
        return np.abs(calculate_eps_suppression(m,M,theta[0])-eps)
    
    minimize_output = scipy.optimize.minimize(epsilon_inverse, x0 = eps, tol=1E-6)
    return minimize_output['x'][0]