from mainfunctions import *
from sage.all import *
import pandas as pd

sens = 9E-6 #Rounding sensitivity 
error = 1E-4 #Maximum allowed error for alert message

N_max = 1E9 #Maximum N checked

coleps=[]
colm=[]
colM=[]
colDiffEvol = []
colHypValue = []
colDiff = []

## Iterations for inverse computation
def iteration_inverse(eps):
    
    np.random.seed(int.from_bytes(os.urandom(4),"big"))

    F=exp(eps)

    ##We check for all m and M in 0.01, 0.02, ..., 0.98, 0.99 such that m<M
    for m100 in [m for m in range(1,99)]: #m runs from 0.01 to 0.98 (0.99 is not necessary)
        for M100 in [M for M in range(m100+1,100)]: #M runs from m+0.01 to 0.99
            m_it=m100/100
            M_it=M100/100
            coleps.append(eps)
            colm.append(m_it)
            colM.append(M_it)
            
            ##Theoretical maximum
            L3 = -ln(1/F+(1-1/F)*M_it)+(1-(1-M_it)/(1-m_it))
            L4 = -ln(1/F+(1-1/F)*m_it)+(1-m_it/M_it)
            th_value=max(L3,L4)

            ##Numerical maximum
            def L3_numerical(N):
                return -ln(1/F+(1-1/F)*(m_it+N*M_it)/(N+1))+N*ln((N+1)/(N+(1-M_it)/(1-m_it)))
            maximum_value = find_local_maximum(L3_numerical,0,N_max)

            value=max(maximum_value[0], L4)
            
            colDiffEvol.append(n(value))
            colHypValue.append(n(th_value))
            colDiff.append(n(value-th_value))
            if(value>th_value+sens):
                print("\nLarger value at (",eps,",",m_it,",",M_it,"): ",value,"+",th_value,"diff:",n(value-th_value))
                print(maximum_value)
            if(abs(th_value-value)>error):
                print("\nUntight value at (",eps,",",m_it,",",M_it,"): ",value,"+",th_value,"diff:",n(value-th_value))
                print(maximum_value)

            #Check whether the superfluous term is actually superfluous
            final_maximum = theoretical_eps_combined(eps,m_it,M_it)
            superfluous_term = L4
            if(superfluous_term>final_maximum[1]+sens):
                print("\nSuperfluous term larger at (",eps,",",m_it,",",M_it,"): ",final_maximum,"+",superfluous_term,"diff:",n(superfluous_term-final_maximum))

### eps between 0 and 1.99 (step 0.01)
for eps in [ep/100 for ep in range(0,200)]:    
    iteration_inverse(eps)
### eps between 2 and 9.9 (step 0.1)
for eps in [ep/10 for ep in range(20,100)]:  
    iteration_inverse(eps)
### eps between 10 and 100 (step 1)
for eps in [float(ep) for ep in range(10,101)]: 
    iteration_inverse(eps)

d={'Epsilon': coleps, 'm': colm, 'M': colM, 'DiffEvol': colDiffEvol, 'HypValue': colHypValue, 'Difference': colDiff}
df = pd.DataFrame(data=d)
df.to_csv('output_inverse.csv',index=False,sep=';')
    
print("Minimum difference (empirical - theoretical): ", df["Difference"].min())
print("Maximum difference (empirical - theoretical): ", df["Difference"].max())  




