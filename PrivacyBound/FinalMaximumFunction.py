from mainfunctions import *
from multiprocessing import Pool, Lock, Manager
import tqdm

#Open config file
import sys
sys.path.append("..")
from config import *

gcoleps=[]
gcolm=[]
gcolM=[]
gcolDiffEvol = []
gcolHypValue = []
gcolDiff = []

print("Checking function:")
pbar = tqdm.tqdm(total=1799721) #(200+80+91)*(99*98/2) (number of eps times number of m and M)

###The computation for eps==0 is done separately since it uses other functions
jobs_eps0=[]
##We check for all m and M in 0.01, 0.02, ..., 0.98, 0.99 such that m<M
for m100 in [m for m in range(1,99)]: #m runs from 0.01 to 0.98 (0.99 is not necessary)
    for M100 in [M for M in range(m100+1,100)]: #M runs from m+0.01 to 0.99
        m_it=m100/100
        M_it=M100/100

        jobs_eps0.append((m_it,M_it))

with Pool(N_CORES) as pool:
    for result in pool.imap(iteration_eps0, jobs_eps0):
        gcoleps.append(0)
        gcolm.append(result[1])
        gcolM.append(result[2])
        gcolDiffEvol.append(result[3])
        gcolHypValue.append(result[4])
        gcolDiff.append(result[5])
        
        pbar.update(1)

###Computation for eps>0
# We cover the epsilons in 
# *eps between 0.01 and 1.99 (step 0.01)
# *eps between 2 and 9.9 (step 0.1)
# *eps between 10 and 100 (step 1)
list_epsilons = [ep/100 for ep in range(1,200)] + [ep/10 for ep in range(20,100)] + [float(ep) for ep in range(10,101)]

jobs=[]
for ep in list_epsilons:
    ##We check for all m and M in 0.01, 0.02, ..., 0.98, 0.99 such that m<M
    for m100 in [m for m in range(1,99)]: #m runs from 0.01 to 0.98 (0.99 is not necessary)
        for M100 in [M for M in range(m100+1,100)]: #M runs from m+0.01 to 0.99
            m_it=m100/100
            M_it=M100/100

            jobs.append((ep,m_it,M_it))

with Pool(N_CORES) as pool:
    for result in pool.imap(iteration, jobs):
        gcoleps.append(result[0])
        gcolm.append(result[1])
        gcolM.append(result[2])
        gcolDiffEvol.append(result[3])
        gcolHypValue.append(result[4])
        gcolDiff.append(result[5])
        
        pbar.update(1)

pbar.close()

d={'Epsilon': gcoleps, 'm': gcolm, 'M': gcolM, 'DiffEvol': gcolDiffEvol, 'HypValue': gcolHypValue, 'Difference': gcolDiff}
df = pd.DataFrame(data=d)
df.to_csv('output.csv',index=False,sep=';')

print("FinalMaximumFunction.py:")
print("Minimum difference (computational - theoretical): ", df["Difference"].min())
print("Maximum difference (computational - theoretical): ", df["Difference"].max(),"\n")  