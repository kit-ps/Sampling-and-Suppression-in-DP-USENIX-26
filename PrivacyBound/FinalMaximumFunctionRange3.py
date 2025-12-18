from mainfunctions import *
from multiprocessing import Pool, Lock, Manager

gcoleps = []
gcolm = []
gcolM = []
gcolDiffEvol = []
gcolHypValue = []
gcolDiff = []

### eps between 10 and 100 (step 1)
with Pool(64) as p:
    results = p.map(iteration,[float(ep) for ep in range(10,101)])

    for r in results:
        gcoleps.extend(r[0])
        gcolm.extend(r[1])
        gcolM.extend(r[2])
        gcolDiffEvol.extend(r[3])
        gcolHypValue.extend(r[4])
        gcolDiff.extend(r[5])

d={'Epsilon': gcoleps, 'm': gcolm, 'M': gcolM, 'DiffEvol': gcolDiffEvol, 'HypValue': gcolHypValue, 'Difference': gcolDiff}
df = pd.DataFrame(data=d)
df.to_csv('output_range3.csv',index=False,sep=';')

print("Minimum difference (empirical - theoretical): ", df["Difference"].min())
print("Maximum difference (empirical - theoretical): ", df["Difference"].max())    


