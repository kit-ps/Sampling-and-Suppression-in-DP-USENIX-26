from mainfunctions import *

sens = 9E-6 #Rounding sensitivity 
error = 1E-4 #Maximum allowed error for alert message

coleps=[]
colm=[]
colM=[]
colDiffEvol = []
colHypValue = []
colDiff = []

##We check for all m and M in 0.01, 0.02, ..., 0.98, 0.99 such that m<M
for m100 in [m for m in range(1,99)]: #m runs from 0.01 to 0.98 (0.99 is not necessary)
    for M100 in [M for M in range(m100+1,100)]: #M runs from m+0.01 to 0.99
        m_it=m100/100
        M_it=M100/100
        coleps.append(0)
        colm.append(m_it)
        colM.append(M_it)
        #time_start=time.time()
        diff_evol_result=diff_evol_eps0(m_it,M_it)
        value=-diff_evol_result['fun']
        th_value=theoretical_eps0(m_it,M_it) ##Different function
        colDiffEvol.append(n(value))
        colHypValue.append(n(th_value[1]))
        colDiff.append(n(value-th_value[1]))
        if(value>th_value[1]+sens):
            print("\nLarger value at (0,",m_it,",",M_it,"): ",value,"+",th_value,"diff:",n(value-th_value[1]))
            print(diff_evol_result)
        if(abs(th_value[1]-value)>error):
            print("\nUntight value at (0,",m_it,",",M_it,"): ",value,"+",th_value,"diff:",n(value-th_value[1]))
            print(diff_evol_result)

d={'Epsilon': coleps, 'm': colm, 'M': colM, 'DiffEvol': colDiffEvol, 'HypValue': colHypValue, 'Difference': colDiff}
df = pd.DataFrame(data=d)
df.to_csv('output_epsilon0.csv',index=False,sep=';')

print("Minimum difference (empirical - theoretical): ", df["Difference"].min())
print("Maximum difference (empirical - theoretical): ", df["Difference"].max())  





