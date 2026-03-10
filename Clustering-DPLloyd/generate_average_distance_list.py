import pandas as pd
import numpy as np
from multiprocessing import Pool, Lock, Manager
import tqdm

#Open config file
import sys
sys.path.append("..")
from config import *

def norm_calculation(row1,row2):
    square_sum = 0 
    for i in range(len(row1)):
        square_sum = square_sum + (row1[i]-row2[i])**2
    return np.sqrt(square_sum)

def iteration(arg):
    df, row, max_norm_value, length_Element = arg
    total_sum=0
    for _, j in df.iterrows(): 
        dist=norm_calculation(row,j)/max_norm_value
        total_sum=total_sum + dist
    outlier_score=total_sum/(length_Element)
    return outlier_score

def generate_average_distance_list(file_name_output, df, columns, normalized_range_value=1):        
    
    max_norm_value = 2*normalized_range_value*np.sqrt(len(columns))

    outlier_score_list=[]
    length_Element=df.shape[0]
    
    #For every element in the database, compute its outlier score and add it to the list
    print("Generate average distance list (1/2):")
    pbar = tqdm.tqdm(total=length_Element)
    
    jobs=[]
    for _, row in df.iterrows():
        jobs.append((df,row,max_norm_value,length_Element)) 

    with Pool(N_CORES) as pool:
        for result in pool.imap(iteration, jobs):
            outlier_score_list.append(result)
            pbar.update(1)

    df=pd.DataFrame(outlier_score_list, columns=["distances"])
    df.to_csv(file_name_output, index=False)

    pbar.close()
