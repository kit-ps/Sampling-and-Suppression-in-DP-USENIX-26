import pandas as pd
import numpy as np
import tqdm

def generate_average_distance_list(file_name_output, df):
    print("Generate average distance list (1/2):")
    length_Element=len(df)
    pbar = tqdm.tqdm(total=length_Element)
    counts=df.value_counts()
    
    outlier_score_list=[]
    #For every element in the database, compute its outlier score and add it to the list
    for i in df: 
        pbar.update(1)
        sum_of_distances = length_Element - counts[i]
        outlier_score=sum_of_distances/(length_Element)
        outlier_score_list.append(outlier_score)
    df=pd.DataFrame(outlier_score_list, columns=["distances"])
    df.to_csv(file_name_output, index=False)

    pbar.close()
