import pandas as pd
import numpy as np


def generate_average_distance_list(file_name_output, df):
    counts=df.value_counts()
    
    outlier_score_list=[]
    length_Element=len(df)
    #For every element in the database, compute its outlier score and add it to the list
    for i in df: 
        sum_of_distances = length_Element - counts[i]
        outlier_score=sum_of_distances/(length_Element)
        outlier_score_list.append(outlier_score)
    df=pd.DataFrame(outlier_score_list, columns=["distances"])
    df.to_csv(file_name_output, index=False)
