import pandas as pd
import numpy as np

def generate_average_distance_list(file_name_output, df, value_range):
    # Normalize vector
    Element_norm=(df/(value_range[1]-value_range[0]))
    
    outlier_score_list=[]
    length_Element=len(Element_norm)
    #For every element in the database, compute its outlier score and add it to the list
    for i in Element_norm: 
        total_sum=0
        for j in Element_norm: 
            dist=np.abs(i-j)
            total_sum=total_sum + dist
        outlier_score=total_sum/(length_Element)
        outlier_score_list.append(outlier_score)
    df=pd.DataFrame(outlier_score_list, columns=["distances"])
    df.to_csv(file_name_output, index=False)