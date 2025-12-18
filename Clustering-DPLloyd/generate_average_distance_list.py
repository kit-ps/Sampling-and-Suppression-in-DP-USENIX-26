import pandas as pd
import numpy as np

def norm_calculation(row1,row2):
    square_sum = 0 
    for i in range(len(row1)):
        square_sum = square_sum + (row1[i]-row2[i])**2
    return np.sqrt(square_sum)

def generate_average_distance_list(file_name_output, df, columns, normalized_range_value=1):        
    
    max_norm_value = 2*normalized_range_value*np.sqrt(len(columns))

    outlier_score_list=[]
    length_Element=df.shape[0]
    
    #For every element in the database, compute its outlier score and add it to the list
    for _, i in df.iterrows():
        total_sum=0
        for _, j in df.iterrows(): 
            dist=norm_calculation(i,j)/max_norm_value
            total_sum=total_sum + dist
        outlier_score=total_sum/(length_Element)
        outlier_score_list.append(outlier_score)
    df=pd.DataFrame(outlier_score_list, columns=["distances"])
    df.to_csv(file_name_output, index=False)
    #dataframenumpy=np.array(outlier_score_list)
    #np.savetxt(file_name_output, dataframenumpy)
