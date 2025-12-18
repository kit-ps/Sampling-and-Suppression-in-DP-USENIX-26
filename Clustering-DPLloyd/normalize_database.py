import pandas as pd
import numpy as np

def normalize_database(database_name, output_file_name, columns, normalized_range_value=1):
	df=pd.read_csv(database_name)
	df_filter = df.filter(columns)
	dimension = len(columns)

	newdf = pd.DataFrame()

	for i in range(dimension):
		minimum=df[columns[i]].min()
		maximum=df[columns[i]].max()

		newdf[columns[i]] = normalized_range_value*(-1+2*(df[columns[i]]-minimum)/(maximum-minimum))

	newdf.to_csv(output_file_name, index=False)