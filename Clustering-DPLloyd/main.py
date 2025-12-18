from principal_function import *

numberofrepeat = 500

generateFileandGraph(database_name="adult_clustering.csv", columns=["age", "fnlwgt", "education-num", "capital-gain", "capital-loss", "hours-per-week"], main_folder_name="Adult_clustering", number_clusters=5, range_columns=[[0,125],[0,2227058],[1,16],[0,149999],[0,6534],[0,100]], numberofrepeat=numberofrepeat)
