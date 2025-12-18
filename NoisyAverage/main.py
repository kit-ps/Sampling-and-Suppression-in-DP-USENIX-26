from principal_function import *

numberofrepeat = 500

print("Adult")

generateFileandGraph(database_name="adult_train.csv", column_name="age", main_folder_name="Adult", value_range=[0,125], numberofrepeat=numberofrepeat)
generateFileandGraph(database_name="adult_train.csv", column_name="hours-per-week", main_folder_name="Adult", value_range=[0,100], numberofrepeat=numberofrepeat)

print("Census")

generateFileandGraph(database_name="census.csv", column_name="FICA", main_folder_name="Census", value_range=[0,11890], numberofrepeat=numberofrepeat)
generateFileandGraph(database_name="census.csv", column_name="FEDTAX", main_folder_name="Census", value_range=[0,31889], numberofrepeat=numberofrepeat)

print("Irish")

generateFileandGraph(database_name="irishn_train.csv", column_name="Age", main_folder_name="Irishn", value_range=[0,125], numberofrepeat=numberofrepeat)
generateFileandGraph(database_name="irishn_train.csv", column_name="HighestEducationCompleted", main_folder_name="Irishn", value_range=[1,10], numberofrepeat=numberofrepeat)