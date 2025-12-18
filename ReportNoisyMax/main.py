from principal_function import *

numberofrepeat=2000

print("Adult")

generateFileandGraph(database_name="adult_train.csv", column_name="age", main_folder_name="Adult", value_range=[0,125], numberofrepeat=numberofrepeat)
generateFileandGraph(database_name="adult_train.csv", column_name="hours-per-week", main_folder_name="Adult", value_range=[0,100], numberofrepeat=numberofrepeat)

print("Irish")

generateFileandGraph(database_name="irishn_train.csv", column_name="Age", main_folder_name="Irishn", value_range=[0,125], numberofrepeat=numberofrepeat)
generateFileandGraph(database_name="irishn_train.csv", column_name="HighestEducationCompleted", main_folder_name="Irishn", value_range=[1,10], numberofrepeat=numberofrepeat)