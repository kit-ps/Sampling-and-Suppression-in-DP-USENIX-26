from principal_function import *

import sys
import argparse

if len(sys.argv)==1:
	##Run experiment from the paper

	repetitions=2000

	generateFileandGraph(database_name="adult_train.csv", column_name="age", main_folder_name="Adult", value_range=[0,125], repetitions=repetitions)
	generateFileandGraph(database_name="adult_train.csv", column_name="hours-per-week", main_folder_name="Adult", value_range=[0,100], repetitions=repetitions)

	generateFileandGraph(database_name="irishn_train.csv", column_name="Age", main_folder_name="Irishn", value_range=[0,125], repetitions=repetitions)
	generateFileandGraph(database_name="irishn_train.csv", column_name="HighestEducationCompleted", main_folder_name="Irishn", value_range=[1,10], repetitions=repetitions)
else:
	parser = argparse.ArgumentParser()
	parser.add_argument('--database-name', required=True)
	parser.add_argument('--column-name', required=True)
	parser.add_argument('--main-folder-name', required=True)
	parser.add_argument('--range-min', required=True, type=int)
	parser.add_argument('--range-max', required=True, type=int)
	parser.add_argument('--repetitions', default=20, type=int)
	parser.add_argument('--list-epsilons', nargs="+", type=float, default=[0.25, 0.5, 1, 2])
	parser.add_argument('--delta', required=False, type=float, default=None)

	args = parser.parse_args()
	
	generateFileandGraph(database_name=args.database_name, column_name=args.column_name, main_folder_name=args.main_folder_name, value_range=[args.range_min,args.range_max], list_epsilons=args.list_epsilons, delta=args.delta, repetitions=args.repetitions)