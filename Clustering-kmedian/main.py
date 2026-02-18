from principal_function import *
import sys
import argparse

if len(sys.argv)==1:
	##Run experiment from the paper

	repetitions = 20

	generateFileandGraph(database_name="database1.csv", data_domain_name="database1_domain.csv", columns=["row1", "row2"], main_folder_name="Database1", number_clusters=4, repetitions=repetitions)
else: 
	parser = argparse.ArgumentParser()
	parser.add_argument('--database-name', required=True)
	parser.add_argument('--data-domain-name', required=True)
	parser.add_argument('--main-folder-name', required=True)
	parser.add_argument('--columns', nargs="+", required=True)
	parser.add_argument('--number-clusters',required=True, type=int)
	parser.add_argument('--repetitions', default=20, type=int)
	parser.add_argument('--list-epsilons', nargs="+", type=float, default=[0.25, 0.5, 1, 2])

	args = parser.parse_args()
	
	generateFileandGraph(database_name=args.database_name, data_domain_name=args.data_domain_name, columns=args.columns, main_folder_name=args.main_folder_name, number_clusters=args.number_clusters, list_epsilons=args.list_epsilons, repetitions=args.repetitions)
