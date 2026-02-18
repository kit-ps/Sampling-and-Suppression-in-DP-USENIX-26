from principal_function import *
import sys
import argparse

def parse_columns(args):
	columns = []
	ranges = []
	for column in args:
		name, lower, upper = column.split(":", 2)
		lower, upper = float(lower), float(upper)
		columns.append(name)
		ranges.append([lower, upper])
	return columns, ranges

if len(sys.argv)==1:
	##Run experiment from the paper

	repetitions = 500

	generateFileandGraph(database_name="adult_clustering.csv", columns=["age", "fnlwgt", "education-num", "capital-gain", "capital-loss", "hours-per-week"], main_folder_name="Adult_clustering", number_clusters=5, range_columns=[[0,125],[0,2227058],[1,16],[0,149999],[0,6534],[0,100]], repetitions=repetitions)
else: 
	parser = argparse.ArgumentParser()
	parser.add_argument('--database-name', required=True)
	parser.add_argument('--main-folder-name', required=True)
	parser.add_argument('--columns-range', nargs="+", required=True)
	parser.add_argument('--number-clusters',required=True, type=int)
	parser.add_argument('--repetitions', default=500, type=int)
	parser.add_argument('--list-epsilons', nargs="+", type=float, default=[0.25, 0.5, 1, 2])
	parser.add_argument('--normalized-range-value', required=False, type=int, default=1)

	args = parser.parse_args()

	columns, range_columns = parse_columns(args.columns_range)
		
	generateFileandGraph(database_name=args.database_name, columns=columns, main_folder_name=args.main_folder_name, range_columns=range_columns, number_clusters=args.number_clusters, normalized_range_value=args.normalized_range_value, list_epsilons=args.list_epsilons, repetitions=args.repetitions)
