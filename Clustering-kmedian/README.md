# Experiment on the $k$-Median Clustering Algorithm 

**This folder contains the code to run our experiment on the $k$-median clustering algorithm.**

For the accepted paper at USENIX Security '26 of

*The Adverse Effects of Omitting Records in Differential Privacy:
How Sampling and Suppression Degrade the Privacy–Utility Tradeoff*

by Àlex Miranda-Pascual, Javier Parra-Arnau, and Thorsten Strufe.

This experiment is part of the artifact available in full at https://zenodo.org/records/17977528.

## Overview 

The code generates the CSV files and plots of the utility results used in the paper. This folder covers our clustering experiment using the $k$-median algorithm. The same script covers both the sampling (Section 3) and outlier-score suppression (Section 6) evaluations.

For a given database and column, the code runs the NoisyAverage algorithm for the specified epsilons. It generates all the CSV files and plots altogether. 

The code is written in Python 3.8.20.

## Setup

The requirements for all folders are the same, meaning that no further setup is necessary if it has already been set up once. For commodity and in case the environment has yet to be set up, the `environment.yml` file is also included in this folder and can be configured and activated by running:

```bash
conda env create -f environment.yml
conda activate SamplingAndSuppression
``` 

## <a name="how-to-run">How to Run</a>

Run the `main.py` file to obtain an instantiation of the experiment with the parameters and databases used in the paper (recall that the experiment contains randomization). 

The code can be run for other parameters or databases with the following command:

```bash
	python main.py --database-name database1.csv --data-domain-name database1_domain.csv --columns row1 row2 --main-folder-name Database1 --number-clusters 4 --list-epsilons 0.25 0.5 1 2 --repetitions 20
```

The necessary and optional parameters are the following:

* `database-name`: Path of the database to test (a CSV file with labeled rows and no indices).
* `data-domain-name`: Path of the data domain (a CSV containing every possible record in the data universe)
* `columns`: List of the columns name of the database to test (must be congruent with the data domain).
* `main-folder-name`: Name of the main folder where the CSV files and plots will be stored (in `str` format). 
* `number-clusters`: Number of clusters for the clustering algorithm.
* `list-epsilons`: List of epsilon values the experiment runs for. It must be introduced as a list of numerical values, all larger than 0, as seen in the command line. It is an optional parameter: The default is set to `0.25 0.5 1 2`. 
* `repetitions`: Number of repetitions to be computed for each value `(m,M)`. An integer must be introduced. It is an optional parameter: The default is set to `20`.

We generated a simple synthetic database `database1.csv` (and its domain file `database1_domain.csv`) by running the `generate_database.py` file. We used this database in our experiment.

Alternatively, the code can be run for other parameters or databases, by creating and running a `.py` file with the following command:

```bash
from principal_function import *

generateFileandGraph(database_name="database1.csv", data_domain_name="database1_domain.csv", columns=["row1", "row2"], main_folder_name="Database1", number_clusters=4, list_epsilons=[0.25,0.5,1,2], repetitions=20)
```

The `generateFileandGraph` function generates all the CSV files and plots for the specified database. The inputs are equivalent to the parameters above.

## Time to Run

The time to run is dependent on the size of the database and the number of iterations. Progress bars show the amount of computations left. Note that two progress bars appear per database execution: The first one covers the generation of average distances between the records, and the second the actual computation of the experiment. After the progress bars have been completed, the code takes a couple of seconds to generate the plots. 

In our case, the run on the `database1` database took slightly more than a day (26&#8201;h). The code contains a parallelization into 64 pools.

## <a name="output">Output</a>

The output CSV files and plots are all included in the directory: `[main_folder_name]`. Inside this directory, the `CSVfiles` and `Plots` folder contains, respectively, the CSV files and plots. Also included in this directory is:

* `[all names in the [columns] list]_distances.csv`: The CSV file containing the average distance of every record in the database to the others, used to generate the outlier scores.

Inside the `CSVfiles` and `Plots` folders, a subfolder with the name created from concatenating the strings in `columns` is created. The CSV files in `CSVfiles`/ `[all names in the [columns] list]` are:

* Files of the type `eps=[epsilon]_[mechanism].csv` containing the average cost of the $k$-median algorithm run multiple times (total number given by `repetitions`). A missing number appearing in the CSV file means that the mechanism could not be run for the given epsilon parameter. One file is generated for every `epsilon` in `list_epsilons`, and there are four `mechanism` variants:
	* `M`: mechanism NoisyAverage (M) without suppression run for the given epsilon.
	* `MoS`: mechanism NoisyAverage (M) with suppression (S) run for the given epsilon.
	* `M_ChangeEpsDelta`: mechanism NoisyAverage (M) without suppression run for the epsilon that ensures that M and MoS have the same privacy parameters.  
	* `MoS_ChangeEpsDelta`: mechanism NoisyAverage (M) with suppression (S) run for the epsilon that ensures that M and MoS have the same privacy parameters.
* Files of the type `eps=[epsilon]_[mechanism]_[mechanism]_Average.csv`, where the empirical mean over the numerical values of all iterations is computed for every `(m,M)`.
* Files of the type `eps=[epsilon]_[mechanism]_[mechanism]_Variance.csv`, where the empirical variance over the numerical values of all iterations is computed for every `(m,M)`.
* Files of the type `eps=[epsilon]_[mechanism]_combined_[Average/Variance].csv` containing the absolute error differences of the mean/variance of M minus those of MoS, of M minus those of MoSChangeEpsDelta, and of MChangeEpsDelta minus those of MoS for every `(m,M)`. These are used in the plot generation.

Each CSV file contains the statistics relevant for both the sampling and outlier-score suppression evaluations. 

The Plots in `Plots`/`[column_name]` are:

* Files of the type `eps=[epsilon]_difference_error_[mechanism_difference]_[statistic]_10--90.pdf`: The plots cover the outlier-score suppression evaluation. For each `epsilon` value, a plot with the utility difference is given over the different values of `(m,M)`. The variations are:
	* `mechanism_difference` is either `M_minus_MoS`, `M_minus_MoSChangeEpsDelta`, or `MChangeEpsDelta_minus_MoS`, depending on which of the three differences is plotted.
	* `statistic` is either `Average`, the average of the average costs, or `Variance`, the variance of the average cost. 
* `[all names in the [columns] list]_uniform_Poisson_sampling_[statistic].pdf`: The plots showing the effect of uniform Poisson sampling. The condition `statistic` is as before, or is the variant `Average+SD`, which adds to the average its 95% confidence interval (generated from the standard deviation).  

We note that no plot is generated if there are not enough non-empty values in the CSV file.

In addition, the code generates a separate folder `PaperPlots` that contains only the plots that are used in the paper and its long version (see next section). 

## Results and Plots for the Paper

The file `main.py` contains the experiment we ran for the $k$-median clustering, including the exact plots used in the paper. The outputs are included in the respectively named folders as mentioned before. Note that our experiment covers more cases and plots than are included in the paper. The `PaperPlots` folder contains only the plots of the evaluations shown in the paper. Running the code for other variables will also include the equivalent subset of evaluations in this folder. Currently, the `PaperPlots` folder contains all the plots used in the paper and only those plots. The `ViewPaperPlots.html` file in the main folder allows the reader to easily find and open all the figures shown in the paper and its long version. We also list them here:

* The plot covering the evaluation for sampling (Section 3). Forms part of Appendix A.3 (long version):
	* `row1_row2_Poisson_sampling_Average+SD.pdf`: Figure 14 (left) in long version.
	
* The plots covering the evaluation for outlier-score suppression (Section 6). Each figure in the paper contains four variants changing the `[epsilon]` value to  `0.25`, `0.5`, `1`, or `2`. Forms part of Appendix A.6 (long version):
	* `eps=[epsilon]_difference_error_M_minus_MoSChangeEpsDelta_Average_10--90.pdf`: Figure 43 in long version.

* The plots covering an additional case that compares the effect of the mechanism with and without outlier-score suppression, but without the privacy amplification. Values for `[epsilon]` are as before. Forms part of Appendix A.9 (long version):
	* `eps=[epsilon]_difference_error_M_minus_MoS_Average_10--90.pdf`: Figure 73 in long version.

## Overview of the Files in the Folder

We briefly describe the `.py` files in this experiment folder:

* `main.py` runs our experiment with the parameters and databases of the paper (see [How to Run](#how-to-run)).
* `principal_function.py` contains the function corresponding to a single run for the given variables (see [How to Run](#how-to-run)). 
* `generate_average_distance_list.py` contains the function used to generate the distance values between the records of the database, a necessary first-step for outlier-score suppression.
* `suppression_algorithm.py` contains the functions that run the mechanisms with and without sampling/outlier-score suppression.
* `kmedian.py` contains the $k$-median algorithm.  
* `suppression_privacy_parameters.py` contains the auxiliary functions that compute the privacy parameters of suppression and their inverses. This file is equal for all the main experiments.
* `graphic_generator.py` contains the functions that generate the plots for sampling and outlier-score suppression.
* `generate_database.py` creates a simple numerical database and its domain file.
* `paperplots.py` contains the function that generates the copies of the plots used in the paper in a separate folder (`PaperPlots`) for convenience. 


The database used in this experiment is `database1.csv` (and its domain file `database1_domain.csv`). In addition, in the experiment folder, we find these subfolders: 

* The `Database1` folder, which contains the output of the individual experiment (see [Output](#output)). 
* The `PaperPlots` folder, which contains only the plots of the previous folders that are used in the paper or its long version.