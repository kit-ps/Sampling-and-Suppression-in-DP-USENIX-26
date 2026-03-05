# Experiment on the DPLloyd Clustering Algorithm 

**This folder contains the code to run our experiment on the DPLloyd clustering algorithm.**

For the accepted paper at USENIX Security '26 of

*The Adverse Effects of Omitting Records in Differential Privacy:
How Sampling and Suppression Degrade the Privacy–Utility Tradeoff*

by Àlex Miranda-Pascual, Javier Parra-Arnau, and Thorsten Strufe.

This experiment is part of the artifact available in full at https://zenodo.org/records/17977527.

## Overview 

The code generates the CSV files and plots of the utility results used in the paper. This folder covers our clustering experiment using the DPLloyd algorithm. The same script covers both the sampling (Section 3) and outlier-score suppression (Section 6) evaluations.

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
	python main.py --database-name adult_clustering.csv --columns age:0:125 fnlwgt:0:2227058 education-num:1:16 capital-gain:0:149999 capital-loss:0:6534 hours-per-week:0:100 --main-folder-name Adult_clustering --number-clusters=5 --normalized-range-value=1 --list-epsilons 0.25 0.5 1 2 --repetitions 500
```

The necessary and optional parameters are the following:

* `database-name`: Path of the database to test (a CSV file with labeled rows and no indices).
* `columns-range`: List of the columns name of the database to test and their ranges. They must be introduced as `name:lower_bound:upper_bound` (separated with `:`), and with spaces between columns.
* `main-folder-name`: Name of the main folder where the CSV files and plots will be stored (in `str` format). 
* `number-clusters`: Number of clusters for the clustering algorithm. 
* `normalized-range-value`: Integer necessary to normalize every column. It is an optional parameter: The default is set to `1`. 
* `list-epsilons`: List of epsilon values the experiment runs for. It must be introduced as a list of numerical values, all larger than 0, as seen in the command line. It is an optional parameter: The default is set to `0.25 0.5 1 2`. 
* `repetitions`: Number of repetitions to be computed for each value `(m,M)`. An integer must be introduced. It is an optional parameter: The default is set to `500`.

Alternatively, the code can be run for other parameters or databases, by creating and running a `.py` file with the following command:

```bash
from principal_function import *

generateFileandGraph(database_name="adult_clustering.csv", columns=["age", "fnlwgt", "education-num", "capital-gain", "capital-loss", "hours-per-week"], main_folder_name="Adult_clustering", number_clusters=5, range_columns=[[0,125],[0,2227058],[1,16],[0,149999],[0,6534],[0,100]], normalized_range_value=1, list_epsilons=[0.25,0.5,1,2], repetitions=500)
```

The `generateFileandGraph` function generates all the CSV files and plots for the specified database. The inputs are equivalent to the parameters above, where `columns-range` is now separated into two, the column names (`columns`) and their ranges (`range_columns`), as seen in the previous command.  


## Time to Run

The time to run is dependent on the size of the database and the number of iterations. Progress bars show the amount of computations left. Note that two progress bars appear per database execution: The first one covers the generation of average distances between the records, and the second the actual computation of the experiment. After the progress bars have been completed, the code takes a couple of seconds to generate the plots. 

In our case, the run on the `adult_clustering` database took slightly less than 2&nbsp;days (46&#8201;h). The code contains a parallelization into 64 pools. 

## <a name="output">Output</a>

The output CSV files and plots are all included in the directory: `[main_folder_name]`. Inside this directory, the `CSVfiles` and `Plots` folder contains, respectively, the CSV files and plots. Also included in this directory are:

* `[database_name]_normalized.csv`: The CSV file containing the normalized database so each value is between −`normalized_range_value` and `normalized_range_value`, as required by DPLloyd.
* `[all names in the [columns] list]_distances.csv`: The CSV file containing the average distance of every record in the database to the others, used to generate the outlier scores.

Inside the `CSVfiles` and `Plots` folders, a subfolder with the name created from concatenating the strings in `columns` is created. The CSV files in `CSVfiles`/ `[all names in the [columns] list]` are:

* Files of the type `eps=[epsilon]_[mechanism].csv` containing the normalized intracluster variance (NICV) of DPLloyd run multiple times (total number given by `repetitions`). A missing number appearing in the CSV file means that the mechanism could not be run for the given epsilon parameter. One file is generated for every `epsilon` in `list_epsilons`, and there are four `mechanism` variants:
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
	* `statistic` is either `Average`, the average of the NICV, or `Variance`, the variance of the NICV. 
* `[all names in the [columns] list]_uniform_Poisson_sampling_[statistic].pdf`: The plots showing the effect of uniform Poisson sampling. The condition `statistic` is as before, or is the variant `Average+SD`, which adds to the average its 95% confidence interval (generated from the standard deviation).  

We note that no plot is generated if there are not enough non-empty values in the CSV file.

In addition, the code generates a separate folder `PaperPlots` that contains only the plots that are used in the paper and its long version (see next section). 

## Results and Plots for the Paper

The file `main.py` contains the experiment we ran for the DPLloyd clustering, including the exact plots used in the paper. The outputs are included in the respectively named folders as mentioned before. Note that our experiment covers more cases and plots than are included in the paper. The `PaperPlots` folder contains only the plots of the evaluations shown in the paper. Running the code for other variables will also include the equivalent subset of evaluations in this folder. Currently, the `PaperPlots` folder contains all the plots used in the paper and only those plots. The `ViewPaperPlots.html` file in the main folder allows the reader to easily find and open all the figures shown in the paper and its long version. We also list them here:

* The plot covering the evaluation for sampling (Section 3). Forms part of Appendix A.3 (long version):
	* `age_fnlwgt_education-num_capital-gain_capital-loss_hours-per-week_uniform_Poisson_sampling_Average+SD.pdf`: Figure 1 (right) and Figure 14 (right) in long version.
	
* The plots covering the evaluation for outlier-score suppression (Section 6). Each figure in the paper contains four variants changing the `[epsilon]` value to  `0.25`, `0.5`, `1`, or `2`. Forms part of Appendix A.6 (long version):
	* `eps=[epsilon]_difference_error_M_minus_MoSChangeEpsDelta_Average_10--90.pdf`: Figure 3 (bottom row) and Figure 44 in long version.

* The plots covering an additional case that compares the effect of the mechanism with and without outlier-score suppression, but without the privacy amplification. Values for `[epsilon]` are as before. Forms part of Appendix A.9 (long version):
	* `eps=[epsilon]_difference_error_M_minus_MoS_Average_10--90.pdf`: Figure 74 in long version.

## Overview of the Files in the Folder

We briefly describe the `.py` files in this experiment folder:

* `main.py` runs our experiment with the parameters and databases of the paper (see [How to Run](#how-to-run)).
* `principal_function.py` contains the function corresponding to a single run for the given variables (see [How to Run](#how-to-run)). 
* `normalize_database.py` contains the function that normalizes the database, a requirement for DPLloyd. 
* `generate_average_distance_list.py` contains the function used to generate the distance values between the records of the database, a necessary first-step for outlier-score suppression.
* `suppression_algorithm.py` contains the functions that run the mechanisms with and without sampling/outlier-score suppression.
* `DPLloyd.py` contains the DPLloyd algorithm.  
* `suppression_privacy_parameters.py` contains the auxiliary functions that compute the privacy parameters of suppression and their inverses. This file is equal for all the main experiments.
* `graphic_generator.py` contains the functions that generate the plots for sampling and outlier-score suppression.
* `paperplots.py` contains the function that generates the copies of the plots used in the paper in a separate folder (`PaperPlots`) for convenience. 

The database used in this experiment is the Adult database (`adult_clustering.csv`). In addition, in the experiment folder, we find these subfolders: 

* The `Adult_clustering` folder, which contains the output of the individual experiment (see [Output](#output)). 
* The `PaperPlots` folder, which contains only the plots of the previous folders that are used in the paper or its long version.  