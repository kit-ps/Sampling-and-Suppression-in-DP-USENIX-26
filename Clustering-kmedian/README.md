# Experiment on the $k$-Median Clustering Algorithm 

**This folder contains the code to run our experiment on the $k$-median clustering algorithm.**

For the submission at USENIX Security '26 of

*The Adverse Effects of Omitting Records in Differential Privacy:
How Sampling and Suppression Degrade the Privacy–Utility Tradeoff*

by Àlex Miranda-Pascual, Javier Parra-Arnau, and Thorsten Strufe.

## Overview 

The code generates the CSV files and plots of the utility results used in the paper. The experiment in this folder covers our clustering experiment using the $k$-median algorithm. 

For a given database and column, the code runs the NoisyAverage algorithm for the specified epsilons and delta. It generates all the CSV files and plots altogether. 

The code is written in Python 3.8.20.

## Installation

Requirements can be installed with the following:
```bash
pip install -r requirements.txt
```

## How to run

To run the code, create and run a `.py` file with the following command:

```bash
from principal_function import *

generateFileandGraph(database_name="database1.csv", data_domain_name="database1_domain.csv", columns=["row1", "row2"], main_folder_name="Database1", number_clusters=4, numberofrepeat=20)
```

The `generateFileandGraph` function generates all the CSV files and plots for the specified database. The inputs of this function are the following:

* `database_name`: Path of the tested database (a CSV file with labeled rows and no indices).
* `data_domain_name`: Path of the data domain (a CSV containing every possible record in the data universe)
* `columns`: List of the columns name of the tested database (each in `str' format).
* `main_folder_name`: Name of the main folder where the CSV files and plots will be stored (in `str' format). 
* `number_clusters`: Number of clusters for the clustering algorithm.
* `list_epsilons`: List of epsilon values the experiment runs for. It must be introduced as a list of numerical values, all larger than 0. The default parameter is set to `[0.25,0.5,1,2]`. 
* `numberofrepeat`: Number of repetitions to be computed for each value `(m,M)`. An integer must be selected.

## Output

The output CSV files and plots are all included in the directory: `main_folder_name`. Inside this directory, the `CSVfiles` and `Plots` folder contains, respectively, the CSV files and plots. Also included in this directory is:

* `[all names in the [columns] list]_distances.csv`: The CSV file containing the average distance of every record in the database to the others, used to generate the outlier scores.

Inside the `CSVfiles` and `Plots` folders, a subfolder with the name created from concatenating the strings in `columns` is created. The CSV files in `CSVfiles`/ `[all names in the [columns] list]` are:

* Files of the type `eps=[epsilon]_[mechanism].csv` containing the average cost of the $k$-median algorithm for every of the `numberofrepeat` iterations. A missing number appearing in the CSV file means that the mechanism could not be run for the given epsilon parameter. One file is generated for every `epsilon` in `list_epsilons`, and there are four `mechanism` runs:
** `M`: mechanism NoisyAverage (M) without suppression run for the given epsilon and delta.
** `MoS`: mechanism NoisyAverage (M) with suppression (S) run for the given epsilon and delta.
** `M_ChangeEpsDelta`: mechanism NoisyAverage (M) without suppression run for the epsilon and delta that ensures that M and MoS have the same privacy parameters.  
** `MoS_ChangeEpsDelta`: mechanism NoisyAverage (M) with suppression (S) run for the epsilon and delta that ensures that M and MoS have the same privacy parameters.
* Files of the type `eps=[epsilon]_[mechanism]_[mechanism]_Average.csv`, where the empirical mean over the numerical values of `numberofrepeat` iterations is computed for every `(m,M)`.
* Files of the type `eps=[epsilon]_[mechanism]_[mechanism]_Variance.csv`, where the empirical variance over the numerical values of `numberofrepeat` iterations is computed for every `(m,M)`.
* Files of the type `eps=[epsilon]_[mechanism]_combined_[Average/Variance].csv` containing the absolute error differences of M - MoS, M - MoSChangeEpsDelta and MChangeEpsDelta - MoS for every `(m,M)`. These are used in the plot creation. 

The Plots in `Plots`/`[column_name]` are:

* Files of the type `eps=[epsilon]_difference_error_[mechanism_difference]_[statistic]_10--90.pdf`: For each `epsilon` value, a plot with the utility difference is given over different values of `(m,M)`. The variations are:
** `mechanism_difference` is either `M_minus_MoS`, `M_minus_MoSChangeEpsDelta` or `MChangeEpsDelta_minus_MoS`, depending on which of the three differences is plotted.
** `statistic` is either `Average`, the average of the average costs, or `Variance`, the variance of the average cost. 
* `[all names in the [columns] list]_uniform_Poisson_sampling_[statistic].pdf`: The plots showing the effect of uniform Poisson sampling. The condition `statistic` is as before, or is the variant `Average+SD`, which adds to the average its 95% confidence interval (generated from the standard deviation).  

## Results for Paper and Time to Run

The file `main.py` contains the experiment we ran for the $k$-median clustering and the outputs are included in the respective folders. Our experiment covers more cases and plots than those included in the final work. In particular, the folder `PaperPlots` contains the copies of only those plots used in our final (long) manuscript for easy checking.

The file `generate_database.py` generates a simple synthetic database to test with the experiment. It is used to generate `database1.csv` used in our experiment.

The time to run `main.py` is dependent on the size of the database. In our case, the experiment on the `database1` database took around a day. The code contains a parallelization into 64 pools.  