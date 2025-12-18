# Experiment on the Mode Computation

**This folder contains the code to run our experiment on the mode computation.**

For the submission at USENIX Security '26 of

*The Adverse Effects of Omitting Records in Differential Privacy:
How Sampling and Suppression Degrade the Privacy–Utility Tradeoff*

by Àlex Miranda-Pascual, Javier Parra-Arnau, and Thorsten Strufe.

## Overview 

The code generates the CSV files and plots of the utility results used in the paper. The experiment in this folder covers our mode computation over numerical bases for the report noisy max algorithm with Laplace, exponential and Gaussian noise, and the exponential mechanism. 

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

generateFileandGraph(database_name="adult_train.csv", column_name="age", main_folder_name="Adult", value_range=[0,125], list_epsilons=[0.25,0.5,1,2], delta=None, numberofrepeat=2000)
```

The `generateFileandGraph` function generates all the CSV files and plots for the specified database. The inputs of this function are the following:

* `database_name`: Path of the tested database (a CSV file with labeled rows and no indices).
* `column_name`: Name of the column of the tested database (in `str' format).
* `main_folder_name`: Name of the main folder where the CSV files and plots will be stored (in `str' format). 
* `value_range`: Range of values for the sensitivity of the Laplace and Gaussian mechanism. It must be introduced as a list of the lower and upper numerical values, i.e., `[0,125]`.
* `list_epsilons`: List of epsilon values the experiment runs for. It must be introduced as a list of numerical values, all larger than 0. The default parameter is set to `[0.25,0.5,1,2]`. 
* `delta`: Delta value for the experiment. A value between 0 and 1 (non-inclusive) or `None` must be selected. The algorithm chooses `delta` as 1/(size_of_database)^2 if `delta==None`. The default parameter is set to `None`.
* `numberofrepeat`: Number of repetitions to be computed for each value `(m,M)`. An integer must be selected.

## Output

The output CSV files and plots are all included in the directory: `main_folder_name`. Inside this directory, the `CSVfiles` and `Plots` folder contains, respectively, the CSV files and plots. Also included in this directory is:

* `[column_name]distance.csv`: The CSV file containing the average distance of every record in the database to the others, used to generate the outlier scores.

Inside the `CSVfiles` and `Plots` folders, a subfolder with the name `column_name` is created. The CSV files in `CSVfiles`/ `[column_name]` are:

* Files of the type `[column_name]_eps=[epsilon]_delta=[delta]_[mechanism].csv` containing whether the mode was correctly outputted by the DP mechanisms (RNM with Laplace, Gaussian, and exponential noise, and exponential mechanism) for every of the `numberofrepeat` iterations. A missing number appearing in the CSV file means that the mechanism could not be run for the given epsilon and delta parameters (e.g., epsilon is larger than 1 for Gaussian). One file is generated for every `epsilon` in `list_epsilons`, and there are four `mechanism` runs:
** `M`: mechanism NoisyAverage (M) without suppression run for the given epsilon and delta.
** `MoS`: mechanism NoisyAverage (M) with suppression (S) run for the given epsilon and delta.
** `M_ChangeEpsDelta`: mechanism NoisyAverage (M) without suppression run for the epsilon and delta that ensures that M and MoS have the same privacy parameters.  
** `MoS_ChangeEpsDelta`: mechanism NoisyAverage (M) with suppression (S) run for the epsilon and delta that ensures that M and MoS have the same privacy parameters.
* Files of the type `[column_name]_eps=[epsilon]_delta=[delta]_[mechanism]_Emp_Prob.csv`, where the empirical probability of correctly returning the mode over the `numberofrepeat` iterations is computed for every `(m,M)`.
* Files of the type `[column_name]_eps=[epsilon]_delta=[delta]_combined_Emp_Prob.csv` containing the differences in empirical probability of incorrectly returning the mode of M - MoS, M - MoSChangeEpsDelta and MChangeEpsDelta - MoS for every `(m,M)`. These are used in the plot creation. 

The Plots in `Plots`/`[column_name]` are:

* Files of the type `[column_name]_eps=[epsilon]_delta=[delta]_difference_[noise/mechanism]_error_[mechanism_difference]_[range].pdf`: For each `epsilon` and the `delta` value, a plot with the utility difference is given over different values of `(m,M)`. The variations are:
** `noise/mechanism` is either `laplace`, `gaussian`, `exponential` or `exponential_mechanism`, depending on the noise added or mechanism variation. 
** `mechanism_difference` is either `M_minus_MoS`, `M_minus_MoSChangeEpsDelta` or `MChangeEpsDelta_minus_MoS`, depending on which of the three differences is plotted.
** `range` is either `10--90`, plotting the values of `(m,M)` between `0.1` and `0.9` (showing suppression between 10 and 90 percent of database), or `1--9`, plotting the values of `(m,M)` between `0.01` and `0.09` (showing suppression between 1 and 9 percent of database).
* `[column_name]_uniform_Poisson_sampling_[noise/mechanism]_[type].pdf`: The plots showing the effect of uniform Poisson sampling. The condition `noise/mechanism` is as before. The condition `[type]` is either `EmpProb` with the empirical probability values, or `EmpProb+SD` with the empirical probability and its Wilson confidence intervals.  

We note that no plot is generated if there are not enough non-empty values in the CSV file.

## Results for Paper and Time to Run

The file `main.py` contains the four experiments we ran for the mode computation and the outputs are included in the respective folders. Our experiment covers more cases and plots than those included in the final work. In particular, the folder `PaperPlots` contains the copies of only those plots used in our final (long) manuscript for easy checking. 

In addition, in the folder `PaperPlots` we remove the plots that are not seen in our final work.  

The time to run is dependent on the size of the database. In our case, each experiment on the `adult_train` database took around 1h 15 min and each experiment on the `irishn_train` took around 2h 30 min. The code contains a parallelization into 64 pools.  