# Experiment on the Mode Computation

**This folder contains the code to run our experiment on the mode computation.**

For the accepted paper at USENIX Security '26 of

*The Adverse Effects of Omitting Records in Differential Privacy:
How Sampling and Suppression Degrade the Privacy–Utility Tradeoff*

by Àlex Miranda-Pascual, Javier Parra-Arnau, and Thorsten Strufe.

This experiment is part of the artifact available in full at https://zenodo.org/records/17977527.

## Overview 

The code generates the CSV files and plots of the utility results used in the paper. The experiment in this folder covers our mode computation over numerical bases for the report noisy max algorithm with Laplace, exponential, and Gaussian noise, and the exponential mechanism. The same script covers both the sampling (Section 3) and outlier-score suppression (Section 6) evaluations.  

For a given database and column, the code runs the NoisyAverage algorithm for the specified epsilons and delta. It generates all the CSV files and plots altogether. 

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
	python main.py --database-name adult_train.csv --column-name age --main-folder-name Adult --range-min 0 --range-max 125 --list-epsilons 0.25 0.5 1 2 --repetitions 2000
```

The necessary and optional parameters are the following:

* `database-name`: Path of the database to test (a CSV file with labeled rows and no indices).
* `column-name`: Name of the column of the database to test.
* `main-folder-name`: Name of the main folder where the CSV files and plots will be stored. 
* `range-min` and `range-max`: Lower and upper boundaries of the range of values used to compute the sensitivity of the Laplace and Gaussian mechanisms, i.e., the range is equivalent to `[range-min,range-max]`.
* `list-epsilons`: List of epsilon values the experiment runs for. It must be introduced as a list of numerical values, all larger than 0, as seen in the command line. It is an optional parameter: The default is set to `0.25 0.5 1 2`. 
* `delta`: Delta value for the experiment. A value between `0` and `1` (non-inclusive) or `None` must be introduced. The algorithm chooses `delta` as 1/(size_of_database)^2 if `None` is selected. It is an optional parameter: The default is set to `None`.
* `repetitions`: Number of repetitions to be computed for each value `(m,M)`. An integer must be introduced. It is an optional parameter: The default is set to `2000`.

Alternatively, the code can be run for other parameters or databases, by creating and running a `.py` file with the following command:

```bash
from principal_function import *

generateFileandGraph(database_name="adult_train.csv", column_name="age", main_folder_name="Adult", value_range=[0,125], list_epsilons=[0.25,0.5,1,2], delta=None, repetitions=2000)
```

The `generateFileandGraph` function generates all the CSV files and plots for the specified database. The inputs are equivalent to the parameters above, where `value_range` is equivalent to `[range-min,range-max]`.

## Time to Run

The time to run is dependent on the size of the database and the number of iterations. Progress bars show the amount of computations left. Note that two progress bars appear per database execution: The first one covers the generation of average distances between the records, and the second the actual computation of the experiment. After the progress bars have been completed, the code takes a couple of seconds to generate the plots.

 In our case, each run on the `adult_train` database took around 1&#8201;h&#8201;45&#8201;min, and each run on the `irishn_train` took around 3&#8201;h&#8201;30 min. Thus, running the `main.py` files took around 10&#8201;h&#8201;30&#8201;min. The code contains a parallelization into 64 pools.

## <a name="output">Output</a>

The output CSV files and plots are all included in the directory: `[main_folder_name]`. Inside this directory, the `CSVfiles` and `Plots` folder contains, respectively, the CSV files and plots. Also included in this directory is:

* `[column_name]distance.csv`: The CSV file containing the average distance of every record in the database to the others, used to generate the outlier scores.

Inside the `CSVfiles` and `Plots` folders, a subfolder with the name `column_name` is created. The CSV files in `CSVfiles`/ `[column_name]` are:

* Files of the type `[column_name]_eps=[epsilon]_delta=[delta]_[mechanism].csv` containing whether the mode was correctly outputted by the DP mechanisms (RNM with Laplace, Gaussian, and exponential noise, and exponential mechanism) run multiple times (total number given by `repetitions`).. A missing number appearing in the CSV file means that the mechanism could not be run for the given epsilon and delta parameters (see Section 6.1). One file is generated for every `epsilon` in `list_epsilons`, and there are four `mechanism` variants:
	* `M`: mechanism NoisyAverage (M) without suppression run for the given epsilon and delta.
	* `MoS`: mechanism NoisyAverage (M) with suppression (S) run for the given epsilon and delta.
	* `M_ChangeEpsDelta`: mechanism NoisyAverage (M) without suppression run for the epsilon and delta that ensures that M and MoS have the same privacy parameters.  
	* `MoS_ChangeEpsDelta`: mechanism NoisyAverage (M) with suppression (S) run for the epsilon and delta that ensures that M and MoS have the same privacy parameters.
* Files of the type `[column_name]_eps=[epsilon]_delta=[delta]_[mechanism]_Emp_Prob.csv`, where the empirical probability of correctly returning the mode over all iterations is computed for every `(m,M)`.
* Files of the type `[column_name]_eps=[epsilon]_delta=[delta]_combined_Emp_Prob.csv` containing the differences in empirical probability of incorrectly returning the mode of M minus those of MoS, of M minus those of MoSChangeEpsDelta, and of MChangeEpsDelta minus those of MoS for every `(m,M)`. These are used in the plot generation.

Each CSV file contains the statistics relevant for both the sampling and outlier-score suppression evaluations, and for each of the four mechanisms tested. 

The Plots in `Plots`/`[column_name]` are:

* Files of the type `[column_name]_eps=[epsilon]_delta=[delta]_difference_[noise/mechanism]_error_[mechanism_difference]_[range].pdf`: The plots cover the outlier-score suppression evaluation. For each `epsilon` and the `delta` value, a plot with the utility difference is given over the different values of `(m,M)`. The variations are:
	* `noise/mechanism` is either `laplace`, `gaussian`, `exponential`, or `exponential_mechanism`, depending on the noise added or mechanism variation. 
	* `mechanism_difference` is either `M_minus_MoS`, `M_minus_MoSChangeEpsDelta`, or `MChangeEpsDelta_minus_MoS`, depending on which of the three differences is plotted.
	* `range` is either `10--90`, plotting the values of `(m,M)` between `0.1` and `0.9` (showing suppression between 10 and 90 percent of the database), or `1--9`, plotting the values of `(m,M)` between `0.01` and `0.09` (showing suppression between 1 and 9 percent of the database).
* `[column_name]_uniform_Poisson_sampling_[noise/mechanism]_[type].pdf`: The plots show the effect of uniform Poisson sampling. The condition `noise/mechanism` is as before. The condition `[type]` is either `EmpProb` with the empirical probability values, or `EmpProb+SD` with the empirical probability and its Wilson 95% confidence intervals.  

We note that no plot is generated if there are not enough non-empty values in the CSV file.

In addition, the code generates a separate folder `PaperPlots` that contains only the plots that are used in the paper and its long version (see next section). 

## Results and Plots for the Paper

The file `main.py` contains the experiment for the four database columns we tested for the mode computation, including the exact plots used in the paper. The outputs are included in the respectively named folders as mentioned before. Note that our experiment covers more cases and plots than are included in the paper. The `PaperPlots` folder contains only the plots of the evaluations shown in the paper. Running the code for other variables will also include the equivalent subset of evaluations in this folder. Currently, the `PaperPlots` folder contains all the plots used in the paper and only those plots. The `ViewPaperPlots.html` file in the main folder allows the reader to easily find and open all the figures shown in the paper and its long version. We also list them here:

### Plots in the Main Body

The plots used in the main paper are the following:
* `age_uniform_Poisson_sampling_laplace_EmpProb+Sd.pdf`: Figure 1 (center) and Figure 10 (top left) in long version.
* `age_eps=[epsilon]_delta=[delta]_difference_laplace_error_M_minus_MoSChangeEpsDelta_10--90.pdf`: Figure 3 (middle row) and Figure 27 in long version. 

### Plots in the Long Version

* The plots covering the evaluation for sampling (Section 3). These plots comprise Appendix A.2 (long version):
	* Adult database: 
		* `age_uniform_Poisson_sampling_laplace_EmpProb+Sd.pdf`: Figure 1 (center) and Figure 10 (top left) in long version.
		* `age_uniform_Poisson_sampling_gaussian_EmpProb+Sd.pdf`: Figure 10 (top right) in long version.
		* `age_uniform_Poisson_sampling_exponential_EmpProb+Sd.pdf`: Figure 10 (bottom left) in long version.
		* `age_uniform_Poisson_sampling_exponential_mechanism_EmpProb+Sd.pdf`: Figure 10 (bottom right) in long version.
		* `hours-per-week_uniform_Poisson_sampling_laplace_EmpProb+Sd.pdf`: Figure 11 (top left) in long version.
		* `hours-per-week_uniform_Poisson_sampling_gaussian_EmpProb+Sd.pdf`: Figure 11 (top right) in long version.
		* `hours-per-week_uniform_Poisson_sampling_exponential_EmpProb+Sd.pdf`: Figure 11 (bottom left) in long version.
		* `hours-per-week_uniform_Poisson_sampling_exponential_mechanism_EmpProb+Sd.pdf`: Figure 11 (bottom right) in long version.
	* Irish database: 
		* `Age_uniform_Poisson_sampling_laplace_EmpProb+Sd.pdf`: Figure 12 (top left) in long version.
		* `Age_uniform_Poisson_sampling_gaussian_EmpProb+Sd.pdf`: Figure 12 (top right) in long version.
		* `Age_uniform_Poisson_sampling_exponential_EmpProb+Sd.pdf`: Figure 12 (bottom left) in long version.
		* `Age_uniform_Poisson_sampling_exponential_mechanism_EmpProb+Sd.pdf`: Figure 12 (bottom right) in long version.
		* `HighestEducationCompleted_uniform_Poisson_sampling_laplace_EmpProb+Sd.pdf`: Figure 13 (top left) in long version.
		* `HighestEducationCompleted_Poisson_sampling_gaussian_EmpProb+Sd.pdf`: Figure 13 (top right) in long version.
		* `HighestEducationCompleted_Poisson_sampling_exponential_EmpProb+Sd.pdf`: Figure 13 (bottom left) in long version.
		* `HighestEducationCompleted_uniform_Poisson_sampling_exponential_mechanism_EmpProb+Sd.pdf`: Figure 13 (bottom right) in long version.	
* The plots covering the evaluation for outlier-score suppression (Section 6). Each figure in the paper contains four variants changing the `[epsilon]` value to  `0.25`, `0.5`, `1`, or `2`, and the `[delta]` value corresponds to the default one obtained by selecting `delta=None`. These plots comprise Appendix A.5 (long version):
	* Adult database
		* `age_eps=[epsilon]_delta=[delta]_difference_laplace_error_M_minus_MoSChangeEpsDelta_10--90.pdf`: Figure 3 (middle row) and Figure 27 in long version. 
		* `age_eps=[epsilon]_delta=[delta]_difference_gaussian_error_M_minus_MoSChangeEpsDelta_10--90.pdf`: Figure 28 in long version.
		* `age_eps=[epsilon]_delta=[delta]_difference_exponential_error_M_minus_MoSChangeEpsDelta_10--90.pdf`: Figure 29 in long version. 
		* `age_eps=[epsilon]_delta=[delta]_difference_exponential_mechanism_error_M_minus_MoSChangeEpsDelta_10--90.pdf`: Figure 30 in long version.
		* `hours-per-week_eps=[epsilon]_delta=[delta]_difference_laplace_error_M_minus_MoSChangeEpsDelta_10--90.pdf`: Figure 31 in long version. 
		* `hours-per-week_eps=[epsilon]_delta=[delta]_difference_gaussian_error_M_minus_MoSChangeEpsDelta_10--90.pdf`: Figure 32 in long version.
		* `hours-per-week_eps=[epsilon]_delta=[delta]_difference_exponential_error_M_minus_MoSChangeEpsDelta_10--90.pdf`: Figure 33 in long version. 
		* `hours-per-week_eps=[epsilon]_delta=[delta]_difference_exponential_mechanism_error_M_minus_MoSChangeEpsDelta_10--90.pdf`: Figure 34 in long version.
	* Irish database
		* `Age_eps=[epsilon]_delta=[delta]_difference_laplace_error_M_minus_MoSChangeEpsDelta_10--90.pdf`: Figure 35 in long version. 
		* `Age_eps=[epsilon]_delta=[delta]_difference_gaussian_error_M_minus_MoSChangeEpsDelta_10--90.pdf`: Figure 36 in long version.
		* `Age_eps=[epsilon]_delta=[delta]_difference_exponential_error_M_minus_MoSChangeEpsDelta_10--90.pdf`: Figure 37 in long version. 
		* `Age_eps=[epsilon]_delta=[delta]_difference_exponential_mechanism_error_M_minus_MoSChangeEpsDelta_10--90.pdf`: Figure 38 in long version.
		* `HighestEducationCompleted_eps=[epsilon]_delta=[delta]_difference_laplace_error_M_minus_MoSChangeEpsDelta_10--90.pdf`: Figure 39 in long version. 
		* `HighestEducationCompleted_eps=[epsilon]_delta=[delta]_difference_gaussian_error_M_minus_MoSChangeEpsDelta_10--90.pdf`: Figure 40 in long version.
		* `HighestEducationCompleted_eps=[epsilon]_delta=[delta]_difference_exponential_error_M_minus_MoSChangeEpsDelta_10--90.pdf`: Figure 41 in long version. 
		* `HighestEducationCompleted_eps=[epsilon]_delta=[delta]_difference_exponential_mechanism_error_M_minus_MoSChangeEpsDelta_10--90.pdf`: Figure 42 in long version.
* The plots covering an additional case that compares the effect of the mechanisms with and without outlier-score suppression, but without the privacy amplification. Values for `[epsilon]` and `[delta]` are as before. These plots comprise Appendix A.8 (long version).
	* Adult database
		* `age_eps=[epsilon]_delta=[delta]_difference_laplace_error_M_minus_MoS_MPE_10--90.pdf`: Figure 57 in long version. 
		* `age_eps=[epsilon]_delta=[delta]_difference_gaussian_error_M_minus_MoS_MPE_10--90.pdf`: Figure 58 in long version.
		* `age_eps=[epsilon]_delta=[delta]_difference_exponential_error_M_minus_MoS_MPE_10--90.pdf`: Figure 59 in long version. 
		* `age_eps=[epsilon]_delta=[delta]_difference_exponential_mechanism_error_M_minus_MoS_MPE_10--90.pdf`: Figure 60 in long version.
		* `hours-per-week_eps=[epsilon]_delta=[delta]_difference_laplace_error_M_minus_MoS_MPE_10--90.pdf`: Figure 61 in long version. 
		* `hours-per-week_eps=[epsilon]_delta=[delta]_difference_gaussian_error_M_minus_MoS_MPE_10--90.pdf`: Figure 62 in long version.
		* `hours-per-week_eps=[epsilon]_delta=[delta]_difference_exponential_error_M_minus_MoS_MPE_10--90.pdf`: Figure 63 in long version. 
		* `hours-per-week_eps=[epsilon]_delta=[delta]_difference_exponential_mechanism_error_M_minus_MoS_MPE_10--90.pdf`: Figure 64 in long version.
	* Irish database
		* `Age_eps=[epsilon]_delta=[delta]_difference_laplace_error_M_minus_MoS_MPE_10--90.pdf`: Figure 65 in long version. 
		* `Age_eps=[epsilon]_delta=[delta]_difference_gaussian_error_M_minus_MoS_MPE_10--90.pdf`: Figure 66 in long version.
		* `Age_eps=[epsilon]_delta=[delta]_difference_exponential_error_M_minus_MoS_MPE_10--90.pdf`: Figure 67 in long version. 
		* `Age_eps=[epsilon]_delta=[delta]_difference_exponential_mechanism_error_M_minus_MoS_MPE_10--90.pdf`: Figure 68 in long version.
		* `HighestEducationCompleted_eps=[epsilon]_delta=[delta]_difference_laplace_error_M_minus_MoS_MPE_10--90.pdf`: Figure 69 in long version. 
		* `HighestEducationCompleted_eps=[epsilon]_delta=[delta]_difference_gaussian_error_M_minus_MoS_MPE_10--90.pdf`: Figure 70 in long version.
		* `HighestEducationCompleted_eps=[epsilon]_delta=[delta]_difference_exponential_error_M_minus_MoS_MPE_10--90.pdf`: Figure 71 in long version. 
		* `HighestEducationCompleted_eps=[epsilon]_delta=[delta]_difference_exponential_mechanism_error_M_minus_MoS_MPE_10--90.pdf`: Figure 72 in long version.

> [!NOTE]
> The plots of the `hours-per-week` in the Adult database and `HighestEducationCompleted` in the Irish database are not incorrect. Since the mode of each column represents approximately half of the records, both M and MoS exhibit a perfect empirical probability of success. Thus, the plots show an error of 0. For completeness, we kept these plots. 

## Overview of the Files in the Folder

We briefly describe the `.py` files in this experiment folder:

* `main.py` runs our experiment with the parameters and databases of the paper (see [How to Run](#how-to-run)).
* `principal_function.py` contains the function corresponding to a single run for the given variables (see [How to Run](#how-to-run)). 
* `generate_average_distance_list.py` contains the function used to generate the distance values between the records of the database, a necessary first-step for outlier-score suppression.
* `suppression_algorithm.py` contains the functions that run the mechanisms with and without sampling/outlier-score suppression. 
* `suppression_privacy_parameters.py` contains the auxiliary functions that compute the privacy parameters of suppression and their inverses. This file is equal for all the main experiments.
* `graphic_generator.py` contains the functions that generate the plots for sampling and outlier-score suppression.
* `paperplots.py` contains the function that generates the copies of the plots used in the paper in a separate folder (`PaperPlots`) for convenience. 

The databases used in this experiment are the Adult (`adult_train.csv`) and Irish (`irishn_train.csv`) databases. In addition, in the experiment folder, we find these subfolders: 

* The `Adult` and `Irishn` folders, which contain the output of the individual experiment (see [Output](#output)). 
* The `PaperPlots` folder, which contains only the plots of the previous folders that are used in the paper or its long version.  