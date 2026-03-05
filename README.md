# Experiments on Sampling and Suppression

This repository includes the complete code for the experiments and plots on the effects of sampling and outlier-score suppression on the privacy–utility tradeoff in differential privacy (DP) for the USENIX Security '26 paper

*The Adverse Effects of Omitting Records in Differential Privacy:
How Sampling and Suppression Degrade the Privacy–Utility Tradeoff*

by Àlex Miranda-Pascual, Javier Parra-Arnau, and Thorsten Strufe. 

This artifact is permanently available at https://zenodo.org/records/17977527.

## Overview 

Each folder contains an independent experiment. These are:

* `NoisyAverage` contains our experiment on the mean computation with NoisyAverage.
* `ReportNoisyMax` contains our experiment on the mode computation with report noisy max and the exponential mechanism. 
* `Clustering-DPLloyd` contains our experiment on the DPLloyd clustering algorithm.
* `Clustering-kmedian` contains our experiment on the $k$-median clustering algorithm.
* `PrivacyBound` contains the code that checks whether the empirical results we obtain match our theorized values for Theorem 5.4.
* `PrivacyBoundPlots` contains the code that generates Figures 2 and 4 of our paper (Figures 2 and 75 of the long version). 

Each folder has a respective README file that explains the details of each experiment.  

Note that the experiments are randomized. Included in each folder, we provide all statistical values in CSV files and plots that were used in the paper.

In addition, we provide the `ViewPaperPlots.html` file in the main folder, which allows the reader to easily find and open all the figures shown in the paper and its long version.

## Technical Description and Setup Instructions

For running the experiment, we used a server with an AMD EPYC 7702P 64-Core Processor running in Ubuntu 24.04. We note that approximately 1.7&#8201;GB of RAM is sufficient to run our code. All our code is written in Python 3.8.20. Some files contain parallelizations with 64 cores. 

The experiments in each folder are independent of each other but share a common setup. For our experiments, we work with Conda and its environments. We recommend following their user guide to install Conda: https://docs.conda.io/projects/conda/en/stable/user-guide/index.html. We used the version `conda 24.7.1`. After installing Conda, the necessary requirements can be set up by creating and activating the environment `SamplingAndSuppression` contained in the `environment.yml` file: 

```bash
conda env create -f environment.yml
conda activate SamplingAndSuppression
```

Running `main.py` from each folder file covers the whole experiment (further details are given in each README file): 

```bash
python main.py
```

Running every `main.py` one after the other takes around 4.5&nbsp;days.  

## Sources of the Used Databases

The databases used for each experiment are included in their respective folder. A total of three different public databases are used, and we explain here their sources.

### "Adult" database

The Adult database was derived from the US Census Bureau’s 1994–1995 Current Population Survey (CPS) can be downloaded from 

[1] B. Becker and R. Kohavi, “Adult”. UCI Machine Learning Repository, 1996. doi: 10.24432/C5XW20.

### "Census" database

The Census database was obtained on July 27, 2000, using the Data Extraction System of the US Bureau of the Census (http://www.census.gov/DES/www/welcome.html) for the Computational Aspects of Statistical Confidentiality (CASC) project, as explained in 

[2] R. Brand, J. Domingo-Ferrer, and J. M. Mateo-Sanz, 
“Reference data sets to test and compare SDC methods for protection of numerical microdata”, 
Tech. Rep. European project IST-2000-25069 CASC, Apr. 2002. [Online]. 
Available: https://research.cbs.nl/casc/CASCrefmicrodata.pdf

The database can also be downloaded from https://sdctools.github.io/sdcMicro/reference/CASCrefmicrodata.html

### "Irish" database

The Irish database is a synthetic database generated from data of the 2011 Irish Census used and described in

[3] V. Ayala-Rivera, A. O. Portillo-Dominguez, L. Murphy, and C. Thorpe, 
“COCOA: a synthetic data generator for testing anonymization techniques”, 
in Privacy in Statistical Databases, J. Domingo-Ferrer and M. Pejić-Bach, Eds., 
in Lecture Notes in Computer Science. Cham: Springer International Publishing, 2016, pp. 163–177. 
doi: 10.1007/978-3-319-45381-1_13.

We directly work with the postprocessing into numerical variables used in Rodríguez-Hoyos et al. [4], consisting of the last two-thirds of `irishcensus100m.csv` in the repository of Ayala-Rivera et al. [3] (https://github.com/ucd-pel/COCOA/). 

[4] A. Rodríguez-Hoyos, J. Estrada-Jiménez, D. Rebollo-Monedero, J. Parra-Arnau, and J. Forné, 
“Does $k$-Anonymous Microaggregation Affect Machine-Learned Macrotrends?”, 
IEEE Access, vol. 6, pp. 28258–28277, 2018, doi: 10.1109/ACCESS.2018.2834858.

## License

Copyright (C) 2025–2026 Àlex Miranda-Pascual, Javier Parra-Arnau, and Thorsten Strufe

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program. If not, see https://www.gnu.org/licenses/.
