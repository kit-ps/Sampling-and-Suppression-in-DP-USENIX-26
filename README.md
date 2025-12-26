# Experiments on Sampling and Suppression

This repository includes the complete code for the experiments and plots on the effects of sampling and outlier-score suppression on the privacy–utility tradeoff in differential privacy (DP) for the USENIX Security '26 submission

*The Adverse Effects of Omitting Records in Differential Privacy:
How Sampling and Suppression Degrade the Privacy–Utility Tradeoff*

by Àlex Miranda-Pascual, Javier Parra-Arnau, and Thorsten Strufe. 

## Overview 

Each folder contains an independent experiment. These are:

* `NoisyAverage` contains our experiment on the mean computation with NoisyAverage.
* `ReportNoisyMax` contains our experiment on the mode computation with report noisy max and the exponential mechanism. 
* `Clustering-DPLloyd` contains our experiment on the DPLloyd clustering algorithm.
* `Clustering-kmedian` contains our experiment on the $k$-median clustering algorithm.
* `PrivacyBound` contains the code that checks whether the empirical result we obtain match our theorized values for Theorem 5.4.
* `PrivacyBoundPlots` contains the code that generates Figures 2 and 4 of our paper. 

Each folder has a respective README file that explains the details of each experiment. 

We also upload the resulting CSV files and plots used in the paper. 

All code is written in Python 3.8.20.

## Databases source

The databases used for each experiment are included in their respective folder. A total of three different public databases are used and we explain here their sources.

### "Adult" database

The Adult database was derived from the US Census Bureau’s 1994–1995 Current Population Survey (CPS) can be downloaded from 

[1] B. Becker and R. Kohavi, “Adult”. UCI Machine Learning Repository, 1996. doi: 10.24432/C5XW20.

### "Census" database

The Census database was obtained on July 27, 2000 using the Data Extraction System of the US Bureau of the Census (http://www.census.gov/DES/www/welcome.html) for the Computational Aspects of Statistical Confidentiality (CASC) project, as explained in 

[2] R. Brand, J. Domingo-Ferrer, and J. M. Mateo-Sanz, 
“Reference data sets to test and compare SDC methods for protection of numerical microdata”, 
Tech. Rep. European project IST-2000-25069 CASC, Apr. 2002. [Online]. 
Available: https://research.cbs.nl/casc/CASCrefmicrodata.pdf

The database can also be downloaded from https://sdctools.github.io/sdcMicro/reference/CASCrefmicrodata.html

### "Irish" database

The Irish database is a synthetic database generated from the data from the 2011 Irish Census as explained in

[3] V. Ayala-Rivera, A. O. Portillo-Dominguez, L. Murphy, and C. Thorpe, 
“COCOA: a synthetic data generator for testing anonymization techniques”, 
in Privacy in Statistical Databases, J. Domingo-Ferrer and M. Pejić-Bach, Eds., 
in Lecture Notes in Computer Science. Cham: Springer International Publishing, 2016, pp. 163–177. 
doi: 10.1007/978-3-319-45381-1_13.

We use directly postprocessing into numerical variables used in Rodríguez-Hoyos et al. [4] consisting of the last two thirds of `irishcensus100m.csv` in the repository of Ayala-Rivera et al. [3] (https://github.com/ucd-pel/COCOA/). 

[4] A. Rodríguez-Hoyos, J. Estrada-Jiménez, D. Rebollo-Monedero, J. Parra-Arnau, and J. Forné, 
“Does $k$-Anonymous Microaggregation Affect Machine-Learned Macrotrends?”, 
IEEE Access, vol. 6, pp. 28258–28277, 2018, doi: 10.1109/ACCESS.2018.2834858.

## License

Copyright (C) 2025 Àlex Miranda-Pascual, Javier Parra-Arnau, and Thorsten Strufe

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
