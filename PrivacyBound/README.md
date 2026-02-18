# Experiment on the Correctness of Theorem 5.4

**This folder contains the code to check computationally the bound of the Theorem 5.4**

For the accepted paper at USENIX Security '26 of

*The Adverse Effects of Omitting Records in Differential Privacy:
How Sampling and Suppression Degrade the Privacy–Utility Tradeoff*

by Àlex Miranda-Pascual, Javier Parra-Arnau, and Thorsten Strufe.

This experiment is part of the artifact available in full at https://zenodo.org/records/17977528.

## Overview 

The code generates checks and computes the difference between the computer-obtained value and theoretical value for multiple epsilon, m and M values. The algorithm returns CSV files with the difference. 

The code is written in Python 3.8.20.

## Setup

The requirements for all folders are the same, meaning that no further setup is necessary if it has already been set up once. For commodity and in case the environment has yet to be set up, the `environment.yml` file is also included in this folder and can be configured and activated by running

```bash
conda env create -f environment.yml
conda activate SamplingAndSuppression
``` 

## <a name="how-to-run">How to Run</a>

The result is obtained by running `main.py`, which runs the following two Python scripts (these can also be run individually):

* `FinalMaximumFunction.py`: Checks and generates the CSV file containing the theoretical and computational result and its difference of the function explained in Remark A.11 (or Remark B.22 in long version of paper) for all epsilon between `0.01` and `1.99` (step=`0.01`), between `2` and `9.9` (step=`0.1`), and between `10` and `100` (step=`1`); and for all `(m,M)` with granularity `0.01`.
* `FinalMaximumInverse.py`: Checks and generates the CSV file containing the theoretical and computational result and its difference of the function explained in Remark A.11 (or Remark B.24 in long version of paper) for all epsilon previously listed and all `(m,M)` with granularity `0.01`.  

## Time to Run

Due to the amount of computations, the time to run for all algorithms is around 21&#8201;h: `FinalMaximumFunction.py` took around 20&#8201;h&#8201;30&#8201;min and `FinalMaximumInverse.py` took around 20&#8201;min. The file `FinalMaximumFunction.py` contains parallelizations with 64 cores. 

## <a name="output">Output</a>

`FinalMaximumFunction.py` outputs a CSV file (`output.csv`) containing the computational (`DiffEvol`) and hypothesized theoretical values (`HypValue`), and its difference (`Difference`=`DiffEvol`−`HypValue`). The script also outputs error messages in the terminal when:
* `DiffEvol` is larger than `HypValue` (up to some floating error), which would contradict our claim in the paper.
* The computational maximum is obtained in a degenerate case, which would contradict our claim in the paper.
Our script does not output any errors, and so the claim given in the paper holds computationally. The script also prints the largest and smallest values of `Difference` (without taking the absolute value). 

`FinalMaximumInverse.py` outputs a CSV file (`output.csv`) containing the computational (`DiffEvol`) and hypothesized theoretical values (`HypValue`), and its difference (`Difference`=`DiffEvol`−`HypValue`).  The script also outputs error messages in the terminal when:
* `DiffEvol` is larger than `HypValue` (up to some floating error), which would contradict our claim in the paper.
* The term `L4` is actually not superfluous, which would contradict our claim in the paper.
Our script does not output any errors, and so the claim given in the paper holds computationally. The script also prints the largest and smallest values of `Difference` (without taking the absolute value). 

We include the terminal output of an execution of `main.py` in the file `terminal.txt`. 

## Results for Paper

The results are used to verify computationally that our theorized result matches the computational maximum obtained. Since the computation shows that the results match up to an error of $2\cdot 10^{-7}$, we can confirm that our theorized result is correct up to this error, as explained in the paper.  

## Overview of the Files in the Folder

Apart from the `.py` files mentioned in [How to Run](#how-to-run) and the outputs (see [Output](#output)), the folder contains `mainfunctions.py`, which gathers the necessary functions to run the comparison for `FinalMaximumFunction.py`.