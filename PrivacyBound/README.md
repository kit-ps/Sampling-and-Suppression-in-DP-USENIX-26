# Experiment on the Correctness of Theorem 5.4

**This folder contains the code to check empirically the bound of the Theorem 5.4**

For the submission at USENIX Security '26 of

*The Adverse Effects of Omitting Records in Differential Privacy:
How Sampling and Suppression Degrade the Privacy–Utility Tradeoff*

by Àlex Miranda-Pascual, Javier Parra-Arnau, and Thorsten Strufe.

## Overview 

The code generates checks and computes the difference between the empirical value and theoretical value for multiple epsilon, m and M values. The algorithm returns CSV files with the difference. 

The code is written in Python 3.8.20.

## Installation

Requirements can be installed with the following:
```bash
pip install -r requirements.txt
```

## How to run

The result is obtained by running the specific python script:

* `FinalMaximumFunctionEps0.py`: Checks and generates the CSV file containing the theoretical and empirical result and its difference of the function explained in Remark A.11 (or Remark B.22 in long version of paper) for epsilon = `0` and all `(m,M)` with granularity `0.01`.
* `FinalMaximumFunctionRange1.py`: Checks and generates the CSV file containing the theoretical and empirical result and its difference of the function explained in Remark A.11 (or Remark B.22 in long version of paper) for all epsilon between `0.01` and `1.99` (step=`0.01`) and all `(m,M)` with granularity `0.01`.
* `FinalMaximumFunctionRange2.py`: Checks and generates the CSV file containing the theoretical and empirical result and its difference of the function explained in Remark A.11 (or Remark B.22 in long version of paper) for all epsilon between `2` and `9.9` (step=`0.1`) and all `(m,M)` with granularity `0.01`.
* `FinalMaximumFunctionRange3.py`: Checks and generates the CSV file containing the theoretical and empirical result and its difference of the function explained in Remark A.11 (or Remark B.22 in long version of paper) for all epsilon between `10` and `100` (step=`1`) and all `(m,M)` with granularity `0.01`.
* `FinalMaximumInverse.py`: Checks and generates the CSV file containing the theoretical and empirical result and its difference of the function explained in Remark A.11 (or Remark B.24 in long version of paper) for all epsilon previously listed and all `(m,M)` with granularity `0.01`.  

## Output

The first four python script each output a CSV file (`output_epsilon0`, `output_range1`, `output_range2` and `output_range3`) containing the empirical (`DiffEvol`) and hypothesized theoretical values (`HypValue`), and its difference (`Difference`=`DiffEvol`−`HypValue`). The script also outputs error messages in the terminal when:
* If the empirical maximum is obtained in a degenerate case, which we do hypothesize. 
* The difference between `HypValue` and `DiffEvol` is too large. 
* `DiffEvol` is larger than `HypValue` (up to some floating error), since we do not expect it to do either.
Our last iteration does not output many errors. It also prints the largest and smallest values of `Difference` (taking sign into consideration). 

`FinalMaximumInverse.py` outputs on the terminal if an error is reached:
* The difference between `HypValue` and `DiffEvol` is too large. 
* `DiffEvol` is larger than `HypValue` (up to some floating error), since we do not expect it to do either.
Our last iteration does not output many errors. It also prints the largest and smallest values of `Difference` (taking sign into consideration). 
* The term `L4`, which we hypothesize to be superfluous, is actually not superfluous.
Our last iteration does not output many errors. It also prints the largest and smallest values of `Difference` (taking sign into consideration).  

## Results for Paper and Time to Run

The results are used to verify empirically that our theorized result matches the empirical maximum obtained. Since the computation shows that the results match up to an error of $2\cdot 10^{-7}$, we can confirm that our theorized result is correct up to this error, as disclosed in the paper. 

Due to the amount of computations, the time to run for all algorithms is around a day: `FinalMaximumFunctionEps0.py` took around 1h 30 min, `FinalMaximumFunctionRange1.py` took around 11h, `FinalMaximumFunctionRange2.py` took around 5h, `FinalMaximumFunctionRange3.py` took around 5h 30 min, and `FinalMaximumInverse.py` took around 20 min. `FinalMaximumFunctionRange1.py`,  `FinalMaximumFunctionRange2.py`, and  `FinalMaximumFunctionRange3.py` are parallelized with 64 cores. 