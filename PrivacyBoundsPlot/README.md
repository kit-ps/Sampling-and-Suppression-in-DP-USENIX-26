# Code to Generate the Plots on the Privacy Parameters of Theorem 5.4

**This folder contains the code to generate the plots for our bound of the Theorem 5.4**

For the submission at USENIX Security '26 of

*The Adverse Effects of Omitting Records in Differential Privacy:
How Sampling and Suppression Degrade the Privacy–Utility Tradeoff*

by Àlex Miranda-Pascual, Javier Parra-Arnau, and Thorsten Strufe.

## Overview 

The code generates the plots for our paper.

The code is written in Python 3.8.20.

## Installation

Requirements can be installed with the following:
```bash
pip install -r requirements.txt
```

## How to run

The result is obtained by running  `privacyboundsplot.py`.

## Output

The plots returned are: 

* `plots_eps_suppression_[epsilon].pdf` for `epsilon=0,0.25,0.5,0.75,1,2`. It consists of the plot of epsilon^S with respect to m and M for the chosen value of `epsilon`. Used in Figure 2 of the paper. 
* `plot_simplied_areas.pdf`. It is the plot that shows the areas where the expression simplifies and the bound is tight with respect to Theorem 5.3. It is Figure 4 of the paper (or Figure 75 in the long version).

## Results for Paper and Time to Run

The plots are used in Figure 2 and 4 (or 75 in the long version).

The time to run is a couple seconds. 