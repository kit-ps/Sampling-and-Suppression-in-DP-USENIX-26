# Code to Generate the Plots on the Privacy Parameters of Theorem 5.4

**This folder contains the code to generate the plots for our bound of the Theorem 5.4**

For the accepted paper at USENIX Security '26 of

*The Adverse Effects of Omitting Records in Differential Privacy:
How Sampling and Suppression Degrade the Privacy–Utility Tradeoff*

by Àlex Miranda-Pascual, Javier Parra-Arnau, and Thorsten Strufe.

This experiment is part of the artifact available in full at https://zenodo.org/records/17977528.

## Overview 

The code generates the plots in Figure 2 and 4 (or 75 in the long version) for our paper.

The code is written in Python 3.8.20.

## Setup

The requirements for all folders are the same, meaning that no further setup is necessary if it has already been set up once. For commodity and in case the environment has yet to be set up, the `environment.yml` file is also included in this folder and can be configured by running

```bash
conda env create -f environment.yml
```

## How to Run

The plots are obtained by running `main.py`.

## Time to Run

The time to run is a couple seconds. 

## <a name="output">Output</a>

The plots returned are: 

* `plots_eps_suppression_[epsilon].pdf` for `epsilon=0,0.25,0.5,0.75,1,2`. They consist of the plots of epsilon^S with respect to m and M for the chosen value of `epsilon`. They are used in Figure 2 of the paper. 
* `plot_simplied_areas.pdf`. It is the plot that shows the areas where the expression simplifies and the bound is tight with respect to Theorem 5.4. It is Figure 4 of the paper (or Figure 75 in the long version).

## Results for Paper

The plots are used in Figures 2 and 4 (or Figures 2 and 75 in the long version).

## Overview of the Files in the Folder

The folder contains the `main.py` file and its output (see [Output](#output)).