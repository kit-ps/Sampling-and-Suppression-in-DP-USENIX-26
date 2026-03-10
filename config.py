import os

# Number of CPU cores to use in parallelization. Use all available cores by default.
# Our code and runtimes were done with N_CORES = 64
N_CORES = os.cpu_count()