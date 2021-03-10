#!/bin/bash -l
#SBATCH -n 150            # Total number of processors to request (16 cores per node)
#SBATCH -p high           
#SBATCH -t 24:00:00      # Run time (hh:mm:ss) - 24 hours
#SBATCH --mail-user=jdherman@ucdavis.edu
#SBATCH --mail-type=ALL

export PATH=~/miniconda3/bin:$PATH

mpirun python sensitivity-analysis.py

# for some reason, using 200 processors causes a floating point exception
# this is a cluster problem, nothing to do with the model I think