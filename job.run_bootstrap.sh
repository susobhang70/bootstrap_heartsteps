#!/bin/bash -x

#SBATCH -n 1                  # Number of cores
#SBATCH -N 1                  # Ensure that all cores are on one machine
#SBATCH -p murphy_secure      # Partition to submit to
#SBATCH -t 1-00:00:00         # Runtime in D-HH:MM:SS, minimum of 10 minutes
#SBATCH --mem=4000            # Memory pool for all cores (see also --mem-per-cpu) MBs
#SBATCH -o joblogs/%A_%a.out  # File to which STDOUT will be written, %j inserts jobid
#SBATCH -e joblogs/%A_%a.err  # File to which STDERR will be written, %j inserts jobid


set -x

date
cdir=$(pwd)

source /n/home12/susobhan/.bashrc
conda activate /n/home12/susobhan/.conda/envs/heartsteps

module load R/4.0.2-fasrc01

python run_boostrap.py ${SLURM_ARRAY_TASK_ID}

# To run first 4 experiments, use: sbatch --array=0-3 job.run_boostrap.sh