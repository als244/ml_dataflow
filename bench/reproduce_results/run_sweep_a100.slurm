#!/bin/bash
#SBATCH --job-name=a100_training_sweep         # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=8        # cpu-cores per task (>1 if multi-threaded tasks
#SBATCH --gres=gpu:1             # number of gpus per node
#SBATCH --time=23:59:00          # total run time limit (HH:MM:SS)
#SBATCH --mail-type=all        # send email
#SBATCH --mail-user=as1669@princeton.edu
#SBATCH --output=logs/run_sweep_a100.out
#SBATCH --error=logs/run_sweep_a100.err



## SCRIPT USAGE

## sbatch --mem=<specify host mem> <this script filename> <experiment name> <output csv filepath>

python sweep_training_environments.py experiment_sweep_config.json $1 $2
