#!/bin/bash
# launch jupyter

#SBATCH --account=glab
#SBATCH -J jupyter
#SBATCH --time=1:00:00
#SBATCH --mem=20G

# Setup Environment
source activate RL-Enduro

jupyter lab & --no-browser --ip "*" --notebook-dir /home/jiangche/RL-Enduro