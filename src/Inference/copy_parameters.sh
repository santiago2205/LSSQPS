#!/bin/bash

#SBATCH --partition=batch
#SBATCH -J Copy_Parameters
#SBATCH -o log_Parameters/%x.%3a.%A.out
#SBATCH -e log_Parameters/%x.%3a.%A.err
#SBATCH --cpus-per-task=6
#SBATCH -N 1
#SBATCH --time=4:00:00
#SBATCH --mem=20G

date

echo "Loading anaconda..."
# module purge
module avail anaconda
module load anaconda3
module load cuda/10.1.243
module list
source activate DeepLabV3
echo "...Anaconda env loaded"

echo "Running python script..."
python copy_param.py \
"$@"
echo "... script terminated"

date