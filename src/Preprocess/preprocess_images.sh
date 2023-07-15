#!/bin/bash

#SBATCH --partition=batch
#SBATCH -J TorchGeo_Dataset
#SBATCH -o log/%x.%3a.%A.out
#SBATCH -e log/%x.%3a.%A.err
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=6
#SBATCH -N 1
#SBATCH --time=20:00:00
#SBATCH --mem=10G

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

python preprocess_torchgeo.py \
--path_dataset=/ibex/scratch/riviers/TorchGeo/ \
--output_dir=DAL-HERS/output/ \
--nC 300 \
--nP 50 \
"$@"

echo "... script terminated"

date
