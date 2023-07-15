#!/bin/bash

#SBATCH --array=1-10
#SBATCH --partition=batch
#SBATCH -J TorchGeoNN_FULL_100%
#SBATCH -o log/%x.%3a.%A.out
#SBATCH -e log/%x.%3a.%A.err
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=6
#SBATCH -N 1
#SBATCH --time=30:00:00
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

echo "Copy folder to tmp/..."
mkdir /tmp/FULL_Dataset
echo "Folder created"
cp -a /ibex/scratch/riviers/TorchGeo/DeepLabV3/FULL_Dataset/. /tmp/FULL_Dataset
echo "... copy finished"

echo "Running python script..."
python DeepLabV3/main.py \
--data_directory=/tmp/FULL_Dataset \
--exp_directory=/ibex/scratch/riviers/TorchGeo/DeepLabV3_Output/FULL_${SLURM_ARRAY_TASK_ID}_100_Test \
--epochs 300 \
--batch_size 32 \
--train_percent 1.0 \
"$@"
echo "... script terminated"

date