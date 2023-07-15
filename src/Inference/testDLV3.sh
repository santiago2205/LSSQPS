#!/bin/bash

#SBATCH --array=1-10
#SBATCH --partition=batch
#SBATCH -J TorchGeo_50p_10-100%%
#SBATCH -o logTestArray/%x.%3a.%A.out
#SBATCH -e logTestArray/%x.%3a.%A.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH -N 1
#SBATCH --time=10:00:00
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
echo "--------100%------------"
python DeepLabV3/inference.py \
--data_directory=/ibex/scratch/riviers/TorchGeo/DeepLabV3/50p_Dataset \
--exp_directory=/ibex/scratch/riviers/TorchGeo/DeepLabV3_Output/50p_${SLURM_ARRAY_TASK_ID}_100_Test \
"$@"
echo "--------90%------------"
python DeepLabV3/inference.py \
--data_directory=/ibex/scratch/riviers/TorchGeo/DeepLabV3/50p_Dataset \
--exp_directory=/ibex/scratch/riviers/TorchGeo/DeepLabV3_Output/50p_${SLURM_ARRAY_TASK_ID}_090_Test \
"$@"
echo "--------80%------------"
python DeepLabV3/inference.py \
--data_directory=/ibex/scratch/riviers/TorchGeo/DeepLabV3/50p_Dataset \
--exp_directory=/ibex/scratch/riviers/TorchGeo/DeepLabV3_Output/50p_${SLURM_ARRAY_TASK_ID}_080_Test \
"$@"
echo "--------70%------------"
python DeepLabV3/inference.py \
--data_directory=/ibex/scratch/riviers/TorchGeo/DeepLabV3/50p_Dataset \
--exp_directory=/ibex/scratch/riviers/TorchGeo/DeepLabV3_Output/50p_${SLURM_ARRAY_TASK_ID}_070_Test \
"$@"
echo "--------60%------------"
python DeepLabV3/inference.py \
--data_directory=/ibex/scratch/riviers/TorchGeo/DeepLabV3/50p_Dataset \
--exp_directory=/ibex/scratch/riviers/TorchGeo/DeepLabV3_Output/50p_${SLURM_ARRAY_TASK_ID}_060_Test \
"$@"
echo "--------50%------------"
python DeepLabV3/inference.py \
--data_directory=/ibex/scratch/riviers/TorchGeo/DeepLabV3/50p_Dataset \
--exp_directory=/ibex/scratch/riviers/TorchGeo/DeepLabV3_Output/50p_${SLURM_ARRAY_TASK_ID}_050_Test \
"$@"
echo "--------40%------------"
python DeepLabV3/inference.py \
--data_directory=/ibex/scratch/riviers/TorchGeo/DeepLabV3/50p_Dataset \
--exp_directory=/ibex/scratch/riviers/TorchGeo/DeepLabV3_Output/50p_${SLURM_ARRAY_TASK_ID}_040_Test \
"$@"
echo "--------30%------------"
python DeepLabV3/inference.py \
--data_directory=/ibex/scratch/riviers/TorchGeo/DeepLabV3/50p_Dataset \
--exp_directory=/ibex/scratch/riviers/TorchGeo/DeepLabV3_Output/50p_${SLURM_ARRAY_TASK_ID}_030_Test \
"$@"
echo "--------20%------------"
python DeepLabV3/inference.py \
--data_directory=/ibex/scratch/riviers/TorchGeo/DeepLabV3/50p_Dataset \
--exp_directory=/ibex/scratch/riviers/TorchGeo/DeepLabV3_Output/50p_${SLURM_ARRAY_TASK_ID}_020_Test \
"$@"
echo "--------10%------------"
python DeepLabV3/inference.py \
--data_directory=/ibex/scratch/riviers/TorchGeo/DeepLabV3/50p_Dataset \
--exp_directory=/ibex/scratch/riviers/TorchGeo/DeepLabV3_Output/50p_${SLURM_ARRAY_TASK_ID}_010_Test \
"$@"
echo "... script terminated"


date