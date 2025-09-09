#!/bin/bash
#SBATCH --time=3-00:00:00               # Max = 7 days
#SBATCH --job-name=fuzzy_dr_training    # Give it your own name
#SBATCH --partition=compchemq           # The partition to run on
#SBATCH --mem=10G                       # Maximum amount of RAM
#SBATCH --qos=compchem                  # Needed to run on compchemq
#SBATCH -N1 --ntasks-per-node=1         # Probably keep as it
#SBATCH --cpus-per-task=30              # Number of CPU's to give the calculation (Max=56)
#SBATCH --ntasks-per-socket=1           # Probably keep as is, use to increase number of nodes
#SBATCH --gres=gpu:1                    # Number of GPU's (Max=4)
#SBATCH --get-user-env                  # Loads your current conda environment
#SBATCH --output=logs/%x-%j.out         # Output file
#SBATCH --error=logs/%x-%j.err          # Error file

# Create logs directory
mkdir -p logs

# Load CUDA *before* activating conda
module load cuda-12.2.2

# Load anaconda
module load anaconda-uoneasy/2023.09-0

# Initialize conda and activate environment
conda init bash
source ~/.bashrc
source activate pose_env

# Verify environment
echo "Python path: $(which python)"
echo "Conda prefix: $CONDA_PREFIX"

# Run your training
python -m dr_former.trainer.train \
--orig_dir data/seq_ecef \
--dr_out_dir data/dr_results \
--scaler_pkl data/dr_results/train_standard_scalers.pkl \
--mode inverted --target_indices 0 1 2 \
--epochs 50
