#!/bin/bash -l
#SBATCH --job-name=HETERO
#SBATCH --account=NEXTGENOPT    
#SBATCH --partition=gpu
#SBATCH --gres=gpu:4
#SBATCH --time=23:59:59
#SBATCH --output=HNY.log
#SBATCH --ntasks=13
#SBATCH --cpus-per-task=3

now=$(date)
echo $now
nvidia-smi

cd ~/FederatedPersonalizedLoadForecasting

export ENVN="APPFLENV"

module load anaconda3
source activate $ENVN
conda info

srun -n 13 python train.py --state NY
cp train.py experiments/train.py
cp train.sh experiments/train.sh
