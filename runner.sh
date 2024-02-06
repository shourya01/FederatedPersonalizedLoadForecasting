#!/bin/bash -l
#SBATCH --job-name=HETEROIL
#SBATCH --output=HIL.log
#SBATCH --account=NEXTGENOPT    
#SBATCH --partition=gpu
#SBATCH --gres=gpu:4
#SBATCH --time=23:59:59
#SBATCH --ntasks=13
#SBATCH --cpus-per-task=3

now=$(date)
echo $now
nvidia-smi

cd ~/FederatedPersonalizedLoadForecasting

export ENVN="APPFLENV"
export STATE="IL"

module load anaconda3
source activate $ENVN
conda info

srun -n 13 python train.py --state $STATE
cp train.py "experiments${STATE}/train.py"
cp train.sh "experiments${STATE}/train.sh"
