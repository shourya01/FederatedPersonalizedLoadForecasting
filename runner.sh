#!/bin/bash -l
#SBATCH --job-name=HETERONY
#SBATCH --output=HNY.log
#SBATCH --account=NEXTGENOPT    
#SBATCH --partition=gpu
#SBATCH --gres=gpu:4
#SBATCH --time=23:59:59
#SBATCH --ntasks=13

now=$(date)
echo $now
nvidia-smi

cd ~/FederatedPersonalizedLoadForecasting

export ENVN="APPFLENV"
export STATE="NY"

module load anaconda3
source activate $ENVN
conda info

srun -n 13 --cpus-per-task 4 python train.py --state $STATE
cp train.py "experiments${STATE}/train.py"
cp train.sh "experiments${STATE}/train.sh"
