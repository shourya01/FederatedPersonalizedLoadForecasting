#!/bin/bash -l
#SBATCH --job-name=HETEROIL
#SBATCH --output=HIL.log
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
export STATE="IL"
export PATH="/home/sbose/.latex/bin/x86_64-linux:${PATH}"

module load anaconda3
source activate $ENVN
conda info

# srun -n 13 --cpus-per-task 4 python train.py --state $STATE --choice_local 0
# srun -n 13 --cpus-per-task 4 python train.py --state $STATE --choice_local 1
srun -n 13 --cpus-per-task 4 python train.py --state $STATE --choice_local 2
# srun -n 13 --cpus-per-task 4 python train.py --state $STATE --choice_local 3
cp train.py "experiments${STATE}/train.py"
cp train.sh "experiments${STATE}/train.sh"
