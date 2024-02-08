#!/bin/bash -l
#SBATCH --job-name=HETERO
#SBATCH --output=HET.log
#SBATCH --account=NEXTGENOPT    
#SBATCH --partition=gpu-large
#SBATCH --gres=gpu:8
#SBATCH --time=23:59:59
#SBATCH --mem=2000GB


nvidia-smi

cd ~/FederatedPersonalizedLoadForecasting

export ENVN="APPFLENV"
export STATE="IL"
export STATE2="NY"
export PATH="/home/sbose/.latex/bin/x86_64-linux:${PATH}"
NUM_PROC=4

module load anaconda3
source activate $ENVN
conda info

srun --exclusive --ntasks 13 --cpus-per-task $NUM_PROC --mem-per-cpu 36GB python train.py --state $STATE --choice_local 0
srun --exclusive --ntasks 13 --cpus-per-task $NUM_PROC --mem-per-cpu 36GB python train.py --state $STATE --choice_local 1
srun --exclusive --ntasks 13 --cpus-per-task $NUM_PROC --mem-per-cpu 36GB python train.py --state $STATE --choice_local 2
srun --exclusive --ntasks 13 --cpus-per-task $NUM_PROC --mem-per-cpu 36GB python train.py --state $STATE --choice_local 3
wait

srun --exclusive --ntasks 13 --cpus-per-task $NUM_PROC --mem-per-cpu 36GB python train.py --state $STATE2 --choice_local 0
srun --exclusive --ntasks 13 --cpus-per-task $NUM_PROC --mem-per-cpu 36GB python train.py --state $STATE2 --choice_local 1
srun --exclusive --ntasks 13 --cpus-per-task $NUM_PROC --mem-per-cpu 36GB python train.py --state $STATE2 --choice_local 2
srun --exclusive --ntasks 13 --cpus-per-task $NUM_PROC --mem-per-cpu 36GB python train.py --state $STATE2 --choice_local 3
wait

cp train.py "experiments${STATE}/train.py"
cp train.sh "experiments${STATE}/train.sh"

cp train.py "experiments${STATE2}/train.py"
cp train.sh "experiments${STATE2}/train.sh"
