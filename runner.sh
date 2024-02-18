#!/bin/bash -l
#SBATCH --job-name=HETERO
#SBATCH --output=HET.log
#SBATCH --account=NEXTGENOPT  
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:8
#SBATCH --exclusive
#SBATCH --ntasks=26
#SBATCH --time=23:59:59

nvidia-smi

cd ~/FederatedPersonalizedLoadForecasting

ENVN="APPFLENV"
STATE="IL"
STATE2="NY"
STATE3="CA"
PATH="/home/sbose/.latex/bin/x86_64-linux:${PATH}"

echo "SLURM_CPUS_ON_NODE 64"
echo "SLURM_JOB_NODELIST ${SLURM_JOB_NODELIST}"
echo "SLURM_NTASKS ${SLURM_NTASKS}"
echo "SLURM_JOB_GPUS ${SLURM_JOB_GPUS}"

# do math
CPU_PER_PROC=$((64/SLURM_NTASKS))
echo "CPU_PER_PROC ${CPU_PER_PROC}"

# start code
module load anaconda3
source activate $ENVN
conda info

srun -n 13 --gpus=4 --cpus-per-task=$CPU_PER_PROC --mem-per-cpu=10GB --exclusive python train.py --state $STATE &
srun -n 13 --gpus=4 --cpus-per-task=$CPU_PER_PROC --mem-per-cpu=10GB --exclusive python train.py --state $STATE2 &
wait
# srun -n 13 --gpus=2 --cpus-per-task=2 --mem-per-cpu=10GB --exclusive python train.py --state $STATE --choice_local 2 &
# srun -n 13 --gpus=2 --cpus-per-task=2 --mem-per-cpu=10GB --exclusive python train.py --state $STATE --choice_local 3 &
# wait

# srun -n 13 --gpus=2 --cpus-per-task=2 --mem-per-cpu=10GB --exclusive python train.py --state $STATE2 --choice_local 0 &
# srun -n 13 --gpus=2 --cpus-per-task=2 --mem-per-cpu=10GB --exclusive python train.py --state $STATE2 --choice_local 1 &
# wait
# srun -n 13 --gpus=2 --cpus-per-task=2 --mem-per-cpu=10GB --exclusive python train.py --state $STATE2 --choice_local 2 &
# srun -n 13 --gpus=2 --cpus-per-task=2 --mem-per-cpu=10GB --exclusive python train.py --state $STATE2 --choice_local 3 &
# wait

cp train.py "experiments${STATE}/train.py"
cp train.sh "experiments${STATE}/train.sh"

cp train.py "experiments${STATE2}/train.py"
cp train.sh "experiments${STATE2}/train.sh"
