#!/bin/bash
#SBATCH -A 'DMS21001'
#SBATCH -o 'NCSN_VOL'
#SBATCH -p 'v100'
#SBATCH -J 'NCSN_VOL'
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH -t 01:00:00
#SBATCH --mail-user 'sriram.ravula@utexas.edu' 
#SBATCH --mail-type 'ALL'
#SBATCH --cpus-per-task 10

export MASTER_PORT=12355
export WORLD_SIZE=4

echo "NODELIST="${SLURM_NODELIST}
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
echo "MASTER_ADDR="$MASTER_ADDR

cd /scratch/04703/sravula/ncsnv2
module load conda
conda activate score2

srun python3 main_parallel.py