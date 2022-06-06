#!/bin/bash
#SBATCH -A 'DMS21001'
#SBATCH -o 'RTMN_FULL'
#SBATCH -p 'v100'
#SBATCH -J 'RTMN_FULL'
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=4
#SBATCH -t 48:00:00
#SBATCH --mail-user 'dvoytan@sparkcognition.com' 
#SBATCH --mail-type 'ALL'
#SBATCH --cpus-per-task 10

export MASTER_PORT=12355
export WORLD_SIZE=32

echo "NODELIST="${SLURM_NODELIST}
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
echo "MASTER_ADDR="$MASTER_ADDR

cd $HOME/ncsnv2
module load cuda
module load conda
conda activate seismic


srun python3 main_ddp2.py --config ddp_test_vel.yml --seed 2240 --exp /scratch/06936/dvoytan/ncsn-logs/ --doc rtmn_full --verbose debug --ni