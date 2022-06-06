#!/bin/bash

#SBATCH -J ncsn_ddp_large_new_model
#SBATCH -o ncsn_ddp_large_new_model.o%j
#SBATCH -e ncsn_ddp_large_new_model.e%j
#SBATCH -p v100
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH -t 48:00:00
#SBATCH --mail-user=dvoytan@sparkcognition.com
#SBATCH --mail-type=all

module load cuda
module load conda
conda activate seismic
cd $HOME/ncsnv2

python3 -m torch.distributed.launch --use_env main_ddp2.py --config ddp_test_vel.yml --seed 42 --exp /scratch/06936/dvoytan/ncsn-logs/ --doc ibalt_052022_1024_256_newModel_28_lambdas --verbose debug --ni
