#!/bin/bash

#SBATCH -J ncsn_ddp_large_new_model
#SBATCH -o ncsn_ddp_large_new_model.o%j
#SBATCH -e ncsn_ddp_large_new_model.e%j
#SBATCH -p v100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH -t 48:00:00
#SBATCH --mail-user=ygandhi@sparkcognition.com
#SBATCH --mail-type=all

module load cuda
conda activate ncsnv2_env
cd $HOME/migration/ncsnv2

python3 -m torch.distributed.launch --use_env main_ddp2.py --config ddp_test_3.yml --seed 42 --exp /scratch/08087/gandhiy/ncsn_experiments/ --doc ibalt_052022_1024_256_newModel_28_lambdas --verbose info --ni
# python3 main_ddp.py --config ddp_test_3.yml --seed 42 --exp /scratch/08087/gandhiy/ncsn_experiments/ --doc ibalt_052022_1024256 --verbose info --ni