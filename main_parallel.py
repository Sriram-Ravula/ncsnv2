import argparse
import traceback
import time
import shutil
import logging
import yaml
import sys
import os
import torch
import numpy as np
import torch.utils.tensorboard as tb
import copy
import torch.distributed as dist

import parallel_inference.parallel_runner

def parse_args_and_config(args_par, config_par):
    parser = argparse.ArgumentParser(description=globals()['__doc__'])

    #NOTE got rid of the training arguments first - these live on config_par now
    parser.add_argument('--seed', type=int, default=2022, help='Random seed')
    parser.add_argument('--verbose', type=str, default='info', help='Verbose level: info | debug | warning | critical')

    parser.add_argument('--world_size', default=-1, type=int, help='number of nodes for distributed training')
    parser.add_argument('--rank', default=-1, type=int, help='node rank for distributed training')
    parser.add_argument('--dist_url', default='env://', type=str, help='url used to set up distributed training')
    parser.add_argument('--dist_backend', default='nccl', type=str, help='distributed backend')
    parser.add_argument('--local_rank', default=-1, type=int, help='local rank for distributed training')

    args = parser.parse_args()

    if "WORLD_SIZE" in os.environ:
        args.world_size = int(os.environ["WORLD_SIZE"])

    if 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ.get("RANK", os.environ.get("SLURM_PROCID", 0)))
        device = args.rank % 4 #torch.cuda.device_count()
    else:
        args.rank = args.local_rank
        device = args.local_rank
    print(device)

    #NOTE this is now going to point to the root of the experiment save folder for denoising
    args.log_path = os.path.join(config_par.get('scorenet_experiments_path'), config_par.get('testname'))

    # parse config file
    with open(config_par.get("scorenet_config_path"), 'r') as f:
        config = yaml.unsafe_load(f)
    # new_config = dict2namespace(config)
    new_config = config

    #make sure the experiment root exists
    if not os.path.exists(args.log_path):
        if args.rank == 0:
            os.makedirs(args.log_path)

    # logging.info("Using device: {}".format(device))
    # print("Using device: {}".format(device))
    new_config.device = device

    # set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = True

    #save all of the experiment parameters!
    if args.rank == 0:
        with open(os.path.join(args.log_path, 'config_score.yml'), 'w') as f:
            yaml.dump(new_config, f, default_flow_style=False)

        with open(os.path.join(args.log_path, 'args_score.yml'), 'w') as f:
            yaml.dump(args, f, default_flow_style=False)

        with open(os.path.join(args.log_path, 'config_par.yml'), 'w') as f:
            yaml.dump(dict2namespace(config_par), f, default_flow_style=False)

        with open(os.path.join(args.log_path, 'args_par.yml'), 'w') as f:
            yaml.dump(args_par, f, default_flow_style=False)    

    return args, new_config

def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace

def main():
    ###################################
    #NOTE Experiment Configs Here #NOTE
    ###################################
    MdlDir='/scratch/projects/sparkcognition/data/models/ncsnnew_1024x256_l5-180_ibaltcntr_305/'
    NCSNSrc='/home/08087/alexard/src/ncsnv2'
    DirExp="/scratch/projects/sparkcognition/data/NCSN_VOL_PAR/"
    DomDir='/scratch/projects/sparkcognition/data/migration/ibalt/volumes/ibaltcntr_ns_krshots/'
    SMLDDir='/scratch/projects/sparkcognition/data/SMLDVOLPRL/'

    config_par={'scorenet_config_path':MdlDir+'config.yml',
        'scorenet_ckpt_path':MdlDir+'checkpoint_999.pth',
        'scorenet_model_path':NCSNSrc,
        'scorenet_sigmas_path':MdlDir+'sigmas_999.txt',
        'scorenet_experiments_path':DirExp,
        'DomDir':DomDir,
        'SMLDDir':SMLDDir,
        'testname':'ncsnnew_1024x256_l5-180_ibaltcntr_305_k80r4_06-21-22',
        'grids':{'trn':[401,1201],'ncsn':[256,1024],'img':[401,1201],'ld':[256,1024]},
        'vid':'k_80_r_4'}
    
    args_par = {
        "indx_lst": list(range(401)),
        "orient": 'y',
        "levels": np.arange(15,29,2),
        "eta_ncsn": 1.e-4,
        "tmax": 25,
        "filter_gradient": [0.001, 0.999], #set this to False if no filter
        "mask_gradient": True,
        "rescale_during_ld": False, 
        "save_all_intermediate": False,
        "new_model": True
    }
    args_par = dict2namespace(args_par)

    ###################################
    #NOTE Experiment Configs Here #NOTE
    ###################################

    args_score, config_score = parse_args_and_config(args_par, config_par)

    if args_score.rank==0:
        print("Writing log file to {}".format(args_score.log_path)) #NOTE args_score.log_path points to the experiment root now
        print("Exp instance id = {}".format(os.getpid()))
        print("Config =")
        print(">" * 80)
        config_dict = copy.copy(vars(config_score))
        print(yaml.dump(config_dict, default_flow_style=False))
        print(">" * 80)
        config_dict = config_par
        print(yaml.dump(config_dict, default_flow_style=False))
        print(">" * 80)
        config_dict = copy.copy(vars(args_par))
        print(yaml.dump(config_dict, default_flow_style=False))
        print("<" * 80)

    return args_score, config_score, args_par, config_par

if __name__ == '__main__':
    args_score, config_score, args_par, config_par = main()

    if args_score.rank==0:
        print("Spawning " + str(args_score.world_size) + " processes for DDP")

    parallel_inference.parallel_runner.run_vol(args_score, config_score, args_par, config_par)