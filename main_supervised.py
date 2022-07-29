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

import runners.supervised_runner

def parse_args_and_config():
    parser = argparse.ArgumentParser(description=globals()['__doc__'])

    parser.add_argument('--config', type=str, required=True,  help='Path to the config file')
    parser.add_argument('--seed', type=int, default=1234, help='Random seed')
    parser.add_argument('--exp', type=str, default='exp', help='Path for saving running related data.')
    parser.add_argument('--doc', type=str, required=True, help='A string for documentation purpose. '
                                                               'Will be the name of the log folder.')
    parser.add_argument('--comment', type=str, default='', help='A string for experiment comment')
    parser.add_argument('--verbose', type=str, default='info', help='Verbose level: info | debug | warning | critical')
    parser.add_argument('--resume_training', action='store_true', help='Whether to resume training')
    parser.add_argument('--ni', action='store_true', help="No interaction. Suitable for Slurm Job launcher")

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
        device = args.rank % 4 
    else:
        args.rank = args.local_rank
        device = args.local_rank

    args.log_path = os.path.join(args.exp, 'logs', args.doc)

    # parse config file
    with open(os.path.join('configs', args.config), 'r') as f:
        config = yaml.safe_load(f)
    new_config = dict2namespace(config)

    tb_path = os.path.join(args.exp, 'tensorboard', args.doc)

    if not args.resume_training:
        if os.path.exists(args.log_path):
            overwrite = False
            if args.ni:
                overwrite = True
            else:
                response = input("Folder already exists. Overwrite? (Y/N)")
                if response.upper() == 'Y':
                    overwrite = True

            if overwrite:
                if args.rank == 0:
                    shutil.rmtree(args.log_path)
                    os.makedirs(args.log_path)
                args.log_sample_path = os.path.join(args.log_path, 'samples')
                args.tb_path = tb_path
                if args.rank == 0:
                    os.makedirs(args.log_sample_path, exist_ok=True)
                    if os.path.exists(tb_path):
                        shutil.rmtree(tb_path)
            else:
                print("Folder exists. Program halted.")
                sys.exit(0)
        else:
            if args.rank == 0:
                os.makedirs(args.log_path)
            args.tb_path = tb_path
            args.log_sample_path = os.path.join(args.log_path, 'samples')
            if args.rank == 0:
                os.makedirs(args.log_sample_path, exist_ok=True)

        if args.rank == 0:
            with open(os.path.join(args.log_path, 'config.yml'), 'w') as f:
                yaml.dump(new_config, f, default_flow_style=False)
            
    else:
        args.tb_path = tb_path
        args.log_sample_path = os.path.join(args.log_path, 'samples')

    # logging.info("Using device: {}".format(device))
    # print("Using device: {}".format(device))
    new_config.device = device

    # set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = True

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
    args, config = parse_args_and_config()

    if args.rank==0:
        print("Writing log file to {}".format(args.log_path))
        print("Exp instance id = {}".format(os.getpid()))
        print("Exp comment = {}".format(args.comment))
        print("Config =")
        print(">" * 80)
        config_dict = copy.copy(vars(config))
        print(yaml.dump(config_dict, default_flow_style=False))
        print("<" * 80)

    return args, config

if __name__ == '__main__':
    args, config = main()

    if args.rank==0:
        print("Spawning " + str(args.world_size) + " processes for DDP")

    runners.supervised_runner.train(args, config)