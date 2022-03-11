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
import torch.multiprocessing as mp

import runners.ddp_runner

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

    args = parser.parse_args()
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
                shutil.rmtree(args.log_path)
                os.makedirs(args.log_path)
                args.log_sample_path = os.path.join(args.log_path, 'samples')
                args.tb_path = tb_path
                os.makedirs(args.log_sample_path, exist_ok=True)
                if os.path.exists(tb_path):
                    shutil.rmtree(tb_path)
            else:
                print("Folder exists. Program halted.")
                sys.exit(0)
        else:
            os.makedirs(args.log_path)
            args.tb_path = tb_path
            args.log_sample_path = os.path.join(args.log_path, 'samples')
            os.makedirs(args.log_sample_path, exist_ok=True)

        with open(os.path.join(args.log_path, 'config.yml'), 'w') as f:
            yaml.dump(new_config, f, default_flow_style=False)
            
    else:
        args.tb_path = tb_path
        args.log_sample_path = os.path.join(args.log_path, 'samples')

    # setup logger
    # level = getattr(logging, args.verbose.upper(), None)
    # if not isinstance(level, int):
    #     raise ValueError('level {} not supported'.format(args.verbose))

    # handler1 = logging.StreamHandler()
    # handler2 = logging.FileHandler(os.path.join(args.log_path, 'stdout.txt'))
    # formatter = logging.Formatter('%(levelname)s - %(filename)s - %(asctime)s - %(message)s')
    # handler1.setFormatter(formatter)
    # handler2.setFormatter(formatter)
    # logger = logging.getLogger()
    # logger.addHandler(handler1)
    # logger.addHandler(handler2)
    # logger.setLevel(level)

    # add device
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    # logging.info("Using device: {}".format(device))
    print("Using device: {}".format(device))
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
    # logging.info("Writing log file to {}".format(args.log_path))
    # logging.info("Exp instance id = {}".format(os.getpid()))
    # logging.info("Exp comment = {}".format(args.comment))
    # logging.info("Config =")
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
    
    n_gpus = torch.cuda.device_count()
    assert n_gpus >= 2, f"Requires at least 2 GPUs to run, but got {n_gpus}"
    world_size = n_gpus

    # logging.info("Spawning " + str(world_size) + " processes")
    print("Spawning " + str(world_size) + " processes for DDP")

    mp.spawn(runners.ddp_runner.train,
             args=(world_size, args, config),
             nprocs=world_size,
             join=True)
             