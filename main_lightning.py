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
import copy

from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, GPUStatsMonitor

from datasets.rtm_n_lightning import RTMDataModule
from models.ncsnv2_lightning import NCSNv2_Lightning

import os

def parse_args_and_config():
    parser = argparse.ArgumentParser(description=globals()['__doc__'])

    parser.add_argument('--config', type=str, required=True,  help='Path to the config file')
    parser.add_argument('--seed', type=int, default=1234, help='Random seed')
    parser.add_argument('--exp', type=str, default='exp', help='Path for saving running related data.')
    parser.add_argument('--doc', type=str, required=True, help='A string for documentation purpose. '
                                                               'Will be the name of the log folder.')
    parser.add_argument('--comment', type=str, default='', help='A string for experiment comment')
    parser.add_argument('--verbose', type=str, default='info', help='Verbose level: info | debug | warning | critical')
    parser.add_argument('--test', action='store_true', help='Whether to test the model')
    parser.add_argument('--sample', action='store_true', help='Whether to produce samples from the model')
    parser.add_argument('--fast_fid', action='store_true', help='Whether to do fast fid test')
    parser.add_argument('--resume_training', action='store_true', help='Whether to resume training')
    parser.add_argument('-i', '--image_folder', type=str, default='images', help="The folder name of samples")
    parser.add_argument('--ni', action='store_true', help="No interaction. Suitable for Slurm Job launcher")

    args = parser.parse_args()
    args.log_path = os.path.join(args.exp, 'logs', args.doc)

    # parse config file
    with open(os.path.join('configs', args.config), 'r') as f:
        config = yaml.load(f)
    new_config = dict2namespace(config)

    tb_path = os.path.join(args.exp, 'tensorboard', args.doc)
    sample_path = os.path.join(args.log_path, 'samples')

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
                os.makedirs(sample_path)
                if os.path.exists(tb_path):
                    shutil.rmtree(tb_path)
            else:
                print("Folder exists. Program halted.")
                sys.exit(0)
        else:
            os.makedirs(args.log_path)
            os.makedirs(sample_path)
            os.makedirs(tb_path)

        with open(os.path.join(args.log_path, 'config.yml'), 'w') as f:
            yaml.dump(new_config, f, default_flow_style=False)
    
    # setup logger
    level = getattr(logging, args.verbose.upper(), None)
    if not isinstance(level, int):
        raise ValueError('level {} not supported'.format(args.verbose))

    handler1 = logging.StreamHandler()
    handler2 = logging.FileHandler(os.path.join(args.log_path, 'stdout.txt'))
    formatter = logging.Formatter('%(levelname)s - %(filename)s - %(asctime)s - %(message)s')
    handler1.setFormatter(formatter)
    handler2.setFormatter(formatter)
    logger = logging.getLogger()
    logger.addHandler(handler1)
    logger.addHandler(handler2)
    logger.setLevel(level)

    # add device
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    logging.info("Using device: {}".format(device))
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

def grab_trainer_args(config):
    args = argparse.Namespace(max_epochs = config.training.n_epochs, 
                     log_save_interval = 1,
                     row_log_interval = 1,
                     check_val_every_n_epoch = 1,
                     num_nodes = 1,
                     gpus = 4,
                     workers = config.data.num_workers,
                     distributed_backend = "ddp")

    return args

def main():
    args, config = parse_args_and_config()
    logging.info("Writing log file to {}".format(args.log_path))
    logging.info("Exp instance id = {}".format(os.getpid()))
    logging.info("Exp comment = {}".format(args.comment))
    logging.info("Config =")
    print(">" * 80)
    config_dict = copy.copy(vars(config))

    print(yaml.dump(config_dict, default_flow_style=False))
    print("<" * 80)

    dataset = RTMDataModule(path="/scratch/08269/rstone/full_rtm_8048",
                            config=config)
    model = NCSNv2_Lightning(args=args, config=config)

    logger = TensorBoardLogger(
        save_dir=os.path.join(args.exp, 'tensorboard', args.doc),
        version=args.exp,
        name='NCSNv2_TRAINING'
    )

    trainer_args = grab_trainer_args(config)
    trainer = Trainer.from_argparse_args(trainer_args, 
                                         logger=logger, 
                                         callbacks=[ModelCheckpoint(save_top_k=-1, 
                                                                    period=5,
                                                                    verbose=True), 
                                                    GPUStatsMonitor()])
    trainer.fit(model, datamodule=dataset)

    return 0

if __name__ == '__main__':
    sys.exit(main())