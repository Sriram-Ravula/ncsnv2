import numpy as np
import glob
import tqdm
import logging
import os
import time
from datetime import timedelta
import gc

from skimage.metrics import structural_similarity as SSIM
from skimage.metrics import peak_signal_noise_ratio as PSNR

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision.utils import make_grid, save_image

from models import anneal_Langevin_dynamics
from models import get_sigmas
from models.ncsnv2 import NCSNv2Deepest, NCSNv2Deepest2
from models.ema import DDPEMAHelper

from parallel_inference.vol_dataset import IbaltParallel

def setup(args_score):
    dist.init_process_group(backend=args_score.dist_backend, init_method=args_score.dist_url,
                                world_size=args_score.world_size, rank=args_score.rank)

def cleanup():
    dist.destroy_process_group()

def grab_data(args_score, config_score, args_par, config_par, n_shots, pin_memory=False):
    rank = args_score.rank
    world_size = args_score.world_size

    dataset = IbaltParallel(args_par, config_par, n_shots)

    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False)

    dataloader = DataLoader(dataset, batch_size=1, pin_memory=pin_memory, \
                        num_workers=1, drop_last=False, sampler=sampler, persistent_workers=True)
    
    return dataloader

def resume(device, ckpt_pth, score, ema=None, sigmas=None):
    map_location = {'cuda:%d' % 0: 'cuda:%d' % device}

    states = torch.load(ckpt_pth, map_location=map_location)

    score.load_state_dict(states[0])

    if ema is not None:
        ema.load_state_dict(states[4])
    
    if sigmas is not None:
        sigmas.copy_(states[5])
    
    return 

def run_vol(args_score, config_score, args_par, config_par):
    setup(args_score)

    #set up the score-based model and parallelize
    torch.cuda.set_device(config_score.device)
    torch.cuda.empty_cache()

    #TODO allow new scroe model as well!
    score = NCSNv2Deepest(config_score).to(config_score.device)
    score = torch.nn.SyncBatchNorm.convert_sync_batchnorm(score)
    score = DDP(score, device_ids=[config_score.device], output_device=config_score.device, find_unused_parameters=False)

    #Set up the exponential moving average
    if config_score.model.ema:
        ema_helper = DDPEMAHelper(mu=config_score.model.ema_rate, rank=config_score.device)
        ema_helper.register(score)
    
    #set up sigmas and n_shots
    sigmas = get_sigmas(config_score).to(config_score.device)
    n_shots = np.asarray(config_score.model.n_shots).squeeze()
    n_shots = torch.from_numpy(n_shots).to(config_score.device)
    if n_shots.numel() == 1:
        n_shots = torch.unsqueeze(n_shots, 0)

    #set up logging (rank 0 only)
    if args_score.rank == 0:
        # setup logger
        level = getattr(logging, args_score.verbose.upper(), None)
        if not isinstance(level, int):
            raise ValueError('level {} not supported'.format(args_score.verbose))

        handler1 = logging.StreamHandler()
        handler2 = logging.FileHandler(os.path.join(exp_path, 'stdout.txt'))
        formatter = logging.Formatter('%(levelname)s - %(filename)s - %(asctime)s - %(message)s')
        handler1.setFormatter(formatter)
        handler2.setFormatter(formatter)
        logger = logging.getLogger()
        logger.addHandler(handler1)
        logger.addHandler(handler2)
        logger.setLevel(level)

        logging.info("\nSTARTING EXPERIMENT!\n\n")
        train_start = time.time()
    
    #load the weights and stuff
    if args_score.rank == 0:
        logging.info("\n\nRESUMING - LOADING SAVED WEIGHTS\n\n")
        tic = time.time()

    load_path = os.path.join(args_score.log_path, 'checkpoint.pth')

    if config_score.model.ema:
        resume(config_score.device, load_path, score, ema=ema_helper, sigmas=sigmas)
    else:
        resume(config_score.device, load_path, score, sigmas=sigmas)
    
    if args_score.rank == 0:
        logging.info("\n\nFINISHED LOADING FROM CHECKPOINT!\n\n")
        toc = time.time()
        logging.info("TIME ELAPSED: " + str(timedelta(seconds=(toc-tic)//1)))

    #grab the dataset
    dataloader = grab_data(args_score, config_score, args_par, config_par, n_shots)

    ############################
    #NOTE     DENOISING    #NOTE
    ############################
    if args_score.rank == 0:
        logging.info("\nSTARTING DENOISING!\n\n")
        sample_start = time.time()
    
    #set the ema as the test model
    if config_score.model.ema:
        test_score = ema_helper.ema_copy(score)
    else:
        test_score = score
    test_score.eval()

    for i, batch in enumerate(dataloader):
        img, imgref, vel, slice_id, sample_idx = batch

        #NOTE figure out which of these we want as numpy arrays and which as tensors
        x_mod = img.to(config_score.device)
        x = imgref.to(config_score.device)
        vel_torch = vel.to(config_score.device)
        slice_ids = torch.tensor(slice_id, device=config_score.device)
        sample_ids = torch.tensor(sample_idx, device=config_score.device)

        #TODO 
        #(1) Langevin Dynamics w/gradient clipping and masking support
        #(2) Calculate Metrics on each device individually
        #(3) Organize and compile results on GPU 0
        #(4) Save and Log stuff
        