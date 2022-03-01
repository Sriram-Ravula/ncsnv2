import numpy as np
import glob
import tqdm
import logging
import os

import torch
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision.utils import make_grid, save_image

from models import anneal_Langevin_dynamics
from models import get_sigmas
from models.ncsnv2 import NCSNv2Deepest
from models.ema import DDPEMAHelper

from datasets import get_dataset

from losses import get_optimizer
from losses.dsm import anneal_dsm_score_estimation, rtm_loss

__all__ = ['DDPRunner']
    
def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def train(rank, world_size, args, config):
    setup(rank, world_size)

    #Grab the datasets
    train_loader, test_loader = grab_data(rank, world_size)

    #set up the score-based model and parallelize
    score = NCSNv2Deepest(config).to(rank)
    score = torch.nn.SyncBatchNorm.convert_sync_batchnorm(score)
    score = DDP(score, device_ids=[rank], output_device=rank, find_unused_parameters=True)
    
    #Set up the exponential moving average
    if self.config.model.ema:
        ema_helper = DDPEMAHelper(mu=config.model.ema_rate)
        ema_helper.register(score)

    #set up optimizer
    optimizer = get_optimizer(config, score.parameters())
    start_epoch = 0
    step = 0

    #set up sigmas and any running lists
    sigmas = get_sigmas(config).to(rank)
    if config.data.dataset == 'RTM_N':
        n_shots = np.asarray(config.model.n_shots).squeeze()
        n_shots = torch.from_numpy(n_shots).to(rank)
        if n_shots.numel() == 1:
            n_shots = torch.unsqueeze(n_shots, 0)
        
        if config.model.sigma_dist == 'rtm_dynamic':
            total_n_shots_count = torch.zeros(n_shots.numel()).to(rank) 
            sigmas_running = sigmas.clone()
    
    #check if we need to resume
    if args.resume_training:
        if rank == 0:
            print("\n\nRESUMING TRAINING - LOADING SAVED WEIGHTS\n\n")
            tic = time.time()

        load_path = os.path.join(args.log_path, 'checkpoint.pth')
        eps = config.optim.eps

        if self.config.model.ema:
            if config.model.sigma_dist == 'rtm_dynamic':
                start_epoch, step = resume(rank, load_path, score, eps, optimizer, ema_helper, sigmas, total_n_shots_count, sigmas_running)
            else:
                start_epoch, step = resume(rank, load_path, score, eps, optimizer, ema_helper)
        else:
            if config.model.sigma_dist == 'rtm_dynamic':
                start_epoch, step = resume(rank, load_path, score, eps, optimizer, None, sigmas, total_n_shots_count, sigmas_running)
            else:
                start_epoch, step = resume(rank, load_path, score, eps, optimizer)
        
        if rank == 0:
            print("\n\nFINISHED LOADING FROM CHECKPOINT!\n\n")
            toc = time.time()
            print("TIME ELAPSED: ", str(toc - tic))
    
    #set up logging (rank 0 only)
    if rank == 0:
        tb_logger = self.config.tb_logger
        print("\n\STARTING TRAINING!\n\n")
        tic = time.time()

    #main training loop
    for epoch in range(start_epoch, config.training.n_epochs):

        train_loader.sampler.set_epoch(epoch)

        epoch_train_loss = 0
        score.train()
        for i, batch in enumerate(train_loader):
            optimizer.zero_grad(set_to_none=True)

            if config.data.dataset == 'RTM_N':
                X, y, X_perturbed, idx = batch 
                X = X.to(rank)
                X_perturbed = X_perturbed.to(rank)
                batch = (X, y, X_perturbed, idx)
            else:
                X, y = batch
                X = X.to(rank)
                batch = (X, y)
            
            if config.model.sigma_dist == 'rtm':
                train_loss = rtm_loss(score, batch, n_shots, sigmas, \
                                dynamic_sigmas=False, anneal_power=config.training.anneal_power, val=False)
            elif config.model.sigma_dist == 'rtm_dynamic':
                train_loss, sum_rmses_list, n_shots_count = rtm_loss(score, batch, n_shots, sigmas, \
                                                                dynamic_sigmas=True, anneal_power=config.training.anneal_power, val=False)
            else:
                train_loss = anneal_dsm_score_estimation(score, batch, sigmas, anneal_power=config.training.anneal_power)
            
            if rank == 0:
                tb_logger.add_scalar('train_loss', train_loss, global_step=step)
                
                logging.info("step: {}, loss: {}".format(step, train_loss.item()))
            
            epoch_train_loss += train_loss.item()

            train_loss.backward()
            optimizer.step()

        #print training epoch stats
        if rank == 0:
            print("\n\nFINISHED LOADING FROM CHECKPOINT!\n\n")
            toc = time.time()
            print("TIME ELAPSED: ", str(toc - tic))
    
def grab_data(self, rank, world_size, pin_memory=False):
    dataset, test_dataset = get_dataset(self.args, self.config)

    train_sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False)
    test_sampler = DistributedSampler(test_dataset, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False)

    train_dataloader = DataLoader(dataset, batch_size=self.config.training.batch_size, pin_memory=pin_memory, \
                        num_workers=self.config.data.num_workers, drop_last=False, shuffle=True, sampler=train_sampler)
    test_dataloader = DataLoader(test_dataset, batch_size=self.config.training.batch_size, pin_memory=pin_memory, \
                        num_workers=self.config.data.num_workers, drop_last=False, shuffle=True, sampler=train_sampler)
    
    return train_dataloader, test_dataloader

def sample(self):

def resume(rank, ckpt_pth, score, eps, optimizer, ema=None, sigmas=None, total_n_shots_count=None, sigmas_running=None):
    map_location = {'cuda:%d' % 0: 'cuda:%d' % rank}

    states = torch.load(ckpt_pth, map_location=map_location)

    score.load_state_dict(states[0])

    states[1]['param_groups'][0]['eps'] = eps
    optimizer.load_state_dict(states[1])

    start_epoch = states[2]
    step = states[3]

    if ema is not None:
        ema.load_state_dict(states[4])
    
    if sigmas is not None:
        sigmas.copy_(states[5])
    
    if total_n_shots_count is not None:
        total_n_shots_count.copy_(states[6])
    
    if sigmas_running is not None:
        sigmas_running.copy_(states[7])
    
    return start_epoch, step

def checkpoint(ckpt_pth, score, optimizer, epoch, step, ema=None, sigmas=None, total_n_shots_count=None, sigmas_running=None):
    states = [
        score.state_dict(),
        optimizer.state_dict(),
        epoch,
        step
    ]

    if ema is not None:
        states.append(ema.state_dict())
    
    if sigmas is not None:
        states.append(sigmas)
    
    if total_n_shots_count is not None:
        states.append(total_n_shots_count)
    
    if sigmas_running is not None:
        states.append(sigmas_running)

    torch.save(states, os.path.join(ckpt_pth, 'checkpoint_{}.pth'.format(epoch)))
    torch.save(states, os.path.join(ckpt_pth, 'checkpoint.pth'))

    return