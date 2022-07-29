import numpy as np
import glob
import tqdm
import logging
import os
import time
from datetime import timedelta
import gc

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision.utils import make_grid, save_image
import torch.utils.tensorboard as tb

from models import anneal_Langevin_dynamics
from models import get_sigmas
from models.ncsnv2 import NCSNv2Deepest2_Supervised_NoCondition
from models.ema import DDPEMAHelper

from datasets import get_dataset

from losses import get_optimizer
from losses.dsm import supervised_unconditional
    
def setup(args):
    dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)

def cleanup():
    dist.destroy_process_group()

def grab_data(args, config, pin_memory=False): #TODO this should be false
    rank = args.rank
    world_size = args.world_size

    dataset, test_dataset = get_dataset(args, config)

    train_sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True, drop_last=False)
    test_sampler = DistributedSampler(test_dataset, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False)

    train_dataloader = DataLoader(dataset, batch_size=config.training.batch_size, pin_memory=pin_memory, \
                        num_workers=config.data.num_workers, drop_last=False, sampler=train_sampler, persistent_workers=True)
    test_dataloader = DataLoader(test_dataset, batch_size=config.training.batch_size, pin_memory=pin_memory, \
                        num_workers=config.data.num_workers, drop_last=False, sampler=test_sampler)
    
    return train_dataloader, test_dataloader

def resume(device, ckpt_pth, score, eps, optimizer, ema=None, sigmas=None, total_n_shots_count=None, sigmas_running=None):
    map_location = {'cuda:%d' % 0: 'cuda:%d' % device}

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

def train(args, config):
    setup(args)

    #set up the score-based model and parallelize
    torch.cuda.set_device(config.device)
    torch.cuda.empty_cache()

    score = NCSNv2Deepest2_Supervised_NoCondition(config).to(config.device)
    score = torch.nn.SyncBatchNorm.convert_sync_batchnorm(score)
    score = DDP(score, device_ids=[config.device], output_device=config.device, find_unused_parameters=False)

    #Grab the datasets
    train_loader, test_loader = grab_data(args, config)
    
    #Set up the exponential moving average
    if config.model.ema:
        ema_helper = DDPEMAHelper(mu=config.model.ema_rate, rank=config.device)
        ema_helper.register(score)

    #set up optimizer
    optimizer = get_optimizer(config, score.parameters())
    start_epoch = 0
    step = 0

    #set up logging (rank 0 only)
    if args.rank == 0:
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

        tb_logger = tb.SummaryWriter(log_dir=args.tb_path)
        logging.info("\nSTARTING TRAINING!\n\n")
        train_start = time.time()
    
    #check if we need to resume
    if args.resume_training:
        if args.rank == 0:
            logging.info("\n\nRESUMING TRAINING - LOADING SAVED WEIGHTS\n\n")
            tic = time.time()

        load_path = os.path.join(args.log_path, 'checkpoint.pth')
        eps = config.optim.eps

        if config.model.ema:
            start_epoch, step = resume(config.device, load_path, score, eps, optimizer, ema_helper)
        else:
            start_epoch, step = resume(config.device, load_path, score, eps, optimizer)

        start_epoch += 1 #account for when checkpointing occurs!
        
        if args.rank == 0:
            logging.info("\n\nFINISHED LOADING FROM CHECKPOINT!\n\n")
            toc = time.time()
            logging.info("TIME ELAPSED: " + str(timedelta(seconds=(toc-tic)//1)))

    #main training loop
    for epoch in range(start_epoch, config.training.n_epochs):
        if args.rank == 0:
            logging.info("\nSTARTING EPOCH " + str(epoch) + " !\n\n")
            train_epoch_start = time.time()

        train_loader.sampler.set_epoch(epoch) #in DDP we must tell distributed samplers what epoch it is for randomization

        #prepare counters for epoch stats
        epoch_train_loss = torch.zeros(1, device=config.device) 
        num_samples = torch.zeros(1, device=config.device)

        score.train()

        for i, batch in enumerate(train_loader):
            if config.data.dataset in ['IBALT_RTM_N', 'RTM_N']:
                X, y, X_perturbed, idx = batch 
                X = X.to(config.device)
                X_perturbed = X_perturbed.to(config.device)
                batch = (X, y, X_perturbed, idx)
            else:
                X, y = batch
                X = X.to(config.device)
                batch = (X, y)
            
            train_loss = supervised_unconditional(score, batch)

            optimizer.zero_grad(set_to_none=True) # moving before loss calc for loss scrubbing
            train_loss.backward()
            optimizer.step()

            #log batch stats
            if args.rank == 0:
                tb_logger.add_scalar('mean_train_loss_batch', train_loss.item(), global_step=step)
                logging.info("step: {}, Batch mean train loss: {:.1f}".format(step, train_loss.item()))
                if step == 0:
                    tb_logger.add_image('training_rtm_full', make_grid(X, 4), global_step=step)
                    if config.data.dataset in ['IBALT_RTM_N', 'RTM_N']:
                        tb_logger.add_image('training_rtm_n', make_grid(X_perturbed, 4), global_step=step)
        
            if config.model.ema:
                ema_helper.update(score)
            
            with torch.no_grad():
                epoch_train_loss += train_loss * X.shape[0] #adding sum instead of mean loss
                num_samples += X.shape[0]

            step += 1
        
        #try to free up GPU memory
        score.module.zero_grad(set_to_none=True)
        del X
        if config.data.dataset in ['IBALT_RTM_N', 'RTM_N']:
            del X_perturbed
        gc.collect()
        torch.cuda.empty_cache()

        #compile all epoch stats across all GPUs
        dist.all_reduce(epoch_train_loss, op=dist.ReduceOp.SUM)
        dist.all_reduce(num_samples, op=dist.ReduceOp.SUM)
        
        #reduce over stats
        epoch_train_loss = epoch_train_loss / num_samples

        #print training epoch stats
        if args.rank == 0:
            logging.info("\n\nFINISHED TRAIN EPOCH!\n\n")
            train_epoch_end = time.time()
            logging.info("TRAIN EPOCH TIME: " + str(timedelta(seconds=(train_epoch_end-train_epoch_start)//1)))

            tb_logger.add_scalar('mean_train_loss_epoch', epoch_train_loss.item(), global_step=epoch)
            logging.info("Epoch mean train loss: {:.1f}".format(epoch_train_loss.item()))

        #try to free up GPU memory
        del epoch_train_loss
        del num_samples
        gc.collect()
        torch.cuda.empty_cache()

        #check if we need to checkpoint
        if (epoch + 1) % config.training.checkpoint_freq == 0:
            if args.rank == 0:
                logging.info("\nSAVING CHECKPOINT\n\n")
                checkpoint_start = time.time()

                if config.model.ema:
                    checkpoint(args.log_path, score, optimizer, epoch, step, ema_helper)
                else:
                    checkpoint(args.log_path, score, optimizer, epoch, step)
            
            if args.rank == 0:
                logging.info("\n\nFINISHED CHECKPOINTING!\n\n")
                checkpoint_end = time.time()
                logging.info("CHECKPOINT TIME: " + str(timedelta(seconds=(checkpoint_end-checkpoint_start)//1)))

            #keep processes from moving on until rank 0 has finished saving
            #this prevents changes to the state variables while saving                          
            dist.barrier() 

        #check if we want to perform a test epoch
        if (epoch + 1) % config.training.test_freq == 0:
            if args.rank == 0:
                logging.info("\nSTARTING TEST EPOCH!\n\n")
                test_epoch_start = time.time()

            if config.model.ema:
                test_score = ema_helper.ema_copy(score)
            else:
                test_score = score
            test_score.eval()

            test_loader.sampler.set_epoch(epoch) #in DDP we must tell distributed samplers what epoch it is for randomization

            #prepare counters for epoch stats
            epoch_test_loss = torch.zeros(1, device=config.device) 
            num_samples = torch.zeros(1, device=config.device)

            for i, batch in enumerate(test_loader):
                if config.data.dataset in ['IBALT_RTM_N', 'RTM_N']:
                    X, y, X_perturbed, idx = batch 
                    X = X.to(config.device)
                    X_perturbed = X_perturbed.to(config.device)
                    batch = (X, y, X_perturbed, idx)
                else:
                    X, y = batch
                    X = X.to(config.device)
                    batch = (X, y)
                
                with torch.no_grad():
                    test_loss = supervised_unconditional(score, batch)

                    epoch_test_loss += test_loss * X.shape[0]
                    num_samples += X.shape[0]
            
            #try to free up GPU memory
            del test_score
            del X
            if config.data.dataset in ['IBALT_RTM_N', 'RTM_N']:
                del X_perturbed
            gc.collect()
            torch.cuda.empty_cache()

            #compile all epoch stats across all GPUs
            dist.all_reduce(epoch_test_loss, op=dist.ReduceOp.SUM)
            dist.all_reduce(num_samples, op=dist.ReduceOp.SUM)
            epoch_test_loss = epoch_test_loss / num_samples

            #print test epoch stats
            if args.rank == 0:
                logging.info("\n\nFINISHED TEST EPOCH!\n\n")
                test_epoch_end = time.time()
                logging.info("TEST EPOCH TIME: " + str(timedelta(seconds=(test_epoch_end-test_epoch_start)//1)))

                tb_logger.add_scalar('mean_test_loss_epoch', epoch_test_loss.item(), global_step=epoch)
                logging.info("Epoch mean test loss: {:.1f}".format(epoch_test_loss.item()))

            #try to free up GPU memory
            del epoch_test_loss
            del num_samples
            gc.collect()
            torch.cuda.empty_cache()
        
        #finish off epoch
        if args.rank == 0:
            logging.info("\n\nFINISHED EPOCH!\n\n")
            train_epoch_end = time.time()
            logging.info("TOTAL EPOCH TIME: " + str(timedelta(seconds=(train_epoch_end - train_epoch_start)//1)))
            logging.info("TOTAL TRAINING TIME: " + str(timedelta(seconds=(train_epoch_end - train_start)//1)))

    #finish distributed processes
    cleanup()
