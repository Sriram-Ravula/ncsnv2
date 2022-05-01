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
from models.ncsnv2 import NCSNv2Deepest, NCSNv2Deepest2
from models.ema import DDPEMAHelper

from datasets import get_dataset

from losses import get_optimizer
from losses.dsm import anneal_dsm_score_estimation, rtm_loss
    
def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def grab_data(args, config, rank, world_size, pin_memory=False): #TODO this should be false
    dataset, test_dataset = get_dataset(args, config)

    train_sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True, drop_last=False)
    test_sampler = DistributedSampler(test_dataset, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False)

    train_dataloader = DataLoader(dataset, batch_size=config.training.batch_size, pin_memory=pin_memory, \
                        num_workers=config.data.num_workers, drop_last=False, sampler=train_sampler, persistent_workers=True)
    test_dataloader = DataLoader(test_dataset, batch_size=config.training.batch_size, pin_memory=pin_memory, \
                        num_workers=config.data.num_workers, drop_last=False, sampler=test_sampler)
    
    return train_dataloader, test_dataloader

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

def train(rank, world_size, args, config):
    setup(rank, world_size)

    #set up the score-based model and parallelize
    torch.cuda.set_device(rank)
    torch.cuda.empty_cache()
    #score = NCSNv2Deepest(config).to(rank)
    score = NCSNv2Deepest2(config).to(rank)
    score = torch.nn.SyncBatchNorm.convert_sync_batchnorm(score)
    score = DDP(score, device_ids=[rank], output_device=rank, find_unused_parameters=False)

    #Grab the datasets
    train_loader, test_loader = grab_data(args, config, rank, world_size)
    
    #Set up the exponential moving average
    if config.model.ema:
        ema_helper = DDPEMAHelper(mu=config.model.ema_rate, rank=rank)
        ema_helper.register(score)

    #set up optimizer
    optimizer = get_optimizer(config, score.parameters())
    start_epoch = 0
    step = 0

    #set up sigmas and any running lists
    sigmas = get_sigmas(config).to(rank)
    if config.data.dataset in ['IBALT_RTM_N', 'RTM_N']:
        n_shots = np.asarray(config.model.n_shots).squeeze()
        n_shots = torch.from_numpy(n_shots).to(rank)
        if n_shots.numel() == 1:
            n_shots = torch.unsqueeze(n_shots, 0)
        
        if config.model.sigma_dist == 'rtm_dynamic':
            total_n_shots_count = torch.zeros(n_shots.numel(), device=rank)
            sigmas_running = torch.zeros(n_shots.numel(), device=rank)

    #set up logging (rank 0 only)
    if rank == 0:
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
        if rank == 0:
            logging.info("\n\nRESUMING TRAINING - LOADING SAVED WEIGHTS\n\n")
            tic = time.time()

        load_path = os.path.join(args.log_path, 'checkpoint.pth')
        eps = config.optim.eps

        if config.model.ema:
            if config.model.sigma_dist == 'rtm_dynamic':
                start_epoch, step = resume(rank, load_path, score, eps, optimizer, ema_helper, sigmas, total_n_shots_count, sigmas_running)
            else:
                start_epoch, step = resume(rank, load_path, score, eps, optimizer, ema_helper)
        else:
            if config.model.sigma_dist == 'rtm_dynamic':
                start_epoch, step = resume(rank, load_path, score, eps, optimizer, None, sigmas, total_n_shots_count, sigmas_running)
            else:
                start_epoch, step = resume(rank, load_path, score, eps, optimizer)

        start_epoch += 1 #account for when checkpointing occurs!
        
        if rank == 0:
            logging.info("\n\nFINISHED LOADING FROM CHECKPOINT!\n\n")
            toc = time.time()
            logging.info("TIME ELAPSED: " + str(timedelta(seconds=(toc-tic)//1)))

    #main training loop
    for epoch in range(start_epoch, config.training.n_epochs):
        if rank == 0:
            logging.info("\nSTARTING EPOCH " + str(epoch) + " !\n\n")
            train_epoch_start = time.time()

        train_loader.sampler.set_epoch(epoch) #in DDP we must tell distributed samplers what epoch it is for randomization

        #prepare counters for epoch stats
        epoch_train_loss = torch.zeros(1, device=rank) 
        num_samples = torch.zeros(1, device=rank)
        if config.model.sigma_dist == 'rtm_dynamic':
            total_n_shots_count_epoch = torch.zeros(n_shots.numel(), device=rank)
            sigmas_running_epoch = torch.zeros(n_shots.numel(), device=rank)

        score.train()

        for i, batch in enumerate(train_loader):
            if config.data.dataset in ['IBALT_RTM_N', 'RTM_N']:
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
                total_n_shots_count_epoch += n_shots_count
                sigmas_running_epoch += sum_rmses_list
            else:
                train_loss = anneal_dsm_score_estimation(score, batch, sigmas, anneal_power=config.training.anneal_power)

            # #checking if we have a valid loss or not
            # if torch.isfinite(train_loss) and (not torch.isnan(train_loss)):
            #     valid_grad = torch.ones(1, device=rank)
            # else: 
            #     valid_grad = torch.zeros(1, device=rank)
            # dist.all_reduce(valid_grad, op=dist.ReduceOp.SUM)

            # if valid_grad < world_size:
            #     del train_loss
            #     if rank == 0:
            #         logging.info("step: {}, Bad loss - skipping batch".format(step))
            #     continue

            #checking if we have a valid loss or not
            if torch.isfinite(train_loss) and (not torch.isnan(train_loss)):
                valid_grad = torch.tensor(0, device=rank)
            else:
                valid_grad = torch.tensor(2, device=rank) ** rank
            dist.all_reduce(valid_grad, op=dist.ReduceOp.SUM)

            if valid_grad > 0:
                del train_loss

                bad_ranks = []
                while(True):
                    bad = int(torch.log2(valid_grad))
                    valid_grad = valid_grad - 2**bad
                    bad_ranks.append(bad)
                    if valid_grad == 0:
                        break

                if rank == 0:
                    logging.info("step: {}, Bad loss ranks {}".format(step, bad_ranks))
                continue

            optimizer.zero_grad(set_to_none=True) # moving before loss calc for loss scrubbing
            train_loss.backward()
            optimizer.step()

            #log batch stats
            if rank == 0:
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
        if config.model.sigma_dist == 'rtm_dynamic':
            dist.all_reduce(total_n_shots_count_epoch, op=dist.ReduceOp.SUM)
            dist.all_reduce(sigmas_running_epoch, op=dist.ReduceOp.SUM)
        
        #reduce over stats
        epoch_train_loss = epoch_train_loss / num_samples
        if config.model.sigma_dist == 'rtm_dynamic':
            total_n_shots_count += total_n_shots_count_epoch
            sigmas_running += sigmas_running_epoch

            offset_eps = 1e-8
            sigmas = (sigmas_running + offset_eps) / (total_n_shots_count + offset_eps)
            score.module.set_sigmas(sigmas)

        #print training epoch stats
        if rank == 0:
            logging.info("\n\nFINISHED TRAIN EPOCH!\n\n")
            train_epoch_end = time.time()
            logging.info("TRAIN EPOCH TIME: " + str(timedelta(seconds=(train_epoch_end-train_epoch_start)//1)))

            tb_logger.add_scalar('mean_train_loss_epoch', epoch_train_loss.item(), global_step=epoch)
            logging.info("Epoch mean train loss: {:.1f}".format(epoch_train_loss.item()))

        #try to free up GPU memory
        del epoch_train_loss
        del num_samples
        if config.model.sigma_dist == 'rtm_dynamic':
            del total_n_shots_count_epoch
            del sigmas_running_epoch
        gc.collect()
        torch.cuda.empty_cache()

        #check if we need to checkpoint
        if (epoch + 1) % config.training.checkpoint_freq == 0:
            if rank == 0:
                logging.info("\nSAVING CHECKPOINT\n\n")
                checkpoint_start = time.time()

                if config.model.sigma_dist == 'rtm_dynamic':
                    np.savetxt(os.path.join(args.log_path, 'sigmas_{}.txt'.format(epoch)), sigmas.detach().cpu().numpy())
                    np.savetxt(os.path.join(args.log_path, 'shot_count_{}.txt'.format(epoch)), total_n_shots_count.detach().cpu().numpy())
                    np.savetxt(os.path.join(args.log_path, 'sigmas_running_{}.txt'.format(epoch)), sigmas_running.detach().cpu().numpy())

                if config.model.ema:
                    if config.model.sigma_dist == 'rtm_dynamic':
                        checkpoint(args.log_path, score, optimizer, epoch, step, ema_helper, sigmas, total_n_shots_count, sigmas_running)
                    else:
                        checkpoint(args.log_path, score, optimizer, epoch, step, ema_helper)
                else:
                    if config.model.sigma_dist == 'rtm_dynamic':
                        checkpoint(args.log_path, score, optimizer, epoch, step, None, sigmas, total_n_shots_count, sigmas_running)
                    else:
                        checkpoint(args.log_path, score, optimizer, epoch, step)
            
            if rank == 0:
                logging.info("\n\nFINISHED CHECKPOINTING!\n\n")
                checkpoint_end = time.time()
                logging.info("CHECKPOINT TIME: " + str(timedelta(seconds=(checkpoint_end-checkpoint_start)//1)))

            #keep processes from moving on until rank 0 has finished saving
            #this prevents changes to the state variables while saving                          
            dist.barrier() 

        #check if we want to perform a test epoch
        if (epoch + 1) % config.training.test_freq == 0:
            if rank == 0:
                logging.info("\nSTARTING TEST EPOCH!\n\n")
                test_epoch_start = time.time()

            if config.model.ema:
                test_score = ema_helper.ema_copy(score)
            else:
                test_score = score
            test_score.eval()

            test_loader.sampler.set_epoch(epoch) #in DDP we must tell distributed samplers what epoch it is for randomization

            #prepare counters for epoch stats
            epoch_test_loss = torch.zeros(1, device=rank) 
            num_samples = torch.zeros(1, device=rank)

            for i, batch in enumerate(test_loader):
                if config.data.dataset in ['IBALT_RTM_N', 'RTM_N']:
                    X, y, X_perturbed, idx = batch 
                    X = X.to(rank)
                    X_perturbed = X_perturbed.to(rank)
                    batch = (X, y, X_perturbed, idx)
                else:
                    X, y = batch
                    X = X.to(rank)
                    batch = (X, y)
                
                with torch.no_grad():
                    if config.model.sigma_dist == 'rtm':
                        test_loss = rtm_loss(test_score, batch, n_shots, sigmas, \
                                        dynamic_sigmas=False, anneal_power=config.training.anneal_power, val=True)
                    elif config.model.sigma_dist == 'rtm_dynamic':
                        test_loss, _, _ = rtm_loss(test_score, batch, n_shots, sigmas, \
                                                    dynamic_sigmas=True, anneal_power=config.training.anneal_power, val=True)
                    else:
                        test_loss = anneal_dsm_score_estimation(test_score, batch, sigmas, anneal_power=config.training.anneal_power)
                
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
            if rank == 0:
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
    
        #check if we would like to sample from the score network
        if (epoch + 1) % config.training.sample_freq == 0:
            if rank == 0:
                logging.info("\nSTARTING SAMPLING!\n\n")
                sample_start = time.time()

            #set the ema as the test model
            if config.model.ema:
                test_score = ema_helper.ema_copy(score)
            else:
                test_score = score
            test_score.eval()

            #prepare the initial samples for Langevin
            num_test_samples = config.sampling.batch_size // world_size

            #NOTE adding rectangle support
            if isinstance(config.data.image_size, list):
                H, W = config.data.image_size
            else:
                H = config.data.image_size
                W = config.data.image_size
                
            init_samples = torch.rand(num_test_samples, config.data.channels, H, W, device=rank)

            if config.data.dataset in ['IBALT_RTM_N', 'RTM_N']:
                true_samples = torch.zeros(num_test_samples, config.data.channels, H, W, device=rank)
                slice_ids = torch.zeros(num_test_samples, device=rank)
                shot_idxs = torch.zeros(num_test_samples, device=rank)
                for i in range(num_test_samples):
                    init_idx = i if config.data.dataset == 'RTM_N' else None #for Ibalt, we can't be sure that a given index will contain a desired k, so we choose it randomly
                    init_k = config.sampling.shot_idx if config.data.dataset == 'RTM_N' else None
                    X, y, X_perturbed, idx = test_loader.dataset.dataset.get_samples(index=init_idx, shot_idx=init_k)
                    init_samples[i] = X_perturbed.to(rank)
                    true_samples[i] = X.to(rank)
                    slice_ids[i] = torch.tensor(y, device=rank)
                    shot_idxs[i] = torch.tensor(idx, device=rank)
            
            #generate the samples
            sigma_start_idx = 0
            if config.data.dataset in ['IBALT_RTM_N', 'RTM_N']:
                sigma_start_idx = config.sampling.shot_idx
            output_samples = anneal_Langevin_dynamics(x_mod=init_samples, scorenet=test_score, sigmas=sigmas[sigma_start_idx:],\
                                n_steps_each=config.sampling.n_steps_each, step_lr=config.sampling.step_lr, \
                                final_only=config.sampling.final_only, verbose=True if rank == 0 else False,\
                                denoise=config.sampling.denoise, add_noise=config.sampling.add_noise)

            #try to free up GPU memory
            del test_score
            gc.collect()
            torch.cuda.empty_cache()

            #compile all sampling stats and samples across GPUs
            out_samples = [torch.zeros_like(output_samples[-1]) for _ in range(world_size)]
            dist.all_gather(out_samples, output_samples[-1])

            if config.data.dataset in ['IBALT_RTM_N', 'RTM_N']:
                in_samples = [torch.zeros_like(init_samples) for _ in range(world_size)]
                dist.all_gather(in_samples, init_samples)

                ground_truth_samples = [torch.zeros_like(true_samples) for _ in range(world_size)]
                dist.all_gather(ground_truth_samples, true_samples)

                slice_ids_all = [torch.zeros_like(slice_ids) for _ in range(world_size)]
                dist.all_gather(slice_ids_all, slice_ids)

                shot_idxs_all = [torch.zeros_like(shot_idxs) for _ in range(world_size)]
                dist.all_gather(shot_idxs_all, shot_idxs)
            
            #reduce over the gathered stuff
            if rank == 0:
                output_samples = torch.cat(out_samples)
                if config.data.dataset in ['IBALT_RTM_N', 'RTM_N']:
                    init_samples = torch.cat(in_samples)
                    true_samples = torch.cat(ground_truth_samples)

                    slice_ids = [slice_ids_all[i].detach().cpu().numpy() for i in range(len(slice_ids_all))]
                    slice_ids = np.concatenate(slice_ids)

                    shot_idxs = [shot_idxs_all[i].detach().cpu().numpy() for i in range(len(shot_idxs_all))]
                    shot_idxs = np.concatenate(shot_idxs)
            
            #save and log and stuff
            if rank == 0:
                image_grid = make_grid(output_samples, 6)
                save_image(image_grid, os.path.join(args.log_sample_path, 'output_samples_{}.png'.format(epoch)))
                tb_logger.add_image('output_samples', image_grid, global_step=epoch)

                if config.data.dataset in ['IBALT_RTM_N', 'RTM_N']:
                    mse = torch.nn.MSELoss()(output_samples, true_samples)

                    tb_logger.add_scalar('sample_mse', mse.item(), global_step=epoch)
                    logging.info("Sample MSE: {:.3f}".format(mse.item()))

                    np.savetxt(os.path.join(args.log_sample_path, 'sample_slice_ids_{}.txt'.format(epoch)), slice_ids)
                    np.savetxt(os.path.join(args.log_sample_path, 'sample_shot_idxs_{}.txt'.format(epoch)), shot_idxs)

                    image_grid = make_grid(init_samples, 6)
                    save_image(image_grid, os.path.join(args.log_sample_path, 'init_samples_{}.png'.format(epoch)))
                    tb_logger.add_image('init_samples', image_grid, global_step=epoch)

                    image_grid = make_grid(true_samples, 6)
                    save_image(image_grid, os.path.join(args.log_sample_path, 'true_samples_{}.png'.format(epoch)))
                    tb_logger.add_image('true_samples', image_grid, global_step=epoch)

                logging.info("\n\nFINISHED SAMPLING!\n\n")
                sample_end = time.time()
                logging.info("SAMPLING TIME: " + str(timedelta(seconds=(sample_end-sample_start)//1)))

            #try to free up GPU memory
            del init_samples
            del output_samples
            del out_samples
            if config.data.dataset in ['IBALT_RTM_N', 'RTM_N']:
                del in_samples
                del true_samples
                del ground_truth_samples
                del slice_ids
                del slice_ids_all
                del shot_idxs
                del shot_idxs_all
            gc.collect()
            torch.cuda.empty_cache()
        
        #finish off epoch
        if rank == 0:
            logging.info("\n\nFINISHED EPOCH!\n\n")
            train_epoch_end = time.time()
            logging.info("TOTAL EPOCH TIME: " + str(timedelta(seconds=(train_epoch_end - train_epoch_start)//1)))
            logging.info("TOTAL TRAINING TIME: " + str(timedelta(seconds=(train_epoch_end - train_start)//1)))

    #finish distributed processes
    cleanup()
