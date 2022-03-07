import numpy as np
import glob
import tqdm
import logging
import os
import time
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
from models.ncsnv2 import NCSNv2Deepest
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

def grab_data(args, config, rank, world_size, pin_memory=False):
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
    score = NCSNv2Deepest(config).to(rank)
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
    if config.data.dataset == 'RTM_N':
        n_shots = np.asarray(config.model.n_shots).squeeze()
        n_shots = torch.from_numpy(n_shots).to(rank)
        if n_shots.numel() == 1:
            n_shots = torch.unsqueeze(n_shots, 0)
        
        if config.model.sigma_dist == 'rtm_dynamic':
            total_n_shots_count = torch.zeros(n_shots.numel(), device=rank)
            sigmas_running = torch.zeros(n_shots.numel(), device=rank)
    
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
        
        if rank == 0:
            logging.info("\n\nFINISHED LOADING FROM CHECKPOINT!\n\n")
            toc = time.time()
            logging.info("TIME ELAPSED: " + str(toc - tic))
    
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

    #main training loop
    for epoch in range(start_epoch, config.training.n_epochs):
        if rank == 0:
            logging.info("\nSTARTING EPOCH " + str(epoch) + " !\n\n")
            train_epoch_start = time.time()
            #delete the states variable afterward in order to save CUDA memory
            mem_use = str(torch.cuda.memory_allocated(device=rank) / 2**30)
            mem_res = str(torch.cuda.memory_reserved(device=rank) / 2**30)
            print("\n\nGB ALLOCATED: " + mem_use)
            print("GB RESERVED: " + mem_res + "\n\n")

        train_loader.sampler.set_epoch(epoch)
        epoch_train_loss = 0
        num_samples = 0
        if config.model.sigma_dist == 'rtm_dynamic':
            total_n_shots_count_epoch = torch.zeros(n_shots.numel(), device=rank)
            sigmas_running_epoch = torch.zeros(n_shots.numel(), device=rank)
        score.train()

        for i, batch in enumerate(train_loader):
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
                total_n_shots_count_epoch += n_shots_count
                sigmas_running_epoch += sum_rmses_list
            else:
                train_loss = anneal_dsm_score_estimation(score, batch, sigmas, anneal_power=config.training.anneal_power)
            
            optimizer.zero_grad(set_to_none=True)
            train_loss.backward()
            optimizer.step()

            #log batch stats
            if rank == 0:
                tb_logger.add_scalar('mean_train_loss_batch', train_loss.item(), global_step=step)
                logging.info("step: {}, Batch mean train loss: {}".format(step, train_loss.item()))
                if step == 0:
                    tb_logger.add_image('training_rtm_243', make_grid(X, 4), global_step=step)
                    if config.data.dataset == 'RTM_N':
                        tb_logger.add_image('training_rtm_n', make_grid(X_perturbed, 4), global_step=step)
                        np.savetxt(os.path.join(args.log_sample_path, 'train_slice_ids_{}.txt'.format(step)), y)
                        np.savetxt(os.path.join(args.log_sample_path, 'train_nshots_idx_{}.txt'.format(step)), idx)
        
            if config.model.ema:
                ema_helper.update(score)
            
            with torch.no_grad():
                epoch_train_loss += train_loss.item() * X.shape[0] #adding sum instead of mean loss
                num_samples += X.shape[0]

            step += 1
        
        #try to free up GPU memory
        score.module.zero_grad(set_to_none=True)
        del X
        if config.data.dataset == 'RTM_N':
            del X_perturbed
        gc.collect()
        torch.cuda.empty_cache()

        #gather communal stats for the epoch
        torch.cuda.set_device(rank)
        data = {"epoch_train_loss": epoch_train_loss,
                "num_samples": num_samples}
        if config.model.sigma_dist == 'rtm_dynamic':
            data["total_n_shots_count_epoch"] = total_n_shots_count_epoch
            data["sigmas_running_epoch"] = sigmas_running_epoch
        outputs = [None for _ in range(world_size)]
        dist.all_gather_object(outputs, data)

        # #compile communal stats
        # total_train_loss = 0
        # ns = 0
        # for out in outputs:
        #     total_train_loss += out["epoch_train_loss"]
        #     ns += out["num_samples"]
        #     if config.model.sigma_dist == 'rtm_dynamic':
        #         total_n_shots_count += out["total_n_shots_count_epoch"].to(rank)
        #         sigmas_running += out["sigmas_running_epoch"].to(rank)
        # epoch_train_loss = total_train_loss / ns

        # #update sigmas if necessary
        # if config.model.sigma_dist == 'rtm_dynamic':
        #     offset_eps = 1e-8
        #     sigmas = (sigmas_running + offset_eps) / (total_n_shots_count + offset_eps)
        #     score.module.set_sigmas(sigmas)

        # #print training epoch stats
        # if rank == 0:
        #     logging.info("\n\nFINISHED TRAIN EPOCH!\n\n")
        #     train_epoch_end = time.time()
        #     logging.info("TRAIN EPOCH TIME: " + str(train_epoch_end - train_epoch_start))

        #     tb_logger.add_scalar('mean_train_loss_epoch', epoch_train_loss, global_step=epoch)
        #     logging.info("Epoch mean train loss: {}".format(epoch_train_loss))

        #try to free up GPU memory
        del outputs
        del data
        gc.collect()
        torch.cuda.empty_cache()

        #check if we need to checkpoint
        if (epoch + 1) % config.training.checkpoint_freq == 0:
            if rank == 0:
                logging.info("\nSAVING CHECKPOINT\n\n")
                checkpoint_start = time.time()

                if config.model.sigma_dist == 'rtm_dynamic':
                    np.savetxt(os.path.join(args.log_sample_path, 'sigmas_{}.txt'.format(epoch)), sigmas.detach().cpu().numpy())
                    np.savetxt(os.path.join(args.log_sample_path, 'shot_count_{}.txt'.format(epoch)), total_n_shots_count.detach().cpu().numpy())
                    np.savetxt(os.path.join(args.log_sample_path, 'sigmas_running_{}.txt'.format(epoch)), sigmas_running.detach().cpu().numpy())

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
                logging.info("CHECKPOINT TIME: " + str(checkpoint_end - checkpoint_start))

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

            test_loader.sampler.set_epoch(epoch)
            epoch_test_loss = 0
            num_samples = 0

            for i, batch in enumerate(test_loader):
                if config.data.dataset == 'RTM_N':
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
                
                    epoch_test_loss += test_loss.item() * X.shape[0]
                    num_samples += X.shape[0]
            
            #try to free up GPU memory
            del test_score
            del X
            if config.data.dataset == 'RTM_N':
                del X_perturbed
            gc.collect()
            torch.cuda.empty_cache()

            #gather communal stats for the epoch
            torch.cuda.set_device(rank)
            data = {"epoch_test_loss": epoch_test_loss,
                    "num_samples": num_samples}
            outputs = [None for _ in range(world_size)]
            dist.all_gather_object(outputs, data)

            #compile communal stats
            total_test_loss = 0
            ns = 0
            for out in outputs:
                total_test_loss += out["epoch_test_loss"]
                ns += out["num_samples"]
            epoch_test_loss = total_test_loss / ns

            #print test epoch stats
            if rank == 0:
                logging.info("\n\nFINISHED TEST EPOCH!\n\n")
                test_epoch_end = time.time()
                logging.info("TEST EPOCH TIME: " + str(test_epoch_end - test_epoch_start))

                tb_logger.add_scalar('mean_test_loss_epoch', epoch_test_loss, global_step=epoch)
                logging.info("Epoch mean test loss: {}".format(epoch_test_loss))

            #try to free up GPU memory
            del data
            del output
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
            init_samples = torch.rand(num_test_samples, config.data.channels, config.data.image_size, 
                                            config.data.image_size, device=rank)

            if config.data.dataset == 'RTM_N':
                true_samples = torch.zeros(num_test_samples, config.data.channels, config.data.image_size, 
                                                config.data.image_size, device=rank)
                slice_ids = []
                shot_idxs = []
                for i in range(num_test_samples):
                    X, y, X_perturbed, idx = test_loader.dataset.dataset.get_samples(index=i, shot_idx=config.sampling.shot_idx)
                    init_samples[i] = X_perturbed.to(rank)
                    true_samples[i] = X.to(rank)
                    slice_ids.append(y)
                    shot_idxs.append(idx)
            
            #generate the samples
            output_samples = anneal_Langevin_dynamics(x_mod=init_samples, scorenet=test_score, sigmas=sigmas,\
                                n_steps_each=config.sampling.n_steps_each, step_lr=config.sampling.step_lr, \
                                final_only=config.sampling.final_only, verbose=True if rank == 0 else False,\
                                denoise=config.sampling.denoise, add_noise=config.sampling.add_noise)

            #try to free up GPU memory
            del test_score
            gc.collect()
            torch.cuda.empty_cache()

            #gather samples
            torch.cuda.set_device(rank)
            data = {"output_samples": output_samples[-1]}
            if config.data.dataset == 'RTM_N':
                data["init_samples"] = init_samples
                data["slice_ids"] = slice_ids
                data["shot_idxs"] = shot_idxs
                data["true_samples"] = true_samples
            outputs = [None for _ in range(world_size)]
            dist.all_gather_object(outputs, data)

            #compile the results
            output_samples = None
            if config.data.dataset == 'RTM_N':
                init_samples = None
                true_samples = None
                slice_ids = []
                shot_idxs = []
            for out in outputs:
                output_samples = torch.cat((output_samples, out["output_samples"].to(rank)), 0) \
                                    if output_samples is not None else out["output_samples"].to(rank)
                if config.data.dataset == 'RTM_N':
                    init_samples = torch.cat((init_samples, out["init_samples"].to(rank)), 0) \
                                        if init_samples is not None else out["init_samples"].to(rank)
                    true_samples = torch.cat((true_samples, out["true_samples"].to(rank)), 0) \
                                        if true_samples is not None else out["true_samples"].to(rank)
                    slice_ids.extend(out["slice_ids"])
                    shot_idxs.extend(out["shot_idxs"])
            
            #save and log and stuff
            if rank == 0:
                image_grid = make_grid(output_samples, 6)
                save_image(image_grid, os.path.join(args.log_sample_path, 'output_samples_{}.png'.format(epoch)))
                tb_logger.add_image('output_samples', image_grid, global_step=epoch)

                if config.data.dataset == 'RTM_N':
                    mse = torch.nn.MSELoss()(output_samples, true_samples)

                    tb_logger.add_scalar('sample_mse', mse.item(), global_step=epoch)
                    logging.info("Sample MSE: {}".format(mse.item()))

                    np.savetxt(os.path.join(args.log_sample_path, 'sample_slice_ids_{}.txt'.format(epoch)), slice_ids)
                    np.savetxt(os.path.join(args.log_sample_path, 'sample_shot_idxs_{}.txt'.format(epoch)), shot_idxs)

                    image_grid = make_grid(init_samples, 6)
                    save_image(image_grid, os.path.join(args.log_sample_path, 'init_samples_{}.png'.format(epoch)))
                    tb_logger.add_image('init_samples', image_grid, global_step=epoch)

                    image_grid = make_grid(true_samples, 6)
                    save_image(image_grid, os.path.join(args.log_sample_path, 'true_samples_{}.png'.format(epoch)))
                    tb_logger.add_image('true_samples', image_grid, global_step=epoch)

                logging.info("\n\nFINISHED SAMPLING!\n\n")
                train_epoch_end = time.time()
                logging.info("SAMPLING TIME: " + str(train_epoch_end - train_epoch_start))

            #try to free up GPU memory
            del init_samples
            del output_samples
            del data
            del outputs
            if config.data.dataset == 'RTM_N':
                del true_samples
            gc.collect()
            torch.cuda.empty_cache()
        
        #finish off epoch
        if rank == 0:
            logging.info("\n\nFINISHED EPOCH!\n\n")
            train_epoch_end = time.time()
            logging.info("TOTAL EPOCH TIME: " + str(train_epoch_end - train_epoch_start))

    #finish distributed processes
    cleanup()
