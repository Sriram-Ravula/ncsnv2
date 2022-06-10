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

    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True, drop_last=False)

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

    #directory for saving run information
    exp_path = os.path.join(config_par.get('scorenet_experiments_path'), config_par.get('testname'))

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

    ####################
    #     DENOISING    #
    ####################
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

        x_mod = img.to(config_score.device)
        x = imgref.to(config_score.device)
        vel_torch = vel.to(config_score.device)
        slice_ids = torch.tensor(slice_id, device=config_score.device)
        sample_ids = torch.tensor(sample_idx, device=config_score.device)

        #TODO enable this to use an arbitrary start and end sigma while grabbing correct sigma_L and c (k index)
        output_samples = anneal_Langevin_dynamics(x_mod=x_mod, scorenet=test_score, sigmas=sigmas[args_par.levels],\
                            n_steps_each=args_par.tmax, step_lr=args_par.eta_ncsn, \
                            final_only=False, verbose=True if args_score.rank == 0 else False,\
                            denoise=False, add_noise=False)
        
        #compile all sampling stats and samples across GPUs
        #NOTE this dimension may be wrong since output_samples is a list of all the intermediate
        out_samples = [torch.zeros_like(output_samples[-1]) for _ in range(args.world_size)]
        dist.all_gather(out_samples, output_samples[-1])

        in_samples = [torch.zeros_like(x_mod) for _ in range(args.world_size)]
        dist.all_gather(in_samples, x_mod)

        ground_truth_samples = [torch.zeros_like(x) for _ in range(args.world_size)]
        dist.all_gather(ground_truth_samples, x)

        slice_ids_all = [torch.zeros_like(slice_ids) for _ in range(args.world_size)]
        dist.all_gather(slice_ids_all, slice_ids)

        sample_ids_all = [torch.zeros_like(sample_ids) for _ in range(args.world_size)]
        dist.all_gather(sample_ids_all, sample_ids)

        #reduce over the gathered stuff
        if args.rank == 0:
            output_samples = torch.cat(out_samples)

            init_samples = torch.cat(in_samples)
            true_samples = torch.cat(ground_truth_samples)

            slice_ids = [slice_ids_all[i].detach().cpu().numpy() for i in range(len(slice_ids_all))]
            slice_ids = np.concatenate(slice_ids)

            shot_idxs = [shot_idxs_all[i].detach().cpu().numpy() for i in range(len(shot_idxs_all))]
            shot_idxs = np.concatenate(shot_idxs)
        
        #save and log and stuff
        if args.rank == 0:
            image_grid = make_grid(output_samples, 6)
            save_image(image_grid, os.path.join(args.log_sample_path, 'output_samples_{}.png'.format(epoch)))
            tb_logger.add_image('output_samples', image_grid, global_step=epoch)

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
        
    #finish distributed processes
    cleanup()