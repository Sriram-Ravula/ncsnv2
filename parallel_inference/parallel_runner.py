import numpy as np
import glob
import tqdm
import logging
import os
import time
from datetime import timedelta
import gc
import pickle

from skimage.metrics import structural_similarity as SSIM
from skimage.metrics import peak_signal_noise_ratio as PSNR
from skimage.metrics import mean_squared_error as MSE

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

from rtm_utils import clipFilterTorch, maskFilterTorch, normalizeFilterTorch

from parallel_inference.vol_dataset import IbaltParallel

def setup(args_score):
    dist.init_process_group(backend=args_score.dist_backend, init_method=args_score.dist_url,
                                world_size=args_score.world_size, rank=args_score.rank)

def cleanup():
    dist.destroy_process_group()

def grab_data(args_score, config_score, args_par, config_par, n_shots, pin_memory=False):
    rank = args_score.rank
    world_size = args_score.world_size

    dataset = IbaltParallel(args_par, config_par, n_shots, rank, world_size)

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
    if args_par.new_model:
        score = NCSNv2Deepest2(config_score).to(config_score.device)
    else:
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
        level = getattr(logging, args_score.verbose.upper(), None)
        if not isinstance(level, int):
            raise ValueError('level {} not supported'.format(args_score.verbose))

        handler1 = logging.StreamHandler()
        handler2 = logging.FileHandler(os.path.join(args_score.log_path, 'stdout.txt'))
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

    if config_score.model.ema:
        resume(config_score.device, config_par.get("scorenet_ckpt_path"), score, ema=ema_helper, sigmas=sigmas)
    else:
        resume(config_score.device, config_par.get("scorenet_ckpt_path"), score, sigmas=sigmas)
    
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

    #make structures to save the final results
    if args_score.rank == 0:
        results_list = [] #holds dict entries of the final results

        num_samples = len(dataloader.dataset)
        num_outs = args_par.tmax * len(args_par.levels)
        output_shape = config_par.get('grids',{'trn':[625,751],'ncsn':[256,256],'img':[401,1201],'ld':[256,1024]})

        results_psnr = torch.zeros(num_samples, num_outs, device=config_score.device)
        results_ssim = torch.zeros(num_samples, num_outs, device=config_score.device)
        results_mse = torch.zeros(num_samples, num_outs, device=config_score.device)

        if args_par.save_all_intermediate:
            output_shape = [num_samples, num_outs, 1, output_shape['ld'][1], output_shape['ld'][0]]
        else:
            output_shape = [num_samples, 1, output_shape['ld'][1], output_shape['ld'][0]]
        results_out = torch.zeros(output_shape, device=config_score.device)

    for i, batch in enumerate(dataloader):
        if args_score.rank == 0:
            logging.info("\nSTARTING BATCH " + str(i+1) + "/" + str(len(dataloader)) + "\n\n")
            batch_start = time.time()

        #(0) grab batch and put everything in the right type
        img, imgref, vel, slice_id, sample_idx = batch

        x_mod = img.to(config_score.device)
        slice_ids = slice_id.to(config_score.device)
        sample_ids = sample_idx.to(config_score.device)
        vel = vel.to(config_score.device)

        x = imgref.detach().cpu().numpy().squeeze()
        
        #(1) Langevin Dynamics w/gradient clipping and masking support
        intermediate_imgs = []

        with torch.no_grad():
            sigma_last = sigmas[-1]

            for l in args_par.levels:
                labels = torch.ones(x_mod.shape[0], device=x_mod.device) * l
                labels = labels.long()

                sigma = sigmas[l]
                step_size = args_par.eta_ncsn * (sigma / sigma_last) ** 2

                for t in range(args_par.tmax):
                    base_grad = test_score(x_mod, labels)

                    if args_par.filter_gradient:
                        base_grad = clipFilterTorch(base_grad, args_par.filter_gradient[1], args_par.filter_gradient[0])
                    if args_par.mask_gradient:
                        base_grad = maskFilterTorch(base_grad, vel)
                    
                    x_mod = x_mod + step_size*base_grad

                    if args_par.rescale_during_ld:
                        x_mod = normalizeFilterTorch(x_mod)
                    
                    intermediate_imgs.append(torch.clip(x_mod.clone().detach(), min=0., max=1.))
        
        #(2) calculate metrics
        intermediate_psnr = []
        intermediate_ssim = []
        intermediate_mse = []
        for x_out in intermediate_imgs:
            psnr_out = PSNR(x_out.detach().cpu().numpy().squeeze(), x)
            ssim_out = SSIM(x_out.detach().cpu().numpy().squeeze(), x)
            mse_out = MSE(x_out.detach().cpu().numpy().squeeze(), x)

            intermediate_psnr.append(psnr_out)
            intermediate_ssim.append(ssim_out)
            intermediate_mse.append(mse_out)
        
        intermediate_psnr = torch.tensor(intermediate_psnr, device=x_mod.device)
        intermediate_ssim = torch.tensor(intermediate_ssim, device=x_mod.device)
        intermediate_mse = torch.tensor(intermediate_mse, device=x_mod.device)
        if args_par.save_all_intermediate: #intermediate_imgs has shape (I, 1, H, W) where I is 1 if !save_all_intermediate
            intermediate_imgs = torch.cat(intermediate_imgs).to(x_mod.device)
        else:
            intermediate_imgs = intermediate_imgs[-1].to(x_mod.device)

        #(3) gather the outputs, psnr, ssim, and slice ID on device 0

        #making lists of length (num_nodes * gpus_per_node) and
        #entries are tensors (T*\tau, 1, H, W) or (1, 1, H, W)
        out_samples = [torch.zeros_like(intermediate_imgs) for _ in range(args_score.world_size)]
        dist.all_gather(out_samples, intermediate_imgs)
        
        #tensors (T*\tau) or (1)
        out_psnr = [torch.zeros_like(intermediate_psnr) for _ in range(args_score.world_size)]
        dist.all_gather(out_psnr, intermediate_psnr)

        #tensors (T*\tau) or (1)
        out_ssim = [torch.zeros_like(intermediate_ssim) for _ in range(args_score.world_size)]
        dist.all_gather(out_ssim, intermediate_ssim)

        #tensors (T*\tau) or (1)
        out_mse = [torch.zeros_like(intermediate_mse) for _ in range(args_score.world_size)]
        dist.all_gather(out_mse, intermediate_mse)

        #tensors (1)
        out_ids = [torch.zeros_like(slice_ids) for _ in range(args_score.world_size)]
        dist.all_gather(out_ids, slice_ids)

        #tensors (1)
        out_sample_ids = [torch.zeros_like(sample_ids) for _ in range(args_score.world_size)]
        dist.all_gather(out_sample_ids, sample_ids)

        #(4) reduce, save, and log stuff
        if args_score.rank == 0:
            #save the denoised image from the first slice as a sample to debug
            if i == 0:
                save_image(imgref, os.path.join(args_score.log_path, "gt_sample.png"))
                save_image(img, os.path.join(args_score.log_path, "kshot_sample.png"))
                save_image(intermediate_imgs[-1].unsqueeze(0), os.path.join(args_score.log_path, "denoised_sample.png"))

            for i in range(len(out_samples)):
                #first add the dictionary entry to the list
                results_entry = {
                    "slice index": out_ids[i].item(),
                    "outputs": out_samples[i].detach().cpu().numpy(),
                    "psnr": out_psnr[i].detach().cpu().numpy(),
                    "ssim": out_ssim[i].detach().cpu().numpy(),
                    "mse": out_mse[i].detach().cpu().numpy()}

                results_list.append(results_entry)

                #now add the psnr and ssim and output to the proper places based on its slice_id
                idx = out_ids[i].item()

                if args_par.save_all_intermediate:
                    results_out[idx] = out_samples[i].detach().clone()
                else:
                    results_out[idx] = out_samples[i].detach().clone().squeeze(0)

                results_psnr[idx] = out_psnr[i].detach().clone()
                results_ssim[idx] = out_ssim[i].detach().clone()
                results_mse[idx] = out_mse[i].detach().clone()

            batch_end = time.time()
            logging.info("\n\nBATCH TIME: " + str(timedelta(seconds=(batch_end-batch_start)//1)) + "\n\n")
            sample_end = time.time()
            logging.info("TOTAL TIME: " + str(timedelta(seconds=(sample_end-sample_start)//1)))
        
    dist.barrier()

    if args_score.rank == 0:
        logging.info("\n\nFINISHED DENOISING!\n\n")
        sample_end = time.time()
        logging.info("TOTAL DENOISING TIME: " + str(timedelta(seconds=(sample_end-sample_start)//1)))

        #log all of the results!
        results_path = os.path.join(args_score.log_path, "results.pkl")
        with open(results_path, 'wb') as handle:
            pickle.dump(results_list, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
        output_path = os.path.join(args_score.log_path, "outputs.pt")
        torch.save(results_out, output_path)

        psnr_path = os.path.join(args_score.log_path, "psnr.pt")
        torch.save(results_psnr, psnr_path)

        ssim_path = os.path.join(args_score.log_path, "ssim.pt")
        torch.save(results_ssim, ssim_path)

        mse_path = os.path.join(args_score.log_path, "mse.pt")
        torch.save(results_mse, mse_path)

    #finish distributed processes
    cleanup()