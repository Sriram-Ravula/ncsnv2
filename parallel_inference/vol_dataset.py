from torch.utils.data import TensorDataset
from torchvision import transforms
import torch

import numpy as np
import tqdm
import os
import time
import shutil
import random
import pickle
import pandas as pd

class IbaltParallel(TensorDataset):
    def __init__(self, args, config_dict, n_shots, debug=True):
        self.args = args
        self.config = config_dict
        self.n_shots = n_shots
        self.debug = debug

        if self.debug:
            print("Starting to build dataset........")
            tic = time.time()

        #arguments in run_smld_volprl
        self.indx_lst = self.args.indx_lst
        self.orient = self.args.orient
        self.levels = self.args.levels

        #ripped directly from SMLDRunnerVOLPRL
        self.DomDir = self.config.get('DomDir')
        self.grids = self.config.get('grids',{'trn':[625,751],'ncsn':[256,256],'img':[401,1201],'ld':[256,1024]})
        self.fltr_img = self.config.get('fltr_img',[0.03,0.95])
        self.vid = self.config.get('vid')

        self.transform_list = transforms.Compose([
            transforms.Resize([self.grids['ld'][1],self.grids['ld'][0]], 
                              interpolation=transforms.InterpolationMode.BICUBIC)
        ]) 
        
        self.vel3D = np.load(self.DomDir+'velocity/vel.npy',mmap_mode='r') / 1000.0
        self.volref = np.load(self.DomDir+'img/imgref.npy',mmap_mode='r')
        self.vol = np.load(self.DomDir+self.vid+'/subdomain_image_ns.npy',mmap_mode='r')

        self.k=int(self.vid.split('_')[1])

        self.rescale = self.config.get('rescale',True)

        self.iz = self.config.get('iz', None)
        if self.iz is None:
            self.iz = (0, self.grids['ld'][1])

        if self.debug:
            toc = time.time()
            print("TIME ELAPSED: ", str(toc - tic))
            print("Finished building dataset!")
    
    def __len__(self):
        return len(self.indx_lst)
    
    def __getitem__(self, index):
        #grab the appropriate slice ID from our list
        i = self.indx_lst[index]

        if self.orient=='x':
            # vel : ld[1] x ld[0]
            vel = self.transform_list(torch.from_numpy(self.vel3D[i, :, self.iz[0]:self.iz[1]].T).unsqueeze(0).unsqueeze(0)).float()

            # img, imgref : img[0] x img[1]
            img=filterImage(self.vol[i, :, self.iz[0]:self.iz[1]], self.vel3D[i, :, self.iz[0]:self.iz[1]], self.fltr_img[1], self.fltr_img[0], 
                            N=self.k, useMask=True, rescale=self.rescale, laplace=False)

            imgref=filterImage(self.volref[i, :, self.iz[0]:self.iz[1]], self.vel3D[i, :, self.iz[0]:self.iz[1]], self.fltr_img[1], self.fltr_img[0], 
                            N=1, useMask=True, rescale=self.rescale, laplace=False)

        elif self.orient=='y':
            vel = self.transform_list(torch.from_numpy(self.vel3D[:, i, self.iz[0]:self.iz[1]].T).unsqueeze(0).unsqueeze(0)).float()

            img = filterImage(self.vol[:, i, self.iz[0]:self.iz[1]], self.vel3D[:, i, self.iz[0]:self.iz[1]], self.fltr_img[1], self.fltr_img[0], 
                            N=self.k, useMask=True, rescale=self.rescale, laplace=False)

            imgref = filterImage(self.volref[:, i, self.iz[0]:self.iz[1]], self.vel3D[:, i, self.iz[0]:self.iz[1]], self.fltr_img[1], self.fltr_img[0], 
                            N=1, useMask=True, rescale=self.rescale, laplace=False)
        
        #now convert the img and reference image to proper tensors and shapes
        imgref = self.transform_list(torch.from_numpy(imgref.T).unsqueeze(0).unsqueeze(0)).float()
        img = self.transform_list(torch.from_numpy(img.T).unsqueeze(0).unsqueeze(0)).float()
        
        return img, imgref, vel, i, index