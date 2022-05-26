from torch.utils.data import TensorDataset
import numpy as np
import torch
import tqdm
import os
import time
import shutil
import torch.nn.functional as F
from torchvision.transforms.functional import hflip as do_hflip
import random
import pickle
import pandas as pd

from rtm_utils import load_exp, load_npy, filterImage, laplaceFilter
from joblib import Parallel, delayed

class Ibalt(TensorDataset):
    def __init__(self, path, transform=None, debug=True, load_path=None, manual_hflip=True, n_shots=None, rescaled=True):
        self.path = path
        self.transform = transform
        self.debug = debug
        self.manual_hflip = manual_hflip
        self.n_shots = n_shots
        self.rescaled = rescaled

        if self.debug:
            print("Starting to build dataset........")
            tic = time.time()

        if load_path is None:
            self.slices = {}
            self.H = 0
            self.W = 0
            df = pd.read_csv("/scratch/projects/sparkcognition/data/migration/ibalt/slices/ibaltcnvxhull_ns_so__nh401_nz1201_dh25_dz10/tbl_slices_outside_ibaltcntr_305from406.csv")
            slc_list = list(df.sid)

            slc_dirs = [f for f in slc_list if f not in ['so_2083', 'so_3647', 'so_587', 'so_317'] ] #slices that have caused issues

            #grb all the valid sids and set image dimensions
            for slc_dir in slc_dirs:
                temp = {} #this will hold the number of shots available to each slice
                files = os.listdir(os.path.join(self.path, slc_dir))
                kshots = [f for f in files if 'nshts' in f]

                if len(kshots) > 0:
                    for k in kshots:
                        temp[k] = len([f for f in os.listdir(os.path.join(self.path, slc_dir, k)) if '.npy' in f])
                    
                    self.slices[slc_dir] = temp

                if self.H == 0:
                    self.W, self.H = np.load(os.path.join(self.path, slc_dir, 'vel.npy'),allow_pickle=True).shape

            #[N, 1, H, W] set of filtered and pre-processed full-shot RTM images in [0, 1]
            #same order as self.slices - use this fact to index and match n_shots
            self.tensors = self.__build_dataset__()

        else:
            self.tensors, self.slices = self.load(load_path)
            self.H, self.W = self.tensors.shape[-2:]
        
        if self.debug:
            toc = time.time()
            print("TIME ELAPSED: ", str(toc - tic))
            print("Finished building dataset!")

    def __len__(self):
        return self.tensors.size(0)

    def __getitem__(self, index):
        #TODO we can make a lot of the lists created in this method class lists!
        #grab full-shot RTM image
        rtm_full_sample = self.tensors[index]
        if self.transform is not None:
            rtm_full_sample = self.transform(rtm_full_sample)
        
        #Find what values of k are available to the slice at the given index 
        slice_id = sorted(self.slices)[index] #so_xxxx
        slice_shots = self.slices[slice_id] #dict of {'nshtsk': num_realizations}

        #Randomly select a value of k with probability proportional to the number of realizations for each k
        choices = sorted(slice_shots) #sorted list of the 'nshtsk' for this slice
        p = np.array([slice_shots[k] for k in choices]) #MATCHING list of values for each 'nshtsk' for this slice
        p = p / np.sum(p) #now p is an empirical probability distribution over the 'nshtsk'
        k_str = np.random.choice(choices, p=p) #contains 'nshtsk' for some valid value of k for the given slice

        #get the corresponding index for the randomly-selected k from the given n_shots list
        k_num = int(k_str.strip('nshts')) #k
        k_idx = self.n_shots == k_num #tensor of shape self.n_shots.shape with True at matching indices
        try:
            k_idx = k_idx.nonzero().item() #finds the index of n_shots matching the chosen value of k
        except:
            print(slice_id)
            print(k_idx)
            k_idx = k_idx.nonzero().item()


        #get the k-shot image for the given slice
        rtm_k = self.__grab_k_shot_img__(sid=slice_id, n_shots=k_str) #has a filtered [1, H, W] image

        #perform a random horizontal flip, making sure both images have the same flip
        if self.manual_hflip:
            hflip = random.random() < 0.5
            if hflip:
                rtm_full_sample = do_hflip(rtm_full_sample)
                rtm_k = do_hflip(rtm_k)

        return rtm_full_sample, index, rtm_k, k_idx
    
    def get_samples(self, index=None, shot_idx=None, flip=False):
        if index is None:
            index = np.random.choice(self.tensors.size(0))
        if shot_idx is None:
            #Find what values of k are available to the slice at the given index 
            slice_id = sorted(self.slices)[index] #so_xxxx
            slice_shots = self.slices[slice_id] #dict of {'nshtsk': num_realizations}

            #Randomly select a value of k with probability proportional to the number of realizations for each k
            choices = sorted(slice_shots) #sorted list of the 'nshtsk' for this slice
            p = np.array([slice_shots[k] for k in choices]) #MATCHING list of values for each 'nshtsk' for this slice
            p = p / np.sum(p) #now p is an empirical probability distribution over the 'nshtsk'
            k_str = np.random.choice(choices, p=p) #contains 'nshtsk' for some valid value of k for the given slice

            #get the corresponding index for the randomly-selected k from the given n_shots list
            k_num = int(k_str.strip('nshts')) #k
            k_idx = self.n_shots == k_num #tensor of shape self.n_shots.shape with True at matching indices
            shot_idx = k_idx.nonzero().item() #finds the index of n_shots matching the chosen value of k
        else:
            slice_id = sorted(self.slices)[index] #so_xxxx
            k_str = 'nshts' + str(self.n_shots[shot_idx])

        #grab full-shot RTM image
        rtm_full_sample = self.tensors[index]
        if self.transform is not None:
            rtm_full_sample = self.transform(rtm_full_sample)
        
        #get the k-shot image for the given slice
        rtm_k = self.__grab_k_shot_img__(sid=slice_id, n_shots=k_str) #has a filtered [1, H, W] image

        #perform a random horizontal flip, making sure both images have the same flip
        if self.manual_hflip:
            hflip = random.random() < 0.5
            if hflip:
                rtm_full_sample = do_hflip(rtm_full_sample)
                rtm_k = do_hflip(rtm_k)

        return rtm_full_sample, index, rtm_k, shot_idx

    def __build_dataset__(self):
        """
        Gathers, compiles, pre-processes, and returns the Full-shot images
        """
        out_tensors = np.zeros(shape=(len(self.slices), 1, self.H, self.W), dtype=np.float32)

        ##### DEBUGGING #####
        min_vals = []
        max_vals = []
        ##### DEBUGGING #####

        #go through all viable slices and filter the full-shot images to store
        for idx, sid in enumerate(sorted(self.slices)):
            slice_path = os.path.join(self.path, sid, 'slice.npy') 
            vel_path = os.path.join(self.path, sid, 'vel.npy')

            img = np.load(slice_path,allow_pickle=True) #these will have shape [W, H]
            vel = np.load(vel_path,allow_pickle=True) / 1000 #convert from m/s to km/s

            filtered_img = filterImage(img, vel, 0.95, 0.03, N=1, useMask=True, laplace=False, rescale=self.rescaled)
            
            ##### DEBUGGING #####
            min_vals.append(np.min(filtered_img))
            max_vals.append(np.max(filtered_img))
            ##### DEBUGGING #####
            
            out_tensors[idx, 0] = filtered_img.T #transposed to have dims [H, W]

        ##### DEBUGGING #####
        mxval = np.mean(np.array(max_vals))
        mnval = np.mean(np.array(min_vals))
        print("Average min value of full-shot image: ", mnval)
        print("Average max value of full-shot image: ", mxval)
        ##### DEBUGGING #####


        return torch.from_numpy(out_tensors).float()
    
    def __grab_k_shot_img__(self, sid, n_shots):
        """
        Given a slice ID and number of shots k, returns a random k-shot RTM image
        """
        base_path = os.path.join(self.path, sid, n_shots) #the root of the k-shot realization directory for a given k

        candidates = os.listdir(base_path) #list of all the realizations of k-shot images for the given slice
        candidates = [k for k in candidates if '.npy' in k]
        cand_name = np.random.choice(candidates) #string name of a specific random realization

        img = np.load(os.path.join(base_path, cand_name),allow_pickle=True) #[W, H] unfiltered
        vel = np.load(os.path.join(self.path, sid, 'vel.npy'),allow_pickle=True)/1000 #from m/s -> km/s

        filtered_img = filterImage(img, vel, 0.95, 0.03, N=int(n_shots.strip('nshts')), useMask=True, laplace=False, rescale=self.rescaled)

        out_img = torch.from_numpy(filtered_img.T).float().unsqueeze(0) #[1, H, W]
        if self.transform is not None:
            out_img = self.transform(out_img)
        
        return out_img

    def save(self, path):
        """Function for saving the compiled full-shot tensor images and slice IDs.
            Useful for preparing data on a single process"""
        try:
            os.mkdir(path)
        except FileExistsError:
            pass

        torch.save(self.tensors, os.path.join(path, 'ibalt_full_shot_images.pt'))

        with open(os.path.join(path, 'ibalt_slices.pickle'), 'wb') as f:
            pickle.dump(self.slices, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    def load(self, path):
        out_tensors = torch.load(os.path.join(path, 'ibalt_full_shot_images.pt'))

        with open(os.path.join(path, 'ibalt_slices.pickle'), 'rb') as f:
            out_slices = pickle.load(f)
        
        if self.debug:
            print("Successfully loaded images and slice ids")

        return out_tensors, out_slices