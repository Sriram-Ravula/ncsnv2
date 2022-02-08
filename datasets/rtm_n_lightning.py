from torch.utils.data import TensorDataset, Subset, DataLoader
import numpy as np
import torch
import tqdm
import os
import time
import shutil
import random
import pickle

from pytorch_lightning import LightningDataModule
import torchvision.transforms as transforms

from rtm_utils import load_exp, load_npy, filterImage, laplaceFilter
from joblib import Parallel, delayed

from datasets.rtm_n import RTM_N

class RTM_Dataset(RTM_N):
    def __init__(self, path, transform=None, debug=True, load_path=None, n_shots=None, manual_hflip=True):
        super().__init__(path, transform, debug, load_path)

        self.n_shots = n_shots
        self.manual_hflip = manual_hflip
        if self.manual_hflip:
            self.flip_image = transforms.functional.hflip()

    def __getitem__(self, index):
        #(0) get the 243-shot RTM image
        rtm243_sample = self.tensors[index]

        if self.transform is not None:
            rtm243_sample = self.transform(rtm243_sample)
        
        #(1) if we just want the 243 image, we are done
        if self.n_shots is None:
            if self.manual_hflip:
                hflip = random.random() < 0.5
                if hflip:
                    rtm243_sample = self.flip_image(rtm243_sample)

            return rtm243_sample, index
        
        #(2) if we are doing RTM_n, then we also want to grab a random n-shot image
        else:
            shot_idx = np.random.randint(low=0, high=self.n_shots.numel(), size=1)[0]
            used_nshots = torch.tensor(self.n_shots[shot_idx])

            n_shot_sample = self.grab_rtm_image((rtm243_sample.unsqueeze(0), [index]), used_nshots).squeeze(0) #this is already transformed

            #perform a random horizontal flip, making sure both images have the same flip
            if self.manual_hflip:
                hflip = random.random() < 0.5
                if hflip:
                    rtm243_sample = self.flip_image(rtm243_sample)
                    n_shot_sample = self.flip_image(n_shot_sample)

            return rtm243_sample, index, n_shot_sample, shot_idx
    
    def save(self, path):
        """Function for saving the compiled 243-shot tensor images and slice IDs.
           Useful for preparing data on a single process"""
        try:
            os.mkdir(path)
        except FileExistsError:
            pass

        torch.save(self.tensors, os.path.join(path, '243_images.pt'))

        with open(os.path.join(path, 'slices.pickle'), 'wb') as f:
            pickle.dump(self.slices, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    def load(self, path):
        out_tensors = torch.load(os.path.join(path, '243_images.pt'))

        with open(os.path.join(path, 'slices.pickle'), 'rb') as f:
            out_slices = pickle.load(f)
        
        return out_tensors, out_slices

class RTMDataModule(LightningDataModule):
    def __init__(self, path, config):
        super().__init__()
        self.path = path
        self.config = config

        self.dataset_save_path = "tmp/rtm243data"

        n_shots = np.asarray(self.config.model.n_shots).squeeze()
        n_shots = torch.from_numpy(n_shots)
        if n_shots.numel() == 1:
            n_shots = torch.unsqueeze(n_shots, 0)
        self.n_shots = n_shots

        self.hflip = self.config.data.random_flip

        if self.config.data.dataset == 'RTM_N' or not self.hflip:
            self.transform = transforms.Compose([transforms.Resize(size = [self.config.data.image_size, self.config.data.image_size],
                                                interpolation=transforms.InterpolationMode.BICUBIC)])
        else:
            self.transform = transforms.Compose([transforms.Resize(size = [self.config.data.image_size, self.config.data.image_size],
                                    interpolation=transforms.InterpolationMode.BICUBIC),
                                    transforms.RandomHorizontalFlip(p=0.5)])
            self.hflip = False
        
    def prepare_data(self):
        """Prepares the RTM_243 image dataset on a single process"""
        dataset = RTM_Dataset(path=self.path) #just need the path for now since we are only using this to save
        dataset.save(self.dataset_save_path)

    def setup(self, stage):
        dataset = RTM_Dataset(path=self.path, transform=self.transform, debug=True, load_path=self.dataset_save_path,
                              n_shots=self.n_shots, manual_hflip=self.hflip)

        #calculate train/val splits
        num_items = len(dataset)
        indices = list(range(num_items))
        random_state = np.random.get_state()
        np.random.seed(2022)
        np.random.shuffle(indices)
        np.random.set_state(random_state)
        self.train_indices, self.val_indices = indices[:int(num_items * 0.95)], indices[int(num_items * 0.95):]

        self.train_dataset = Subset(dataset, self.train_indices)
        self.val_dataset = Subset(dataset, self.val_indices)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.config.training.batch_size, shuffle=True,
                                num_workers=self.config.data.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.config.training.batch_size, shuffle=True,
                                 num_workers=self.config.data.num_workers, drop_last=True)

    def test_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.config.training.batch_size, shuffle=True,
                                 num_workers=self.config.data.num_workers, drop_last=True)

    def predict_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.config.training.batch_size, shuffle=True,
                                 num_workers=self.config.data.num_workers, drop_last=True)
    
    def teardown(self, stage):
        try:
            shutil.rmtree(self.dataset_save_path)
        except:
            print('Could not clean-up dataset automatically.')