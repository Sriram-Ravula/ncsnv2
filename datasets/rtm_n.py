from torch.utils.data import TensorDataset
import numpy as np
import torch
import tqdm
import os

from seismic_migration.utils.tools import load_exp, load_npy, filterImage
from seismic_migration.utils.custom_plots import longs_colormap
from mpl_toolkits.axes_grid1 import make_axes_locatable

"""
A dataset class for RTM_n images.
Looks at a folder of RTM images and:
    - enumerates the list of their indices
    - it preprocesses returns the RTM_243 images (already hardcoded and saved) 
    - 
    - preprocesses the images to stitch together "n-shots" when loaded at train time
"""
class RTM_N(TensorDataset):
    def __init__(self, path, transform=None, debug=True):
        self.debug = debug
        self.path = path
        self.transform = transform

        self.slices = []
        self.H = 0
        self.W = 0

        #grb all the valid sids and set image dimensions
        for dI in os.listdir(path):

            slice_path = os.path.join(path,dI)

            if os.path.isdir(slice_path):

                img_path = os.path.join(slice_path, 'image.npy')
                shots_path = os.path.join(slice_path, 'shots')

                if os.path.isdir(img_path) and os.path.isdir(shots_path):

                    self.slices.append(dI)

                    if self.H == 0:

                        self.H, self.W = load_exp(img_path)['vel'].shape

        #[N, 1, H, W] set of filtered and pre-processed RTM243 images in [0, 1]
        #same order as self.slices - use this fact to index and match n_shots
        self.tensors = self.__build_dataset__()

    def __len__(self):
        return self.tensors.size(0)

    def __getitem__(self, index):
        sample = self.tensors[index]

        if self.transform:
            sample = self.transform(sample)

        target = 0 #dummy target

        return sample, index

    def __build_dataset__():
    """
    Gathers, compiles, pre-processes, and returns the RTM_243 images
    """
        if self.debug:
            print("Starting to build dataset........")
        
        image_dataset = torch.zeros(len(self.slices), 1, self.H, self.W)
        
        for idx, sid in tqdm.tqdm(enumerate(self.slices)):
            rtm243_path = os.path.join(self.path, sid)
            exp = load_exp(rtm243_path) #dictionary with the current subslice contents

            #filter and pre-process the rtm243 image 
            rtm243_img = filterImage(exp['image'], exp['vel'], 0.95, 0.03, N=1, rescale=True,laplace=True).T

            image_dataset[idx] = torch.from_numpy(rtm243_img).unsqueeze(0).float()

        if self.debug:
            print("Finished building dataset!")
        
        return image_dataset
    
    def grab_rtm_image(image_index, n_shots):
        
