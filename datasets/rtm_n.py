from torch.utils.data import TensorDataset
import numpy as np
import torch
import tqdm
import os

from rtm_utils import load_exp, load_npy, filterImage, laplaceFilter

class RTM_N(TensorDataset):
    """
    A dataset class for RTM_n images.
    Looks at a folder of RTM images and:
        - enumerates the list of their indices
        - it preprocesses returns the RTM_243 images (already hardcoded and saved) 
        - 
        - preprocesses the images to stitch together "n-shots" when loaded at train time
    """
    def __init__(self, path, transform=None, debug=True):
        self.debug = debug
        self.path = path
        self.transform = transform

        self.slices = []
        self.H = 0
        self.W = 0

        #grb all the valid sids and set image dimensions
        for dI in os.listdir(path):

            slice_path = os.path.join(path, dI)

            if os.path.isdir(slice_path):

                img_path = os.path.join(slice_path, 'image.npy')
                shots_path = os.path.join(slice_path, 'shots')
                config_path = os.path.join(slice_path, 'config.yaml')
                velocity_path = os.path.join(slice_path, 'slice.npy')

                if os.path.isfile(img_path) and os.path.isdir(shots_path) and os.path.isfile(config_path) and os.path.isfile(velocity_path):

                    self.slices.append(dI)

                    if self.H == 0:

                        self.W, self.H = load_exp(slice_path)['vel'].shape #shape is transposed of how it should be viewed

        #TODO: Remove this debugging line
        self.slices = self.slices[0:40]

        #[N, 1, H, W] set of filtered and pre-processed RTM243 images in [0, 1]
        #same order as self.slices - use this fact to index and match n_shots
        self.tensors = self.__build_dataset__()

    def __len__(self):
        return self.tensors.size(0)

    def __getitem__(self, index):
        sample = self.tensors[index]

        if self.transform:
            sample = self.transform(sample)

        return sample, index

    def __build_dataset__(self):
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
    
    def grab_rtm_image(self, input_sample, n_shots):
        """
        Given an image index and number of shots per image, gather and preprocess an RTM_n image.
        Args:
            input_sample: (X, y) pair of rtm_243 image and its index. (tensor:[N, 1, H, W], list:[N])
            n_shots: the number of shots we want to select for each rtm_n image corresponding to the input. list:[N]
        Returns:
            rtm_n_img: a tensor with same dimensions as X comprising rtm_n images. tensor:[N, 1, H, W]
        """
        image_orig, image_index = input_sample #the RTM_243 image and its index

        rtm_n_img = torch.zeros_like(image_orig) #holds the images to be returned

        #artifact caused by "n_shots.squeeze().tolist()" producing a float if n_shots has a single element
        if torch.numel(n_shots) == 1:
            n_shots_iter = [n_shots.item()]
        else:
            n_shots_iter = n_shots.squeeze().tolist()

        for i, n in enumerate(n_shots_iter):
            idxs = np.random.choice(243, size=int(n), replace=False) #random indices of n_shots to grab

            exp = load_exp(os.path.join(self.path, self.slices[image_index[i]])) #dictionary of slice information

            shot_paths = [s for s in exp['shots'] if int(os.path.basename(s).split("-")[1][:-4]) in idxs] #individual shots

            image_k = np.zeros((exp['vel'].shape)) #holds the final processed image

            #filter and sum each shot image
            for spth in shot_paths:
                image_k = image_k + laplaceFilter(load_npy(spth)[20:-20, 20:-20])

            new_x = filterImage(image_k, exp['vel'], 0.95, 0.03, N=n, rescale=True, laplace=False).T
        
            rtm_n_img[i] = self.transform(torch.from_numpy(new_x).unsqueeze(0).to(image_orig.device).type(image_orig.dtype))

        return rtm_n_img
