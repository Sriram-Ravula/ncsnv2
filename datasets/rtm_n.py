from torch.utils.data import TensorDataset
import numpy as np
import torch
import tqdm
import os
import time
import shutil

from rtm_utils import load_exp, load_npy, filterImage, laplaceFilter
from joblib import Parallel, delayed

class RTM_N(TensorDataset):
    """
    A dataset class for RTM_n images.
    Looks at a folder of RTM images and:
        - enumerates the list of their indices
        - it preprocesses stores the RTM_243 images (already hardcoded and saved) 
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
        #self.slices = self.slices[0:1000]

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

        folder = '/tmp/joblib_memmap' #temporary location to store results from parallel workers!
        try:
            os.mkdir(folder)
        except FileExistsError:
            pass

        output_filename_memmap = os.path.join(folder, 'output_memmap')

        image_dataset = np.memmap(output_filename_memmap, dtype=np.float32, 
                   shape=(len(self.slices), 1, self.H, self.W), mode='w+')

        def grab_and_process_imgs(i, s, output):
            """
            Calculate the RTM-243 image for the given subslice ID and sotre it in a given index of an output array.
            Modifies the contents of output.

            Args:
                i: index in the output argument to store the result
                s: subslice id 
                output: tensor or array which will hold the results. Contents are modified!
            """
            rtm243_path = os.path.join(self.path, s)
            exp = load_exp(rtm243_path) #dictionary with the current subslice contents

            #filter and pre-process the rtm243 image 
            rtm243_img = filterImage(exp['image'], exp['vel'], 0.95, 0.03, N=1, rescale=True,laplace=True).T

            output[i] = np.expand_dims(rtm243_img, axis=0) #expand to [1, H, W]

        if self.debug:
            print("Starting to build dataset........")
            tic = time.time()

        Parallel(n_jobs=-1)(delayed(grab_and_process_imgs)(idx, sid, image_dataset) for idx, sid in enumerate(self.slices))

        if self.debug:
            toc = time.time()
            print("TIME ELAPSED: ", str(toc - tic))
            print("Finished building dataset!")

        try:
            shutil.rmtree(folder)
        except:  # noqa
            print('Could not clean-up automatically.')
        
        return torch.from_numpy(image_dataset).float()
    
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

        folder = '/tmp/joblib_memmap' #temporary location to store results from parallel workers!
        try:
            os.mkdir(folder)
        except FileExistsError:
            pass

        output_filename_memmap = os.path.join(folder, 'output_memmap')

        rtm_n_img = np.memmap(output_filename_memmap, dtype=np.float32, 
                   shape=(image_orig.shape[0], 1, self.H, self.W), mode='w+')

        def get_single_rtm_img(i, n, path, slices, img_idx, output):
            """
            Edits rtm_n_img[i] to hold an rtm_n image.
            Args:
                i: the index in the batch of the rtm_n image we are creating
                n: the number of random shots to use to create the rtm_n image
                path: the root folder containing all subslices
                slices: list of names of all the subslices
                img_idx: the indeces of the given image in the batch image within the subslice list
                output: the final array where we store the rtm_n images
            """
            idxs = np.random.choice(243, size=int(n), replace=False) #random indices of n_shots to grab

            exp = load_exp(os.path.join(path, slices[img_idx[i]])) #dictionary of slice information

            shot_paths = [s for s in exp['shots'] if int(os.path.basename(s).split("-")[1][:-4]) in idxs] #individual shots

            image_k = np.zeros((exp['vel'].shape)) #holds the final processed image

            #filter and sum each shot image
            for spth in shot_paths:
                image_k = image_k + laplaceFilter(load_npy(spth)[20:-20, 20:-20])

            new_x = filterImage(image_k, exp['vel'], 0.95, 0.03, N=n, rescale=True, laplace=False).T

            output[i] = np.expand_dims(new_x, axis=0)

        if torch.numel(n_shots) == 1:
            get_single_rtm_img(0, n_shots.item(), self.path, self.slices, image_index, rtm_n_img) 

        else:
            Parallel(n_jobs=-1)(delayed(get_single_rtm_img)(i, n, self.path, self.slices, image_index, rtm_n_img) \
                for i, n in enumerate(n_shots.squeeze().tolist()))

        #convert the results to torch with all the correct attributes and transforms
        rtm_n_img = torch.from_numpy(rtm_n_img).to(image_orig.device).type(image_orig.dtype)
        rtm_n_img = self.transform(rtm_n_img)

        try:
            shutil.rmtree(folder)
        except:  # noqa
            print('Could not clean-up automatically.')


        return rtm_n_img
