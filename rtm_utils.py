import os
import yaml
import numpy as np
import matplotlib.pyplot as plt
import torch

from os.path import join
from scipy.signal import butter, lfilter
from scipy.ndimage import correlate1d, generic_laplace, correlate


def get_ranges(nshots=301, l=10):
    ranges = []
    for j in range(nshots//l):
        en = (j+1)*l
        ranges.append((j*l, en))

    if en <= nshots:
        ranges.append((en, nshots))
    return ranges


def butter_bandpass(lowcut, highcut, nyq, order=5):
    low = lowcut / nyq
    high = highcut / nyq
    b,a = butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, f_nyq, order=5):
    b, a = butter_bandpass(lowcut, highcut, f_nyq, order=order)
    y = lfilter(b,a,data)
    return y

def d2(input, axis, output, mode, cval, weights):
    return correlate1d(input, weights, axis, output, mode, cval, 0)

def d22d(input, axis, output, mode, cval, weights):
    return correlate(input, weights, output, mode, cval, 0)

def laplaceFilter(im):
    return generic_laplace(im, d22d, mode='reflect', extra_keywords = {'weights' : [[1,1,1], [1, -8, 1], [1, 1, 1]]})

def clipFilter(im, qmax, qmin):
    flat = im.flatten()
    sorted_im = sorted(flat)
    max_idx = int(np.floor(qmax* (len(sorted_im))))
    min_idx = int(np.floor(qmin* (len(sorted_im))))
    im = np.clip(im, a_min=sorted_im[min_idx], a_max = sorted_im[max_idx - 1])
    return im

def clipFilterTorch(im, qmax, qmin):
    num_elements = im.numel()

    min_clip_num = int(num_elements * qmin) #number of elements below the qmin percentile 
    max_clip_num = int(num_elements * (1 - qmax)) #" " above the qmax percentile

    min_k = torch.max(torch.topk(im.flatten(), k=min_clip_num, largest=False)[0])
    max_k = torch.min(torch.topk(im.flatten(), k=max_clip_num, largest=True)[0])

    im = torch.clip(im, min=min_k, max=max_k)

    return im


def maskFilter(im, vel, threshold=1.65, return_mask = False):
    mask = np.ones(vel.shape)
    x,y = np.where(vel < threshold)
    mask[x,y] = 0
    if return_mask:
        return mask * im, mask
    else:
        return mask * im

def maskFilterTorch(im, vel, threshold=1.65, return_mask = False):
    mask = torch.ones_like(vel)
    mask[mask < threshold] = 0
    if return_mask:
        return mask * im, mask
    else:
        return mask * im

def normalizeFilter(im):
    return (im - np.min(im))/(np.max(im) - np.min(im))
    # return im/np.max(abs(im))

def normalizeFilterTorch(im):
    return (im - torch.min(im))/(torch.max(im) - torch.min(im))

def filterImage1(data, vel, vmax, vmin):
    return filterImage(data, vel, vmax, vmin)

def filterImage_valueClip(data, vmax, vmin):
    data = laplaceFilter(data)
    data = np.clip(data, a_min=vmin, a_max=vmax)
    data = maskFilter(data, vel)
    data = normalizeFilter(data)
    return data

def filterImage(data, vel, qmax, qmin, N=243, useMask=True, rescale=True, verbose=False, laplace=True):
    data = data/N
    values = []
    values.append([np.max(data), np.min(data), np.max(data) - np.min(data)])
    if laplace:
        data = laplaceFilter(data)
        values.append([np.max(data), np.min(data), np.max(data) - np.min(data)])
    data = clipFilter(data, qmax=qmax, qmin=qmin)
    values.append([np.max(data), np.min(data), np.max(data) - np.min(data)])
    if useMask:
        data = maskFilter(data, vel)
    values.append([np.max(data), np.min(data), np.max(data) - np.min(data)])
    if rescale:
        data = normalizeFilter(data)
        values.append([np.max(data), np.min(data), np.max(data) - np.min(data)])
    
    
    if verbose:
        return data, values
    else:
        return data

## old method, but still used in some notebooks
def quantile(data, vel, nbl, qmax=1.0, qmin=0, useMask= True, plot=True, cmap='nipy_spectral', title='quantile image'):
    data = data[nbl:-nbl, nbl:-nbl]
    if useMask:
        finalIm = normalizeFilter(
            maskFilter(
                clipFilter(
                    laplaceFilter(
                        data
                    ), qmax=qmax, qmin=qmin
                ), vel
            )
        )
    else:
        finalIm = normalizeFilter(
            clipFilter(
                laplaceFilter(
                    data
                ), qmax=qmax, qmin=qmin
            )
        )
    if plot:
        plt.figure(figsize=(12,9))
        plt.title(title)
        plt.imshow(finalIm, cmap=cmap)
    return finalIm


def korina_filter(data, vel, qmax=1.0, qmin=0.0, useMask=True, nbl=40, cmap='nipy_spectral', title='gradient filtered image', kernel_shape=(3,3), plot=True):
    import torch
    from kornia.morphology import gradient
    torch_data = torch.tensor(data).reshape((-1, 1, data.shape[0], data.shape[1]))
    kernel = torch.ones(kernel_shape)
    gradient_filter = gradient(torch_data, kernel)
    filtered = data - gradient_filter.reshape((1, int(data.shape[0]), int(data.shape[1]), 1)).numpy().squeeze()

    mask = np.ones(data.shape)
    x,y = np.where(vel < 1.5)
    nbl_x = x + nbl
    nbl_y = y 
    mask[nbl_x, nbl_y] = 0

    flat = filtered.flatten()
    sorted_im = sorted(flat)
    max_idx = int(np.floor(qmax* (len(sorted_im))))
    min_idx = int(np.floor(qmin* (len(sorted_im))))
    q_im = np.clip(filtered, a_min=sorted_im[min_idx], a_max = sorted_im[max_idx - 1])
    if useMask:
        q_im = q_im * mask
    if plot:
        plt.figure(figsize=(18,12))
        plt.imshow(q_im.T, cmap=cmap)
        plt.colorbar()
    return q_im

def load_exp(path, fldrs=['shots', 'traces']):
    if os.path.isdir(join(path, 'seam2d-0')):
        path = join(path, 'seam2d-0')
    out_dict = {}

    for fld in fldrs:
        tmp = [join(path, fld, f) for f in os.listdir(join(path, fld)) if 'id' in f]
        out_dict[fld] = tmp
    
    with open(join(path, 'config.yaml'), 'rb') as f:
        out_dict['config'] = yaml.load(f, Loader=yaml.FullLoader)
    with open(join(path, 'slice.npy'), 'rb') as f:
        out_dict['vel'] = np.load(f)
    with open(join(path, 'image.npy'), 'rb') as f:
        image = np.load(f)
    nbl = out_dict['config']['rtm_cnfg']['nbl']
    out_dict['image'] = image[nbl:-nbl, nbl:-nbl]
    return out_dict
    
def load_npy(path):
    with open(path, 'rb') as f:
        t = np.load(f)
    return t



    
        