import os
import torch
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10, LSUN
from datasets.celeba import CelebA
from datasets.ffhq import FFHQ
from datasets.velocity_fine import Velocity
from datasets.rtm_n import RTM_N
from datasets.ibalt import Ibalt
from torch.utils.data import Subset
import numpy as np

def get_dataset(args, config):
    if config.data.random_flip is False:
        tran_transform = test_transform = transforms.Compose([
            transforms.Resize(config.data.image_size),
            transforms.ToTensor()
        ])
    else:
        tran_transform = transforms.Compose([
            transforms.Resize(config.data.image_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor()
        ])
        test_transform = transforms.Compose([
            transforms.Resize(config.data.image_size),
            transforms.ToTensor()
        ])

    if config.data.dataset == 'CIFAR10':
        dataset = CIFAR10(os.path.join(args.exp, 'datasets', 'cifar10'), train=True, download=True,
                          transform=tran_transform)
        test_dataset = CIFAR10(os.path.join(args.exp, 'datasets', 'cifar10_test'), train=False, download=True,
                               transform=test_transform)

    elif config.data.dataset == 'CELEBA':
        if config.data.random_flip:
            dataset = CelebA(root=os.path.join(args.exp, 'datasets', 'celeba'), split='train',
                             transform=transforms.Compose([
                                 transforms.CenterCrop(140),
                                 transforms.Resize(config.data.image_size),
                                 transforms.RandomHorizontalFlip(),
                                 transforms.ToTensor(),
                             ]), download=True)
        else:
            dataset = CelebA(root=os.path.join(args.exp, 'datasets', 'celeba'), split='train',
                             transform=transforms.Compose([
                                 transforms.CenterCrop(140),
                                 transforms.Resize(config.data.image_size),
                                 transforms.ToTensor(),
                             ]), download=True)

        test_dataset = CelebA(root=os.path.join(args.exp, 'datasets', 'celeba_test'), split='test',
                              transform=transforms.Compose([
                                  transforms.CenterCrop(140),
                                  transforms.Resize(config.data.image_size),
                                  transforms.ToTensor(),
                              ]), download=True)


    elif config.data.dataset == 'LSUN':
        train_folder = '{}_train'.format(config.data.category)
        val_folder = '{}_val'.format(config.data.category)
        if config.data.random_flip:
            dataset = LSUN(root=os.path.join(args.exp, 'datasets', 'lsun'), classes=[train_folder],
                             transform=transforms.Compose([
                                 transforms.Resize(config.data.image_size),
                                 transforms.CenterCrop(config.data.image_size),
                                 transforms.RandomHorizontalFlip(p=0.5),
                                 transforms.ToTensor(),
                             ]))
        else:
            dataset = LSUN(root=os.path.join(args.exp, 'datasets', 'lsun'), classes=[train_folder],
                             transform=transforms.Compose([
                                 transforms.Resize(config.data.image_size),
                                 transforms.CenterCrop(config.data.image_size),
                                 transforms.ToTensor(),
                             ]))

        test_dataset = LSUN(root=os.path.join(args.exp, 'datasets', 'lsun'), classes=[val_folder],
                             transform=transforms.Compose([
                                 transforms.Resize(config.data.image_size),
                                 transforms.CenterCrop(config.data.image_size),
                                 transforms.ToTensor(),
                             ]))

    elif config.data.dataset == "FFHQ":
        if config.data.random_flip:
            dataset = FFHQ(path=os.path.join(args.exp, 'datasets', 'FFHQ'), transform=transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor()
            ]), resolution=config.data.image_size)
        else:
            dataset = FFHQ(path=os.path.join(args.exp, 'datasets', 'FFHQ'), transform=transforms.ToTensor(),
                           resolution=config.data.image_size)

        num_items = len(dataset)
        indices = list(range(num_items))
        random_state = np.random.get_state()
        np.random.seed(2019)
        np.random.shuffle(indices)
        np.random.set_state(random_state)
        train_indices, test_indices = indices[:int(num_items * 0.9)], indices[int(num_items * 0.9):]
        test_dataset = Subset(dataset, test_indices)
        dataset = Subset(dataset, train_indices)
    
    elif config.data.dataset == 'VELOCITY_FINE':
        tran_transform = transforms.Compose([
            transforms.Resize(size = [config.data.image_size, config.data.image_size], \
                interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(p=0.5)
        ])

        dataset = Velocity(path=os.path.join(args.exp, 'datasets', '8047_vel_imgs.npy'), transform=tran_transform)

        num_items = len(dataset)
        indices = list(range(num_items))
        random_state = np.random.get_state()
        np.random.seed(2240)
        np.random.shuffle(indices)
        np.random.set_state(random_state)
        train_indices, test_indices = indices[:int(num_items * 0.9)], indices[int(num_items * 0.9):]
        test_dataset = Subset(dataset, test_indices)
        dataset = Subset(dataset, train_indices)
    
    elif config.data.dataset == 'VELOCITY_DIFFERENCE':
        tran_transform = transforms.Compose([
            transforms.Resize(size = [config.data.image_size, config.data.image_size], \
                interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(p=0.5)
        ])

        dataset = Velocity(path=os.path.join(args.exp, 'datasets', '8047_dy_vel_imgs.npy'), transform=tran_transform)

        num_items = len(dataset)
        indices = list(range(num_items))
        random_state = np.random.get_state()
        np.random.seed(2240)
        np.random.shuffle(indices)
        np.random.set_state(random_state)
        train_indices, test_indices = indices[:int(num_items * 0.9)], indices[int(num_items * 0.9):]
        test_dataset = Subset(dataset, test_indices)
        dataset = Subset(dataset, train_indices)

    elif config.data.dataset == 'VELOCITY_RTM':
        tran_transform = transforms.Compose([
            transforms.Resize(size = [config.data.image_size, config.data.image_size], \
                interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(p=0.5)
        ])

        dataset = Velocity(path="/scratch/04703/sravula/experiments/datasets/rtm_n/243_images.pt", transform=tran_transform)

        num_items = len(dataset)
        indices = list(range(num_items))
        random_state = np.random.get_state()
        np.random.seed(2022)
        np.random.shuffle(indices)
        np.random.set_state(random_state)

        train_indices, test_indices = indices[:int(num_items * 0.975)], indices[int(num_items * 0.975):]
        test_dataset = Subset(dataset, test_indices)
        dataset = Subset(dataset, train_indices)
    
    elif config.data.dataset == 'RTM_N':
        tran_transform = transforms.Compose([
            transforms.Resize(size = [config.data.image_size, config.data.image_size], \
                interpolation=transforms.InterpolationMode.BICUBIC),    #no horizontal flip - would affect RTM_n image!
        ])

        n_shots = np.asarray(config.model.n_shots).squeeze()
        n_shots = torch.from_numpy(n_shots)
        #make sure it has dimension > 0 if it is a singleton (useful for indexing)
        if n_shots.numel() == 1:
            n_shots = torch.unsqueeze(n_shots, 0)

        dataset = RTM_N(path="/scratch/08269/rstone/full_rtm_8048", transform=tran_transform, \
                        load_path="/scratch/04703/sravula/experiments/datasets/rtm_n", manual_hflip=config.data.random_flip,\
                        n_shots=n_shots)

        num_items = len(dataset)
        indices = list(range(num_items))
        random_state = np.random.get_state()
        np.random.seed(2022)
        np.random.shuffle(indices)
        np.random.set_state(random_state)

        train_indices, test_indices = indices[:int(num_items * 0.975)], indices[int(num_items * 0.975):]
        test_dataset = Subset(dataset, test_indices)
        dataset = Subset(dataset, train_indices)
    
    elif config.data.dataset == 'IBALT_RTM_N':
        tran_transform = transforms.Compose([
            transforms.Resize(size = [config.data.image_size, config.data.image_size], \
                interpolation=transforms.InterpolationMode.BICUBIC),    #no horizontal flip - would affect RTM_n image!
        ])

        n_shots = np.asarray(config.model.n_shots).squeeze()
        n_shots = torch.from_numpy(n_shots)
        #make sure it has dimension > 0 if it is a singleton (useful for indexing)
        if n_shots.numel() == 1:
            n_shots = torch.unsqueeze(n_shots, 0)

        dataset = Ibalt(path='/scratch/08087/gandhiy/data/migration/ibalt/slices/ibaltcnvxhull_ns_so__nh401_nz1201_dh25_dz10', \
                        transform=tran_transform, manual_hflip=config.data.random_flip, n_shots=n_shots)

        num_items = len(dataset)
        indices = list(range(num_items))
        random_state = np.random.get_state()
        np.random.seed(2022)
        np.random.shuffle(indices)
        np.random.set_state(random_state)

        idx = max(1, int(num_items * 0.975))
        train_indices, test_indices = indices[:idx], indices[idx:]
        test_dataset = Subset(dataset, test_indices)
        dataset = Subset(dataset, train_indices)

    return dataset, test_dataset

def logit_transform(image, lam=1e-6):
    image = lam + (1 - 2 * lam) * image
    return torch.log(image) - torch.log1p(-image)

def data_transform(config, X):
    if config.data.uniform_dequantization:
        X = X / 256. * 255. + torch.rand_like(X) / 256.
    if config.data.gaussian_dequantization:
        X = X + torch.randn_like(X) * 0.01

    if config.data.rescaled:
        X = 2 * X - 1.
    elif config.data.logit_transform:
        X = logit_transform(X)

    if hasattr(config, 'image_mean'):
        return X - config.image_mean.to(X.device)[None, ...]

    return X

def inverse_data_transform(config, X):
    if hasattr(config, 'image_mean'):
        X = X + config.image_mean.to(X.device)[None, ...]

    if config.data.logit_transform:
        X = torch.sigmoid(X)
    elif config.data.rescaled:
        X = (X + 1.) / 2.

    return torch.clamp(X, 0.0, 1.0)
