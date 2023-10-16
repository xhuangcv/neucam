import csv
from curses import endwin
import glob
import math
import os

import matplotlib.colors as colors
import numpy as np
import scipy.io.wavfile as wavfile
import scipy.ndimage
import scipy.special
import skimage
import skimage.filters
import skvideo.io
import torch
import imageio
import cv2
import utils
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import Resize, Compose, ToTensor, Normalize, RandomCrop


def get_mgrid(sidelen, dim=2):
    '''Generates a flattened grid of (x,y,...) coordinates in a range of -1 to 1.'''
    if isinstance(sidelen, int):
        sidelen = dim * (sidelen,)

    if dim == 2:
        pixel_coords = np.stack(np.mgrid[:sidelen[0], :sidelen[1]], axis=-1)[None, ...].astype(np.float64)
        pixel_coords[0, :, :, 0] = pixel_coords[0, :, :, 0] / (sidelen[0] - 1)
        pixel_coords[0, :, :, 1] = pixel_coords[0, :, :, 1] / (sidelen[1] - 1)
    elif dim == 3:
        pixel_coords = np.stack(np.mgrid[:sidelen[0], :sidelen[1], :sidelen[2]], axis=-1)[None, ...].astype(np.float64)
        pixel_coords[..., 0] = pixel_coords[..., 0] / max(sidelen[0] - 1, 1)
        pixel_coords[..., 1] = pixel_coords[..., 1] / (sidelen[1] - 1)
        pixel_coords[..., 2] = pixel_coords[..., 2] / (sidelen[2] - 1)
    else:
        raise NotImplementedError('Not implemented for dim=%d' % dim)

    pixel_coords -= 0.5
    pixel_coords *= 2.
    pixel_coords = torch.Tensor(pixel_coords).squeeze(0)

    return pixel_coords


def rescale_img(x, mode='scale', perc=None, tmax=1.0, tmin=0.0):
    if (mode == 'scale'):
        if perc is None:
            xmax = torch.max(x)
            xmin = torch.min(x)
        else:
            xmin = np.percentile(x.detach().cpu().numpy(), perc)
            xmax = np.percentile(x.detach().cpu().numpy(), 100 - perc)
            x = torch.clamp(x, xmin, xmax)
        if xmin == xmax:
            return 0.5 * torch.ones_like(x) * (tmax - tmin) + tmin
        x = ((x - xmin) / (xmax - xmin)) * (tmax - tmin) + tmin
    elif (mode == 'clamp'):
        x = torch.clamp(x, 0, 1)
    return x


def to_uint8(x):
    return (255. * x).astype(np.uint8)


def to_numpy(x):
    return x.detach().cpu().numpy()


def gaussian(x, mu=[0, 0], sigma=1e-4, d=2):
    x = x.numpy()
    if isinstance(mu, torch.Tensor):
        mu = mu.numpy()

    q = -0.5 * ((x - mu) ** 2).sum(1)
    return torch.from_numpy(1 / np.sqrt(sigma ** d * (2 * np.pi) ** d) * np.exp(q / sigma)).float()


class Video(Dataset):
    def __init__(self, path_to_video, seq_size, resize=True, not_load_exps=False, 
                 load_mask=False, load_gt=False, load_flow=False, load_weights=-1, load_depth=False, load_gamma=False):
        super().__init__()

        max_frame, H, W= seq_size

        if 'npy' in path_to_video:
            self.vid = np.load(path_to_video)
        elif 'mp4' in path_to_video:
            frame_list = []
            reader = imageio.get_reader(path_to_video)
            for _, im in enumerate(reader):
                frame_resized = cv2.resize(im, (W, H), interpolation=cv2.INTER_AREA)
                frame_list.append((frame_resized / 255.).astype(np.float32))
            self.vid = np.array(frame_list)
        else:
            
            file_names = [f for f in sorted(os.listdir(path_to_video)) 
                         if any([f.endswith(ex) for ex in ['JPG', 'jpg', 'png', 'jpeg', 'PNG', 'tif', 'tiff']])]
            file_paths = [os.path.join(path_to_video, f) for f in file_names]

            N = min([len(file_paths), max_frame])
            first_img = imageio.imread(file_paths[0])
            resize = (H != first_img.shape[0] or W != first_img.shape[1])

            # Load gt images
            if load_gt:
                all_frames = []
                for i,p in enumerate(file_paths[:N]):
                    img = imageio.imread(p)
                    if resize:
                        img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
                    all_frames.append(img)
                if file_names[0].endswith('tif') or file_names[0].endswith('tiff'):
                    all_frames = np.stack(all_frames, 0) / (2**16-1)
                else:
                    all_frames = np.stack(all_frames, 0) / 255
                self.vid = all_frames.astype(np.float32)
            else:
                self.vid = np.zeros([max_frame, H, W, 3])

            # Load optical flow
            self.flow = None
            # if load_flow:
            #     self.flow = np.zeros([N, H, W, 6])
            #     out_flow_dir = os.path.join(path_to_video, 'flow')
            #     for i  in range(N-1):
            #         j = i + 1
            #         fn1 = file_names[i]
            #         fn2 = file_names[j]

            #         flow12_fn = os.path.join(out_flow_dir, f'{fn1}_{fn2}.npy')
            #         flow21_fn = os.path.join(out_flow_dir, f'{fn2}_{fn1}.npy')
            #         flow12 = np.load(flow12_fn)
            #         flow21 = np.load(flow21_fn)

            #         if resize:
            #             flow12 = utils.resize_flow(flow12, newh=H, neww=W)
            #             flow21 = utils.resize_flow(flow21, newh=H, neww=W)
            #         mask_flow = utils.compute_consistency(flow12, flow21) < 1.
            #         mask_flow_reverse = utils.compute_consistency(flow21, flow12) < 1.

            #         self.flow[i, :, :, 0:2] = flow12
            #         self.flow[i, :, :, 2] = mask_flow
            #         self.flow[j, :, :, 3:5] = flow21
            #         self.flow[j, :, :, 5] = mask_flow_reverse
            #     self.flow = self.flow.astype(np.float32)
            
            # Load initial depth
            self.depth = None
            if load_depth:
                depth_names = [f for f in sorted(os.listdir(os.path.join(path_to_video, 'depth'))) 
                         if any([f.endswith(ex) for ex in ['JPG', 'jpg', 'png', 'jpeg', 'PNG']])]
                depth_paths = [os.path.join(path_to_video, 'depth', f) for f in depth_names]

                all_depth = []
                for i,p in enumerate(depth_paths[:N]):
                    depth = imageio.imread(p)
                    if resize:
                        depth = cv2.resize(depth, (W, H), interpolation=cv2.INTER_AREA)
                    all_depth.append(depth)
                all_depth = np.stack(all_depth, 0).astype(np.float32)
                all_depth_normalized = (all_depth - np.min(all_depth)) / (np.max(all_depth) - np.min(all_depth))
                self.depth = all_depth_normalized[..., np.newaxis]
                if self.depth.shape[0] == 1:
                    self.depth = np.tile(self.depth, (self.vid.shape[0], 1, 1, 1))

            # Load initial mask
            self.mask = None
            if load_mask:
                mask_names = [f for f in sorted(os.listdir(os.path.join(path_to_video, 'mask'))) 
                         if any([f.endswith(ex) for ex in ['JPG', 'jpg', 'png', 'jpeg', 'PNG']])]
                mask_paths = [os.path.join(path_to_video, 'mask', f) for f in mask_names]

                all_mask = []
                for i,p in enumerate(mask_paths[:N]):
                    mask = imageio.imread(p)
                    if resize:
                        mask = cv2.resize(mask, (W, H), interpolation=cv2.INTER_AREA)
                    all_mask.append(mask)
                all_mask = np.stack(all_mask, 0) / 255.
                self.mask = all_mask[..., np.newaxis].astype(np.float32)

            # Load refine weights
            self.weights = None
            if load_weights != -1:
                self.weights = np.concatenate([np.zeros([1,H,W,1]) + 0.2, np.ones([1,H,W,1]), np.zeros([1,H,W,1]) + 0.2], 0)
                self.weights = self.weights.astype(np.float32)



            # Load exposures
            self.exps = None
            # not_load_exps = False
            if not not_load_exps:
                with open(os.path.join(path_to_video, 'exposure.txt'), "r") as f:
                    candidate_exps = [float(t) for t in f.read().splitlines()]
                all_exps = []
                for i,p in enumerate(file_paths[:N]):
                    all_exps.append(candidate_exps[i % len(candidate_exps)])
                all_exps = np.tile(np.stack(all_exps, 0).reshape(-1, 1, 1, 1), [1, H, W, 1])
                self.exps = all_exps.astype(np.float32)
            
            # Load gamma weights
            self.gamma_weights = None
            if load_gamma:
                # low_exp_idx = np.argmin(self.exps[:,0,0,0])
                # high_exp_idx = np.argmax(self.exps[:,0,0,0])
                self.gamma_weights = np.ones([N,H,W,1]).astype(np.float32)
                # self.gamma_weights[low_exp_idx, ...] += 1
                # self.gamma_weights[1, ...] += 0.5
                # self.gamma_weights[high_exp_idx, ...] += 0.2
                
        self.num_frame = N 
        self.shape = self.vid.shape[:-1]
        self.channels = self.vid.shape[-1]

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        return self.vid


class ImageFile(Dataset):
    def __init__(self, filename):
        super().__init__()
        self.img = Image.open(filename)
        self.img_channels = len(self.img.mode)

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        return self.img


class Implicit3DWrapperPatch(torch.utils.data.Dataset):
    def __init__(self, dataset, sidelength=None, sample_fraction=1., is_training=True):

        self.dataset = dataset.vid[..., :3]
        self.flow = None
        self.exps = None
        self.mask = None
        self.weights = None
        self.gamma_weights = None
        self.depth = None
        self.is_training = is_training
        if dataset.flow is not None:
            self.flow = torch.from_numpy(dataset.flow)
        if dataset.exps is not None:
            self.exps = torch.from_numpy(dataset.exps)
        if dataset.mask is not None:
            self.mask = torch.from_numpy(dataset.mask)
        if dataset.weights is not None:
            self.weights = torch.from_numpy(dataset.weights)
        if dataset.depth is not None and (dataset.depth.shape[0] == dataset.vid.shape[0]):
            self.depth = ((torch.from_numpy(dataset.depth)- 0.5) / 0.5)
            
        self.src_mgrid = torch.from_numpy(np.stack(np.mgrid[:dataset.shape[1]-29, :dataset.shape[2]-29], axis=-1)[None, ...].astype(np.int)).view(-1,2)
        self.mgrid = get_mgrid(sidelength, dim=3)
        self.data = ((torch.from_numpy(self.dataset) - 0.5) / 0.5)
        self.sample_fraction = sample_fraction
        self.N_samples = int(self.sample_fraction * self.src_mgrid.shape[0])


    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):

        if self.sample_fraction < 1.:
            coord_idx = torch.randint(0, self.src_mgrid.shape[0], (self.N_samples,))
        else:
            coord_idx = torch.arange(self.src_mgrid.shape[0])
   
        coord_y = self.src_mgrid[coord_idx, 0]
        coord_x = self.src_mgrid[coord_idx, 1]


        # x_list = [0, 1, 2]
        # y_list = [0, 1, 2]
        x_list = list(np.arange(30))
        y_list = list(np.arange(30))
        coord_y_list = []
        coord_x_list = []
        for i in y_list:
            for j in x_list:
                coord_y_list.append(coord_y + i)
                coord_x_list.append(coord_x + j)
        coord_y = torch.cat(coord_y_list)
        coord_x = torch.cat(coord_x_list)
        
        data = self.data[:, coord_y, coord_x, :].view(-1, 3)
        coords = self.mgrid[:, coord_y, coord_x, :].view(-1, 3)
        if self.flow is not None:
            flow = self.flow[:, coord_y, coord_x, :].view(-1, 6)
        if self.exps is not None:
            exps = self.exps[:, coord_y, coord_x, :].view(-1, 1)
        if self.mask is not None:
            mask = self.mask[:, coord_y, coord_x, :].view(-1, 1)
        if self.weights is not None:
            weights = self.weights[:, coord_y, coord_x, :].view(-1, 1)
        if self.depth is not None:
            depth = self.depth[:, coord_y, coord_x, :].view(-1, 1)

        gt_dict = {'img': data}
        in_dict = {'idx': idx,  'coords': coords, 'coord_idx': torch.stack([coord_y, coord_x,], 0)}
        if self.flow is not None:
            in_dict.update({'flow': flow})
        if self.exps is not None:
            in_dict.update({'exps': exps})
        if self.mask is not None:
            in_dict.update({'mask': mask})
        if self.weights is not None:
            in_dict.update({'weights': weights})
        if self.depth is not None:
            in_dict.update({'depth': depth})
        
        return in_dict, gt_dict


class Implicit3DWrapper(torch.utils.data.Dataset):
    def __init__(self, dataset, sidelength=None, sample_fraction=1.):

        self.dataset = dataset.vid[..., :3]
        self.flow = None
        self.exps = None
        self.mask = None
        self.weights = None
        self.gamma_weights = None
        self.depth = None
        if dataset.flow is not None:
            self.flow = torch.from_numpy(dataset.flow).view(-1, 6)
        if dataset.exps is not None:
            self.exps = torch.from_numpy(dataset.exps).view(-1, 1)
        if dataset.mask is not None:
            self.mask = torch.from_numpy(dataset.mask).view(-1, 1)
        if dataset.weights is not None:
            self.weights = torch.from_numpy(dataset.weights).view(-1, 1)
        if dataset.gamma_weights is not None:
            self.gamma_weights = torch.from_numpy(dataset.gamma_weights).view(-1, 1)
        if dataset.depth is not None and (dataset.depth.shape[0] == dataset.vid.shape[0]):
            self.depth = ((torch.from_numpy(dataset.depth)- 0.5) / 0.5).view(-1, 1)
            
        self.mgrid = get_mgrid(sidelength, dim=3).view(-1, 3)
        self.data = ((torch.from_numpy(self.dataset) - 0.5) / 0.5).view(-1, 3)
        self.sample_fraction = sample_fraction
        self.N_samples = int(self.sample_fraction * self.mgrid.shape[0])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):

        if self.sample_fraction < 1.:
            coord_idx = torch.randint(0, self.data.shape[0], (self.N_samples,))
        else:
            coord_idx = torch.arange(self.data.shape[0])
        
        data = self.data[coord_idx, :]
        coords = self.mgrid[coord_idx, :]
        if self.flow is not None:
            flow = self.flow[coord_idx, :]
        if self.exps is not None:
            exps = self.exps[coord_idx, :]
        if self.mask is not None:
            mask = self.mask[coord_idx, :]
        if self.weights is not None:
            weights = self.weights[coord_idx, :]
        if self.gamma_weights is not None:
            gamma_weights = self.gamma_weights[coord_idx, :]
        if self.depth is not None:
            depth = self.depth[coord_idx, :]

        gt_dict = {'img': data}
        in_dict = {'idx': idx,  'coords': coords, 'coord_idx': coord_idx}
        if self.flow is not None:
            in_dict.update({'flow': flow})
        if self.exps is not None:
            in_dict.update({'exps': exps})
        if self.mask is not None:
            in_dict.update({'mask': mask})
        if self.weights is not None:
            in_dict.update({'weights': weights})
        if self.gamma_weights is not None:
            in_dict.update({'gamma_weights': gamma_weights})
        if self.depth is not None:
            in_dict.update({'depth': depth})
        
        return in_dict, gt_dict
