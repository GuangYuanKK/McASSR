import functools
import random
import math
from PIL import Image

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from datasets import register
from utils import to_pixel_samples
from utils import make_coord

def to_tensor(data):
    return torch.from_numpy(data)

@register('sr-implicit-paired')
class SRImplicitPaired(Dataset):

    def __init__(self, dataset, inp_size=None, augment=False, sample_q=None):
        self.dataset = dataset
        self.inp_size = inp_size
        self.augment = augment
        self.sample_q = sample_q

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img_lr, img_hr, img_ref = self.dataset[idx]

        s = img_hr.shape[-2] // img_lr.shape[-2] # assume int scale

        h_lr, w_lr = img_lr.shape[-2:]
        img_hr = img_hr[:, :h_lr * s, :w_lr * s]
        img_ref = img_ref[:, :h_lr * s, :w_lr * s]
        crop_lr, crop_hr, crop_ref = img_lr, img_hr, img_ref
        crop_ref_lr = resize_fn(img_ref, w_lr)


        if self.augment:
            hflip = random.random() < 0.5
            vflip = random.random() < 0.5
            dflip = random.random() < 0.5

            def augment(x):
                if hflip:
                    x = x.flip(-2)
                if vflip:
                    x = x.flip(-1)
                if dflip:
                    x = x.transpose(-2, -1)
                return x
            ###### Tar #######
            crop_lr = augment(crop_lr)
            crop_hr = augment(crop_hr)
            ###### Ref #######
            crop_ref_lr = augment(crop_ref_lr)
            crop_ref = augment(crop_ref)

        ###### Tar #######
        hr_coord, hr_rgb = to_pixel_samples(crop_hr.contiguous())
        ###### Ref #######
        _, ref_rgb = to_pixel_samples(crop_ref.contiguous())
        # print(ref_rgb.shape)

        if self.sample_q is not None:
            sample_lst = np.random.choice(
                len(hr_coord), self.sample_q, replace=False)
            ###### Tar #######
            hr_coord = hr_coord[sample_lst]
            hr_rgb = hr_rgb[sample_lst]
            ###### Ref #######
            ref_rgb = ref_rgb[sample_lst]
        ref_w = int(np.sqrt(ref_rgb.shape[0]))
        ref_c = ref_rgb.shape[1]
        # print(ref_w,ref_c)
        ref_hr = ref_rgb.contiguous().view(ref_c, ref_w, ref_w)
        ###### Tar #######
        cell = torch.ones_like(hr_coord)
        cell[:, 0] *= 2 / crop_hr.shape[-2]
        cell[:, 1] *= 2 / crop_hr.shape[-1]


        return {
            'inp': crop_lr,
            'inp_hr_coord': hr_coord,
            'inp_cell': cell,
            'ref': crop_ref_lr,
            'ref_hr': ref_hr,
            'gt': hr_rgb
        }
    
def resize_fn(img, size):
    return transforms.ToTensor()(
        transforms.Resize(size, Image.BICUBIC)(
            transforms.ToPILImage()(img)))


@register('sr-implicit-downsampled')
class SRImplicitDownsampled(Dataset):

    def __init__(self, dataset, inp_size=None, scale_min=1, scale_max=None,
                 augment=False, sample_q=None):
        self.dataset = dataset
        self.inp_size = inp_size
        self.scale_min = scale_min
        if scale_max is None:
            scale_max = scale_min
        self.scale_max = scale_max
        self.augment = augment
        self.sample_q = sample_q

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):

        T2_img, T1_img = self.dataset[idx]

        s = random.uniform(self.scale_min, self.scale_max)

        w_lr = self.inp_size
        w_hr = round(w_lr * s)
        x0 = random.randint(0, T2_img.shape[-2] - w_hr)
        y0 = random.randint(0, T2_img.shape[-1] - w_hr)

        ####### prepare inp #########
        T2_crop_hr = T2_img[:, x0: x0 + w_hr, y0: y0 + w_hr]
        T2_crop_lr = resize_fn(T2_crop_hr, w_lr)
        ####### prepare ref #########
        T1_crop_hr = T1_img[:, x0: x0 + w_hr, y0: y0 + w_hr]
        T1_crop_lr = resize_fn(T1_crop_hr, w_lr)

        if self.augment:
            hflip = random.random() < 0.5
            vflip = random.random() < 0.5
            dflip = random.random() < 0.5

            def augment(x):
                if hflip:
                    x = x.flip(-2)
                if vflip:
                    x = x.flip(-1)
                if dflip:
                    x = x.transpose(-2, -1)
                return x

            ####### prepare inp #########
            T2_crop_lr = augment(T2_crop_lr)
            T2_crop_hr = augment(T2_crop_hr)
            ####### prepare ref #########
            T1_crop_lr = augment(T1_crop_lr)
            T1_crop_hr = augment(T1_crop_hr)

        ####### prepare inp #########
        T2_hr_coord, T2_hr_rgb = to_pixel_samples(T2_crop_hr.contiguous())

        ####### prepare ref #########
        _, T1_hr_rgb = to_pixel_samples(T1_crop_hr.contiguous())

        if self.sample_q is not None:
            sample_lst = np.random.choice(
                len(T2_hr_coord), self.sample_q, replace=False)
            ####### prepare inp #########
            T2_hr_coord = T2_hr_coord[sample_lst]
            T2_hr_rgb = T2_hr_rgb[sample_lst]
            ####### prepare ref #########
            T1_hr_rgb = T1_hr_rgb[sample_lst]

        ref_w = int(np.sqrt(T1_hr_rgb.shape[0]))
        ref_c = T1_hr_rgb.shape[1]
        T1_ref_hr = T1_hr_rgb.view(ref_c, ref_w, ref_w)
        ####### prepare inp #########
        T2_cell = torch.ones_like(T2_hr_coord)
        T2_cell[:, 0] *= 2 / T2_crop_hr.shape[-2]
        T2_cell[:, 1] *= 2 / T2_crop_hr.shape[-1]


        return {
            'inp': T2_crop_lr,
            'inp_hr_coord': T2_hr_coord,
            'inp_cell': T2_cell,
            'ref': T1_crop_lr,
            'ref_hr': T1_ref_hr,
            'gt': T2_hr_rgb,
        }
