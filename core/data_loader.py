"""
StarGAN v2
Copyright (c) 2020-present NAVER Corp.

This work is licensed under the Creative Commons Attribution-NonCommercial
4.0 International License. To view a copy of this license, visit
http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.
"""

from pathlib import Path
from itertools import chain
import os
import random
import cv2

from munch import Munch
from PIL import Image
import numpy as np

import torch
from torch.utils import data
from torch.utils.data.sampler import WeightedRandomSampler
from torchvision import transforms
from torchvision.datasets import ImageFolder


def listdir(dname):
    fnames = list(chain(*[list(Path(dname).rglob('*.' + ext))
                          for ext in ['png', 'jpg', 'jpeg', 'JPG']]))
    return fnames


class DefaultDataset(data.Dataset):
    def __init__(self, root, from_dir, mask_root, mask_type, transform=None, p_transform=None):
        self.samples, self.targets = self._make_dataset(root)
        self.transform = transform
        self.p_transform = p_transform
        self.from_dir = from_dir
        self.mask_root = mask_root
        self.mask_type = mask_type

    def _make_dataset(self, root):
        domains = os.listdir(root)
        fnames, labels = [], []
        for idx, domain in enumerate(sorted(domains)):
            class_dir = os.path.join(root, domain)
            cls_fnames = listdir(class_dir)
            fnames += cls_fnames
            labels += [idx] * len(cls_fnames)
        return fnames, labels

    def __getitem__(self, index):
        fname = str(self.samples[index])
        img = Image.open(fname).convert('RGB')
        if fname.split('/')[-3] == 'train':
            p_name = fname.replace('/train/', '/train_parsing/', 1)
        else:
            p_name = fname.replace('/val/', '/val_parsing/', 1)
        p_name = p_name[:-4] + '.png'
        parsing = Image.open(p_name)
        if self.mask_type == 'hybrid':
            mask = self._hybrid(shape=256, max_angle=4, max_len=40, max_width=10, times=20, margin=10, bbox_shape=30)
        elif self.mask_type == 'bbox':
            mask = self.bbox2mask(shape=256, margin=50, bbox_shape=100, times=1)
        else:
            mask = self.rectanglemask(shape=256)
        if self.transform is not None:
            img = self.transform(img)
            parsing = self.p_transform(parsing)
        label = self.targets[index]
        mask = torch.from_numpy(mask)
        return img, parsing.float(), label, mask

    def __len__(self):
        return len(self.samples)

    def random_bbox(self, shape, margin, bbox_shape):
        """Generate a random tlhw with configuration.
        Args:
            config: Config should have configuration including IMG_SHAPES, VERTICAL_MARGIN, HEIGHT, HORIZONTAL_MARGIN, WIDTH.
        Returns:
            tuple: (top, left, height, width)
        """
        img_height = shape
        img_width = shape
        height = bbox_shape
        width = bbox_shape
        ver_margin = margin
        hor_margin = margin
        maxt = img_height - ver_margin - height
        maxl = img_width - hor_margin - width
        t = np.random.randint(low=ver_margin, high=maxt)
        l = np.random.randint(low=hor_margin, high=maxl)
        h = height
        w = width
        return (t, l, h, w)

    def rectanglemask(self, shape):

        bbox = (64, 64, 128, 128)
        height = shape
        width = shape
        mask = np.zeros((height, width), np.float32)
        mask[(bbox[0]): (bbox[0] + bbox[2]), (bbox[1]): (bbox[1] + bbox[3])] = 1.
        return mask.reshape((1, ) + mask.shape).astype(np.float32)

    def bbox2mask(self, shape, margin, bbox_shape, times):
        """Generate mask tensor from bbox.
        Args:
            bbox: configuration tuple, (top, left, height, width)
            config: Config should have configuration including IMG_SHAPES,
                MAX_DELTA_HEIGHT, MAX_DELTA_WIDTH.
        Returns:
            tf.Tensor: output with shape [1, H, W, 1]
        """
        bboxs = []
        for i in range(times):
            bbox = self.random_bbox(shape, margin, bbox_shape)
            bboxs.append(bbox)
        height = shape
        width = shape
        mask = np.zeros((height, width), np.float32)
        for bbox in bboxs:
            h = int(bbox[2] * 0.1) + np.random.randint(int(bbox[2] * 0.2 + 1))
            w = int(bbox[3] * 0.1) + np.random.randint(int(bbox[3] * 0.2) + 1)
            mask[(bbox[0] + h): (bbox[0] + bbox[2] - h), (bbox[1] + w): (bbox[1] + bbox[3] - w)] = 1.
        return mask.reshape((1, ) + mask.shape).astype(np.float32)

    def _hybrid(self, shape, max_angle=4, max_len=40, max_width=10, times=15, margin=10, bbox_shape=30):
        height = shape
        width = shape
        mask = np.zeros((height, width), np.float32)
        times = np.random.randint(times - 5, times)
        for i in range(times):
            start_x = np.random.randint(width)
            start_y = np.random.randint(height)
            for j in range(5 + np.random.randint(10)):
                angle = 0.01 + np.random.randint(max_angle)
                if i % 2 == 0:
                    angle = 2 * 3.1415926 - angle
                length = 10 + np.random.randint(max_len)
                brush_w = 5 + np.random.randint(max_width)
                end_x = (start_x + length * np.sin(angle)).astype(np.int32)
                end_y = (start_y + length * np.cos(angle)).astype(np.int32)
                cv2.line(mask, (start_y, start_x), (end_y, end_x), 1.0, brush_w)
                start_x, start_y = end_x, end_y

        bboxs = []
        for i in range(times - 5):
            bbox = self._random_bbox(shape, margin, bbox_shape)
            bboxs.append(bbox)

        for bbox in bboxs:
            h = int(bbox[2] * 0.1) + np.random.randint(int(bbox[2] * 0.2 + 1))
            w = int(bbox[3] * 0.1) + np.random.randint(int(bbox[3] * 0.2) + 1)
            mask[(bbox[0] + h): (bbox[0] + bbox[2] - h), (bbox[1] + w): (bbox[1] + bbox[3] - w)] = 1.
        return mask.reshape((1,) + mask.shape).astype(np.float32)


class ReferenceDataset(data.Dataset):
    def __init__(self, root, transform=None, p_transform=None):
        self.samples, self.targets = self._make_dataset(root)
        self.transform = transform
        self.p_transform = p_transform

    def _make_dataset(self, root):
        domains = os.listdir(root)
        fnames, fnames2, labels = [], [], []
        for idx, domain in enumerate(sorted(domains)):
            class_dir = os.path.join(root, domain)
            cls_fnames = listdir(class_dir)
            fnames += cls_fnames
            fnames2 += random.sample(cls_fnames, len(cls_fnames))
            labels += [idx] * len(cls_fnames)
        return list(zip(fnames, fnames2)), labels

    def __getitem__(self, index):
        fname, fname2 = self.samples[index]
        fname, fname2 = str(fname), str(fname2)
        label = self.targets[index]
        img = Image.open(fname).convert('RGB')
        img2 = Image.open(fname2).convert('RGB')
        if fname.split('/')[-3] == 'train':
            p_name = fname.replace('/train/', '/train_parsing/', 1)
            p_name2 = fname2.replace('/train/', '/train_parsing/', 1)
        else:
            p_name = fname.replace('/val/', '/val_parsing/', 1)
            p_name2 = fname2.replace('/val/', '/val_parsing/', 1)
        p_name = p_name[:-4] + '.png'
        p_name2 = p_name2[:-4] + '.png'
        parsing = Image.open(p_name)
        parsing2 = Image.open(p_name2)
        if self.transform is not None:
            img = self.transform(img)
            img2 = self.transform(img2)
            parsing = self.p_transform(parsing)
            parsing2 = self.p_transform(parsing2)
        return img, parsing.float(), img2, parsing2.float(), label

    def __len__(self):
        return len(self.targets)


def _make_balanced_sampler(labels):
    class_counts = np.bincount(labels)
    class_weights = 1. / class_counts
    weights = class_weights[labels]
    return WeightedRandomSampler(weights, len(weights))


def get_train_loader(root, from_dir, mask_root, mask_type, which='source',
                     img_size=256,  batch_size=8, prob=0.5, num_workers=4, pre=False):
    print('Preparing DataLoader to fetch %s images '
          'during the training phase...' % which)

    """
    crop = transforms.RandomResizedCrop(
        img_size, scale=[0.8, 1.0], ratio=[0.9, 1.1])
    rand_crop = transforms.Lambda(
        lambda x: crop(x) if random.random() < prob else x)
    """

    transform = transforms.Compose([
        # rand_crop,
        transforms.Resize([img_size, img_size]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                             std=[0.5, 0.5, 0.5]),
    ])
    p_transform = transforms.Compose([
        # rand_crop,
        transforms.Resize([img_size, img_size], transforms.InterpolationMode.NEAREST),
        transforms.PILToTensor()
    ])
    if pre or which == 'source':
        dataset = DefaultDataset(root, from_dir, mask_root, mask_type, transform, p_transform)
    else:
        dataset = ReferenceDataset(root, transform, p_transform)

    sampler = _make_balanced_sampler(dataset.targets)
    return data.DataLoader(dataset=dataset,
                           batch_size=batch_size,
                           sampler=sampler,
                           num_workers=num_workers,
                           pin_memory=True,
                           drop_last=True)


def get_eval_loader(root, from_dir, mask_root, mask_type, img_size=256, batch_size=32,
                    imagenet_normalize=True, shuffle=True,
                    num_workers=4, drop_last=False):
    print('Preparing DataLoader for the evaluation phase...')
    if imagenet_normalize:
        height, width = 299, 299
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    else:
        height, width = img_size, img_size
        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]

    transform = transforms.Compose([
        transforms.Resize([img_size, img_size]),
        transforms.Resize([height, width]),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    p_transform = transforms.Compose([
        # rand_crop,
        transforms.Resize([img_size, img_size], transforms.InterpolationMode.NEAREST),
        transforms.PILToTensor()
    ])
    dataset = DefaultDataset(root, from_dir, mask_root, mask_type, transform=transform, p_transform=p_transform)
    return data.DataLoader(dataset=dataset,
                           batch_size=batch_size,
                           shuffle=shuffle,
                           num_workers=num_workers,
                           pin_memory=True,
                           drop_last=drop_last)


def get_test_loader(root, from_dir, mask_root, mask_type, img_size=256, batch_size=32,
                    shuffle=True, num_workers=4):
    print('Preparing DataLoader for the generation phase...')
    transform = transforms.Compose([
        transforms.Resize([img_size, img_size]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                             std=[0.5, 0.5, 0.5]),
    ])

    p_transform = transforms.Compose([
        # rand_crop,
        transforms.Resize([img_size, img_size], transforms.InterpolationMode.NEAREST),
        transforms.PILToTensor()
    ])

    dataset = DefaultDataset(root, from_dir, mask_root, mask_type, transform, p_transform)
    return data.DataLoader(dataset=dataset,
                           batch_size=batch_size,
                           shuffle=shuffle,
                           num_workers=num_workers,
                           pin_memory=True)


class InputFetcher:
    def __init__(self, loader, loader_ref=None, latent_dim=16, mode='', pre=False):
        self.loader = loader
        self.loader_ref = loader_ref
        self.latent_dim = latent_dim
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.mode = mode

    def _fetch_inputs(self):
        try:
            x, p, y, m = next(self.iter)
        except (AttributeError, StopIteration):
            self.iter = iter(self.loader)
            x, p, y, m = next(self.iter)
        return x, p, y, m

    def _fetch_refs(self):
        try:
            x, p, x2, p2, y = next(self.iter_ref)
        except (AttributeError, StopIteration):
            self.iter_ref = iter(self.loader_ref)
            x, p, x2, p2, y = next(self.iter_ref)
        return x, p, x2, p2, y

    def __next__(self):
        x, p, y, m = self._fetch_inputs()
        if self.mode == 'train':
            x_ref, p_ref, x_ref2, p_ref2, y_ref = self._fetch_refs()
            z_trg = torch.randn(x.size(0), self.latent_dim)
            z_trg2 = torch.randn(x.size(0), self.latent_dim)
            inputs = Munch(x_src=x, p_src=p, y_src=y, y_ref=y_ref, mask=m,
                           x_ref=x_ref, p_ref=p_ref, x_ref2=x_ref2, p_ref2=p_ref2,
                           z_trg=z_trg, z_trg2=z_trg2)
        elif self.mode == 'val':
            x_ref, p_ref, y_ref, _ = self._fetch_inputs()
            inputs = Munch(x_src=x, p_src=p, y_src=y, mask=m,
                           x_ref=x_ref, p_ref=p_ref, y_ref=y_ref)
        elif self.mode == 'test':
            inputs = Munch(x=x, p=p, y=y, mask=m)
        else:
            raise NotImplementedError

        return Munch({k: v.to(self.device)
                      for k, v in inputs.items()})