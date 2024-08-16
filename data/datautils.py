import os
from typing import Tuple
from PIL import Image
import numpy as np

import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from data.hoi_dataset import BongardDataset
try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

from data.fewshot_datasets import *
import data.augmix_ops as augmentations

ID_to_DIRNAME={
    'I': 'ImageNet',
    'A': 'imagenet-adversarial/imagenet-a',
    'K': 'ImageNet-Sketch',
    'R': 'imagenet-rendition/imagenet-r', # filtered_images_resized / background_removal_ / imagenet-r
    'V': 'imagenetv2/imagenetv2-matched-frequency-format-val',
    'flower102': 'oxford_flowers',
    'dtd': 'dtd',
    'pets': 'oxford_pets/images',
    'cars': 'stanford_cars',
    'ucf101': 'ucf101/UCF-101-midframes',
    'caltech101': 'caltech-101/101_ObjectCategories',
    'food101': 'food-101',
    'sun397': 'sun397/SUN397',
    'aircraft': 'fgvc_aircraft',
    'eurosat': 'eurosat/2750',
}

def build_dataset(set_id, transform, args, attack_step=0, 
                  n_shot=None, split="all", bongard_anno=False):
    #example of folder attack name: attack-{steps_our}steps-{attack_name}
    if set_id == 'I':
        # ImageNet validation set
        testdir = os.path.join(os.path.join(args.data, ID_to_DIRNAME[set_id]), 'val')
        testset = datasets.ImageFolder(testdir, transform=transform)

    elif set_id in ['A', 'K', 'R', 'V']:
        # if args.our_attack and (attack_step !=0): #get attack created in previous step
        #     testdir = os.path.join(args.data, ID_to_DIRNAME[set_id]+'-'+ args.attack_folder_name)
        # if args.evaluate_on_attack:
        #     testdir = os.path.join(args.data, ID_to_DIRNAME[set_id]+'-'+ args.attack_folder_name)
        # else: 
        testdir = os.path.join(args.data, ID_to_DIRNAME[set_id])
        # testdir = "/home/raza.imam/Documents/TPT/datasets/ImageNet/Imagenet_1k/train"
        from data import filtered_dataset
        if args.images_per_class == None:
            testset = datasets.ImageFolder(testdir, transform=transform)
        else:
            testset = filtered_dataset.FilteredImageNetDataset(testdir, transform=transform, images_per_class=args.images_per_class)
        
        
    elif set_id in fewshot_datasets:
        if args.dataset_mode == 'train' and n_shot:
            testset = build_fewshot_dataset(set_id, os.path.join(args.data, ID_to_DIRNAME[set_id.lower()]), transform, mode=args.dataset_mode, n_shot=n_shot)
        else:
            testset = build_fewshot_dataset(set_id, os.path.join(args.data, ID_to_DIRNAME[set_id.lower()]), transform, mode=args.dataset_mode)
    elif set_id == 'bongard':
        assert isinstance(transform, Tuple)
        base_transform, query_transform = transform
        testset = BongardDataset(args.data, split, args.dataset_mode, base_transform, query_transform, bongard_anno)
    else:
        raise NotImplementedError
        
    return testset

def build_dataset_old(set_id, transform, data_root, mode='test', n_shot=None, split="all", bongard_anno=False):
    if set_id == 'I':
        # ImageNet validation set
        testdir = os.path.join(os.path.join(data_root, ID_to_DIRNAME[set_id]), 'val')
        testset = datasets.ImageFolder(testdir, transform=transform)
    elif set_id in ['A', 'K', 'R', 'V']:
        testdir = os.path.join(data_root, ID_to_DIRNAME[set_id])
        testset = datasets.ImageFolder(testdir, transform=transform)
    elif set_id in fewshot_datasets:
        if mode == 'train' and n_shot:
            testset = build_fewshot_dataset(set_id, os.path.join(data_root, ID_to_DIRNAME[set_id.lower()]), transform, mode=mode, n_shot=n_shot)
        else:
            testset = build_fewshot_dataset(set_id, os.path.join(data_root, ID_to_DIRNAME[set_id.lower()]), transform, mode=mode)
    elif set_id == 'bongard':
        assert isinstance(transform, Tuple)
        base_transform, query_transform = transform
        testset = BongardDataset(data_root, split, mode, base_transform, query_transform, bongard_anno)
    else:
        raise NotImplementedError
        
    return testset

# AugMix Transforms
def get_preaugment():
    return transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        # transforms.ColorJitter(brightness=0., contrast=0.2, saturation=0.2, hue=0.1),
        # transforms.RandomRotation(degrees=15),
        # transforms.RandomVerticalFlip(),
        # transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        # transforms.RandomPerspective(distortion_scale=0.1, p=0.5, interpolation=3),
        # transforms.RandomErasing(p=0.2, scale=(0.02, 0.2), ratio=(0.3, 3.3)),
        ])

def augmix(image, preprocess, aug_list, severity=1):
    preaugment = get_preaugment()
    x_orig = preaugment(image)
    x_processed = preprocess(x_orig)
    if len(aug_list) == 0:
        return x_processed
    w = np.float32(np.random.dirichlet([1.0, 1.0, 1.0]))
    m = np.float32(np.random.beta(1.0, 1.0))

    mix = torch.zeros_like(x_processed)
    for i in range(3):
        x_aug = x_orig.copy()
        for _ in range(np.random.randint(1, 4)):
            x_aug = np.random.choice(aug_list)(x_aug, severity)
        mix += w[i] * preprocess(x_aug)
    mix = m * x_processed + (1 - m) * mix
    return mix


class AugMixAugmenter(object):
    def __init__(self, base_transform, preprocess, n_views=2, augmix=False, 
                    severity=1):
        self.base_transform = base_transform
        self.preprocess = preprocess
        self.n_views = n_views
        if augmix:
            self.aug_list = [] #augmentations.augmentations
        else:
            self.aug_list = []
        self.severity = severity
        
    def __call__(self, x, again_transform=True, n_views=5):
        if again_transform == True:
            preprocess = self.preprocess
            n_views = self.n_views
            image = self.preprocess(self.base_transform(x))
        else:
            preprocess = transforms.Compose([
                # transforms.ToTensor(),
                transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                     std=[0.26862954, 0.26130258, 0.27577711])
                ])
            # image = transforms.ToTensor()(x)
            n_views = n_views
            image = preprocess(x)
        # image = self.preprocess(self.base_transform(x))
        views = [augmix(x, preprocess, self.aug_list, self.severity) for _ in range(n_views)]
        return [image] + views



