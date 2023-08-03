import os
import random
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torchvision.transforms as transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2

from utils.utils import GetDataPath

def DataTransform(args, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], mode='train'):
    reference_image_path_list = GetDataPath(os.path.join(args.root_path, "Train"))
    # mean: [0.4804353415966034, 0.48068004846572876, 0.48095497488975525]    std: [0.15915930271148682, 0.15888161957263947, 0.15881265699863434]
    if mode == "train":
        transform = A.Compose([A.CLAHE(p=0.01), 
                               A.Resize(height=args.load_size, width=args.load_size),
                               A.augmentations.domain_adaptation.PixelDistributionAdaptation(reference_image_path_list, transform_type='minmax'),
                               A.RandomCrop(height=args.crop_size, width=args.crop_size),
                               A.HorizontalFlip(p=0.5),
                               A.VerticalFlip(p=0.5),
                               A.Perspective(p=0.5),
                               A.Transpose(p=0.2),
                               A.RandomGridShuffle(p=0.01),
                               A.OneOf([A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, p=0.2), A.Affine(p=0.2)]),
                               A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.1),
                               A.OneOf([A.Blur(p=0.01), A.MedianBlur(p=0.01)]),
                               A.Normalize(mean=(mean[0], mean[1], mean[2]), std=(std[0], std[1], std[2])),
                               ToTensorV2()
                            ])
    else:
        transform = A.Compose([A.Resize(height=args.load_size, width=args.load_size),
                            #    A.augmentations.domain_adaptation.HistogramMatching(reference_image_path_list),
                               A.augmentations.domain_adaptation.PixelDistributionAdaptation(reference_image_path_list, transform_type='minmax'),
                               A.Normalize(mean=(mean[0], mean[1], mean[2]), std=(std[0], std[1], std[2])),
                               ToTensorV2()
                            ])
    return transform

def compute_mean_std(data_path: str, resize=224, data_fmt=[".png", ".bmp", ".jpg", ".JPG", ".JPEG"]):
    """compute the mean and std of cifar100 dataset
    Args:
        cifar100_training_dataset or cifar100_test_dataset
        witch derived from class torch.utils.data

    Returns:
        a tuple contains mean, std value of entire dataset
    """
    import gc
    print("claculate mean and std of dataset...")

    data_r_list = list()
    data_g_list = list()
    data_b_list = list()
    file_list = list()
    for dirPath, dirNames, fileNames in os.walk(data_path):
        for item in fileNames:
            if os.path.splitext(item)[1] in data_fmt:
                file_list.append(os.path.join(dirPath, item))
    random.shuffle(file_list)

    for item in tqdm(range(min(len(file_list), 10000))):
        image = Image.open(file_list[item])
        image = transforms.Resize(size=resize)(image)
        temp_tensor = transforms.ToTensor()(image)
        data_r_list.append(temp_tensor[:, :, 0])
        data_g_list.append(temp_tensor[:, :, 1])
        data_b_list.append(temp_tensor[:, :, 2])
    data_r = torch.stack(data_r_list).float()
    data_g = torch.stack(data_g_list).float()
    data_b = torch.stack(data_b_list).float()
    mean = [torch.mean(data_r).item(), torch.mean(data_g).item(), torch.mean(data_b).item()]
    std = [torch.std(data_r).item(), torch.std(data_g).item(), torch.std(data_b).item()]

    del file_list, temp_tensor, data_r_list, data_g_list, data_b_list, data_r, data_g, data_b
    gc.collect()
    print("mean: {}\tstd: {}".format(mean, std))
    return mean, std