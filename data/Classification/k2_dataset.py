import os
import cv2
import torch
import torch.utils.data as data
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2

from utils.utils import GetDataPath, ListDir

class K2Dataset(data.Dataset):
    def __init__(self, args, mode):
        self.args = args
        self.mode = mode
        self.mixup_interval = self.args.mixup_interval
        self.mixup_alpha = self.args.mixup_alpha
        self.data_fmt = [".png", ".bmp", ".jpg", ".JPG", ".JPEG"]
        self.data_dir = self.args.root_path
        self.data_path = None
        for item in os.listdir(self.data_dir):
            if item.lower() == self.mode:
                self.data_path = os.path.join(self.data_dir, item)
                break

        self.labels_map = dict()
        self.labels_map = self.__ReturnLabelsMap(self.args.labels)
        self.data_dict = self.__CreateData(self.data_path, self.data_fmt, self.labels_map)
        self.transforms = self.DataTransform(mean=[0.5667182803153992, 0.5667171478271484, 0.5666833519935608],
                                           std=[0.15075330436229706, 0.14988024532794952, 0.1496531218290329],
                                           mode=self.mode)

    def DataTransform(self, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], mode='train', reference_img_path = None):
        if not reference_img_path:
            reference_img_path = self.args.root_path
        reference_image_path_list = GetDataPath(reference_img_path)
        # mean: [0.4804353415966034, 0.48068004846572876, 0.48095497488975525]    std: [0.15915930271148682, 0.15888161957263947, 0.15881265699863434]
        transform = None
        if mode == "train":
            transform = A.Compose([A.CLAHE(p=0.01), 
                                A.Resize(height=self.args.load_size, width=self.args.load_size, interpolation=cv2.INTER_NEAREST),
                                #    A.augmentations.domain_adaptation.PixelDistributionAdaptation(reference_image_path_list, transform_type='minmax'),
                                A.CenterCrop(height=self.args.crop_size, width=self.args.crop_size),
                                A.HorizontalFlip(p=0.5),
                                A.VerticalFlip(p=0.5),
                                A.Perspective(p=0.5),
                                A.Transpose(p=0.4),
                                A.OneOf([A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, p=0.2),
                                            A.Affine(p=0.2)]),
                                A.OneOf([A.augmentations.transforms.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=15, val_shift_limit=15, always_apply=False, p=0.3),
                                            A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.3),
                                            A.augmentations.transforms.RGBShift(r_shift_limit=5, g_shift_limit=5, b_shift_limit=5, always_apply=False, p=0.3)]),
                                A.MedianBlur(always_apply=True),
                                A.RandomGridShuffle(p=0.4),
                                #    A.OneOf([A.Blur(p=0.9),
                                #             A.MedianBlur(p=0.9)]),
                                A.Normalize(mean=(mean[0], mean[1], mean[2]), std=(std[0], std[1], std[2])),
                                ToTensorV2()
                                ])
        else:
            transform = A.Compose([A.Resize(height=self.args.load_size, width=self.args.load_size, interpolation=cv2.INTER_NEAREST),
                                A.CenterCrop(height=self.args.crop_size, width=self.args.crop_size),
                                #    A.augmentations.domain_adaptation.PixelDistributionAdaptation(reference_image_path_list, transform_type='minmax'),
                                #    A.augmentations.domain_adaptation.HistogramMatching(reference_image_path_list),
                                #    A.augmentations.domain_adaptation.FDA(reference_image_path_list, p=1, read_fn=lambda x: x),
                                #    A.OneOf([A.Blur(p=0.9), A.MedianBlur(p=0.9)]),
                                A.MedianBlur(always_apply=True),
                                A.Normalize(mean=(mean[0], mean[1], mean[2]), std=(std[0], std[1], std[2])),
                                ToTensorV2()
                                ])
        return transform
    
    def __ReturnLabelsMap(self, labels):
        labels_list = [x.strip() for x in labels.split(',')]
        sorted(labels_list)
        labels_map = dict()
        for index, item in enumerate(labels_list):
            labels_map[item] = index
        return labels_map
    
    def __CreateData(self, root_path, data_fmt, labels_map):
        dataset = list()
        for label_dir in os.listdir(root_path):
            if label_dir in labels_map.keys():
                label_dir_path = os.path.join(root_path, label_dir)
                for data in os.listdir(label_dir_path):
                    if os.path.splitext(data)[1] in data_fmt:
                        dataset.append({"image_path": os.path.join(label_dir_path, data), "label": labels_map[label_dir]})
        return dataset

    def __getitem__(self, index):
        image_path, label = self.data_dict[index]["image_path"], self.data_dict[index]["label"]
        image = Image.open(image_path)
        target = torch.nn.functional.one_hot(torch.LongTensor([label]), len(self.labels_map.keys()))
        target = target.type(torch.FloatTensor)
        ## MixUP
        if self.mode == "train" and index > 0 and index % self.mixup_interval == 0:
            mixup_idx = np.random.randint(0, len(self.data_dict)-1)
            mixup_image_path, mixup_label = self.data_dict[mixup_idx]["image_path"], self.data_dict[mixup_idx]["label"]
            mixup_image = Image.open(mixup_image_path)
            mixup_target = torch.nn.functional.one_hot(torch.LongTensor([mixup_label]), len(self.labels_map.keys()))
            mixup_image = self.transforms(image=np.asarray(mixup_image))["image"]
            lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)
            image = lam * image + (1 - lam) * mixup_image
            target = lam * target + (1 - lam) * mixup_target

        return image, target, image_path
              
    def __len__(self):
        return len(self.data_dict)