import os
import torch
import torch.utils.data as data
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
import cv2

class K2Dataset(data.Dataset):
    def __init__(self, args, mode, transforms=None):
        self.args = args
        self.mode = mode
        self.alb_transforms = transforms
        self.data_fmt = [".png", ".bmp", ".jpg", ".JPG", ".JPEG"]
        if self.mode != "train":
            self.data_dir = r"D:\datasets\K2_datasets\CIMS_230829"
        else:
            self.data_dir = self.args.root_path
        self.data_path = None
        for item in os.listdir(self.data_dir):
            if item.lower() == self.mode:
                self.data_path = os.path.join(self.data_dir, item)
                break

        self.labels_map = dict()
        self.labels_map = self.__ReturnLabelsMap(self.args.labels)
        self.data_dict = self.__CreateData(self.data_path, self.data_fmt, self.labels_map)
    
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
        if self.alb_transforms:
            image = self.alb_transforms(image=np.asarray(image))["image"]

        return image, target, image_path
              
    def __len__(self):
        return len(self.data_dict)