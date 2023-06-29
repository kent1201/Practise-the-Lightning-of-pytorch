import os
import torch
import random
import numpy as np
import torch.utils.data as data
from PIL import Image
import torchvision.transforms as transforms

random.seed(0)
   
def ReturnLabelsMap(labels):
    labels_list = [x.strip() for x in labels.split(',')]
    sorted(labels_list)
    labels_map = dict()
    for index, item in enumerate(labels_list):
        labels_map[item] = index
    return labels_map

def CreateDataset(root_path, data_fmt, labels_map):
    dataset = list()
    for label_dir in os.listdir(root_path):
        if label_dir in labels_map.keys():
            label_dir_path = os.path.join(root_path, label_dir)
            for data in os.listdir(label_dir_path):
                if os.path.splitext(data)[1] in data_fmt:
                    dataset.append({"image_path": os.path.join(label_dir_path, data), "label": labels_map[label_dir]})
    return dataset

class ClassificationDataset(data.Dataset):
    def __init__(self, args, mode='train'):
        self.root = args.root_path
        self.data_fmt = [".png", ".bmp", ".jpg", ".JPG", ".JPEG"]
        if mode not in ["train", "val", "test", "inference"]:
            raise ValueError("mode {} not correct.".format(mode))
        self.mode = mode
        data_dir = None
        for item in os.listdir(self.root):
            if item.lower() == self.mode:
                data_dir = item
                break
        if not data_dir:
            raise FileNotFoundError("Dir {} not correct.".format(mode))
        self.data_path = os.path.join(self.root, data_dir)

        self.labels_map = dict()
        self.labels_map = ReturnLabelsMap(args.labels)
        self.dataset = CreateDataset(self.data_path, self.data_fmt, self.labels_map)
        random.shuffle(self.dataset)
        self.transform = DataTransform(args=args, mode=self.mode)

    def __getitem__(self, index):
        image_path, label = self.dataset[index]["image_path"], self.dataset[index]["label"]
        image = Image.open(image_path)
        target = torch.nn.functional.one_hot(torch.LongTensor([label]), len(self.labels_map.keys()))
        if self.transform:
            image = self.transform(image)
        if self.mode == "inference":
            return image
        return image, target
        
    def __len__(self):
        return len(self.dataset)

def DataLoader(dataset, batch_size, num_workers=0, shuffle=False):
    return torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

def DataTransform(args, mode='train'):
    if mode == "train":
        transform = transforms.Compose([transforms.Resize([args.load_size, args.load_size]),
                                        transforms.RandomCrop(args.crop_size),
                                        transforms.RandomHorizontalFlip(p=0.5),
                                        transforms.RandomVerticalFlip(p=0.5),
                                        transforms.RandomRotation(degrees=15),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.2016, 0.1240, -0.0568], std=[0.2636, 0.2394, 0.1653])])
    else:
        transform = transforms.Compose([transforms.Resize([args.load_size, args.load_size]),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.2016, 0.1240, -0.0568], std=[0.2636, 0.2394, 0.1653])])
    return transform

if __name__=="__main__":
    base_dataset = ClassificationDataset(root=r"/mnt/d/datasets/K2_data_0417/9/train")
    print(base_dataset.labels_map)
    print(len(base_dataset))