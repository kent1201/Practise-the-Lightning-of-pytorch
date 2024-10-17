import os
import random
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch
import torchvision.transforms as transforms
from data.Classification.k2_dataset import K2Dataset
from data.Classification.animals_64classes_dataset import Animals64ClassesDataset

DATASETS = {
    "K2": K2Dataset,
    "animals_64classes": Animals64ClassesDataset
}

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


if __name__=="__main__":
    from PIL import Image
    import albumentations as A
    from albumentations.pytorch import ToTensorV2

    def GetDataPath(data_dir, data_fmt=[".png", ".bmp", ".jpg", ".JPG", ".JPEG"]):
        """
        data_dir(str): images directory path
        data_fmt(list): the data format want to get. default: [".png", ".bmp", ".jpg", ".JPG", ".JPEG"]
        """
        data_path = list()
        for dirPath, dirNames, fileNames in os.walk(data_dir):
            for f in fileNames:
                if os.path.splitext(f)[1] in data_fmt:
                    data_path.append(os.path.join(dirPath, f))
        return data_path

    reference_image_path_list = GetDataPath(os.path.join(r"D:\datasets\K2_datasets\CIMS_ModelB_v2_2", "Train"))
    a_transform = A.augmentations.domain_adaptation.PixelDistributionAdaptation(reference_image_path_list, transform_type='minmax') # A.augmentations.domain_adaptation.PixelDistributionAdaptation(reference_image_path_list, transform_type='minmax')
    image_path = r"D:\datasets\K2_datasets\CIMS_ModelB_v2_2\Test\CP08\Null_6-HL-J_22350729_22350729002_6.jpg"
    org_image = Image.open(image_path)
    new_image = Image.fromarray(a_transform(image=np.asarray(org_image))["image"])
    new_image.save(r"D:\datasets\K2_datasets\CIMS_ModelB_v2_2\Test\CP08\Convert_Null_6-HL-J_22350729_22350729002_6.jpg")