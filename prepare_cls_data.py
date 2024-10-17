import os
import shutil
import random
from tqdm import tqdm
import numpy as np
# import albumentations as A
from PIL import Image
from matplotlib import pyplot as plt

from common_data import transform, SRC_DIR, DATASET_DIR
from utils.utils import ListDir, CheckSavePath

random.seed(42)



def DataAugmentation(data_dict, dst_img_dir, up_limit = 600, aug_ratio = 0.3, mode='train') -> None:
    """
    對指定類別的圖像進行數據增強。

    Params:
        data_dict (dict): 包含每個類別的圖像路徑的字典。
        dst_img_dir (str): 增強後圖像保存的目錄。
        up_limit (int): 每個類別的圖像最少需要增強至這個數量。
        aug_ratio (float): 增強的比例，基於當前已有的圖像數量。
        mode (str): 增強的數據模式（例如 'train'）。
    Returns:
        None
    """
    min_generate_iter = 0
    for key, value in data_dict.items():
        min_generate_iter = 0
        if len(value) < up_limit:
            min_generate_iter = min(up_limit - len(value), int(len(value) * aug_ratio))
            print("len(value): {}\tmin_generate_iter: {}".format(len(value), min_generate_iter))
            random.shuffle(value)
            print("Preparing the augmentation of {} data...".format(key))
            for new_image_index in tqdm(range(0, min_generate_iter)):
                org_image = Image.open(value[new_image_index])
                new_image = Image.fromarray(transform(image=np.asarray(org_image))["image"])
                new_img_dir = os.path.join(dst_img_dir, mode, key)
                CheckSavePath(new_img_dir)
                new_image_path = os.path.join(new_img_dir, "aug_"+os.path.basename(value[new_image_index]))
                new_image.save(new_image_path)
                

def train_val_test_split(img_list, labels_list, split_ratio=0.8):
    """
    將圖像數據分為訓練、驗證和測試集。

    Params:
        img_list (list): 包含所有圖像路徑的列表。
        labels_list (list): 包含所有類別名稱的列表。
        split_ratio (float): 訓練集的分割比例，默認為 0.8。
    Returns:
        train_data_list (dict): 訓練集數據字典。
        val_data_list (dict): 驗證集數據字典。
        test_data_list (dict): 測試集數據字典。
    """
    data_list = dict()
    train_data_list = dict()
    val_data_list = dict()
    test_data_list = dict()
    for label in labels_list:
        data_list[label] = list()
        train_data_list[label] = list()
        val_data_list[label] = list()
        test_data_list[label] = list()
    for img_path in img_list:
        current_label = os.path.basename(os.path.dirname(img_path))
        data_list[current_label].append(img_path)
    for key in data_list.keys():
        size = int(len(data_list[key]) * split_ratio)
        val_test_size = int((len(data_list[key]) - size) * 0.5)
        train_data_list[key] = data_list[key][:size]
        val_data_list[key] = data_list[key][size:size+val_test_size]
        test_data_list[key] = data_list[key][size+val_test_size:]
    return train_data_list, val_data_list, test_data_list

def LimitSample(train_data_dict, limit_count=600):
    """
    限制每個類別的訓練樣本數量。

    Params:
        train_data_dict (dict): 訓練集數據字典。
        limit_count (int): 每個類別的最大樣本數量。
    Returns:
        train_data_dict (dict): 更新後的訓練集數據字典。
    """
    for label in train_data_dict.keys():
        if len(train_data_dict[label]) > limit_count:
            limit_data_list = train_data_dict[label]
            random.shuffle(limit_data_list)
            train_data_dict[label] = limit_data_list[:limit_count]
    return train_data_dict

def MoveData(data_dict, dst_img_dir, dir='train'):
    """
    將圖像數據移動到指定目錄。

    Params:
        data_dict (dict): 包含圖像路徑的字典，每個鍵對應一個類別。
        dst_img_dir (str): 圖像數據目標保存目錄。
        dir (str): 圖像數據的子目錄（例如 'train', 'val', 'test'）。
    Returns:
        None
    """
    for key, values in data_dict.items():
            save_dir = os.path.join(dst_img_dir, dir, key)
            CheckSavePath(save_dir)
            for file_path in values:
                filename = os.path.basename(file_path)
                save_path = os.path.join(save_dir, filename)
                shutil.copy(file_path, save_path)
                print("Image: {} copy to {}".format(file_path, save_path))

def prepare_data(src_img_dir, dst_img_dir, labels_list, split=0.8, limit_count=600):
    """
    準備數據集，包括數據增強、分割和移動數據。

    Params:
        src_img_dir (str): 原始圖像的目錄。
        dst_img_dir (str): 處理後數據的目標目錄。
        labels_list (list): 類別標籤列表。
        split (float): 訓練數據分割比例。
        limit_count (int): 每個類別的最大樣本數量。
    Returns:
        None
    """
    
    src_img_list = ListDir(src_img_dir, fmt=['.bmp', '.png', '.jpg'])
    random.shuffle(src_img_list)
    
    train_data_dict, val_data_dict, test_data_dict = train_val_test_split(src_img_list, labels_list, split_ratio=split)
    train_data_dict = LimitSample(train_data_dict, limit_count=limit_count)
    val_data_dict = LimitSample(val_data_dict, limit_count=round(min([len(value) for value in val_data_dict.values()])*1.25))
    DataAugmentation(train_data_dict, dst_img_dir, up_limit=limit_count)

    if len(train_data_dict):
        MoveData(train_data_dict, dst_img_dir, dir='train')
    if len(val_data_dict):
        MoveData(val_data_dict, dst_img_dir, dir='val')
    if len(test_data_dict):
        MoveData(test_data_dict, dst_img_dir, dir='test')
        
def Histogram(img_dir, labels_list):
    """
    生成每個類別的圖像直方圖。

    Params:
        img_dir (str): 圖像數據的目錄。
        labels_list (list): 類別標籤列表。
    Returns:
        None
    """
    src_img_list = ListDir(img_dir, fmt=['.bmp', '.png', '.jpg'])
    data_dict = dict()
    for label in labels_list:
        data_dict[label] = 0
    for img_path in src_img_list:
        current_label = os.path.basename(os.path.dirname(img_path))
        data_dict[current_label] += 1

    plt.bar(list(data_dict.keys()), data_dict.values())

    plt.xlabel('Class')
    plt.ylabel('Count')
    plt.title('Histogram of Classes')
    
    plt.xticks(list(data_dict.keys()), fontsize=8)

    # 显示直方图
    plt.show()

    

if __name__=='__main__':

    labels_list = [label for label in os.listdir(SRC_DIR)]

    ## Prepare train/test
    # CheckSavePath(src_img_dir)
    # CheckSavePath(src_label_dir)
    CheckSavePath(DATASET_DIR)
    prepare_data(SRC_DIR, DATASET_DIR, labels_list, split=0.8, limit_count=500)
    Histogram(os.path.join(DATASET_DIR, "train"), labels_list)
    Histogram(os.path.join(DATASET_DIR, "val"), labels_list)