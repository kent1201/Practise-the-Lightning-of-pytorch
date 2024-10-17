import json
import os
import shutil
import random
from tqdm import tqdm
import numpy as np
# import albumentations as A
from PIL import Image
# from matplotlib import pyplot as plt

from common_data import transform, SRC_DIR, DATASET_DIR
from utils.utils import ListDir, CheckSavePath

random.seed(42)

def save_to_json(data_dict, json_path):
    """
    將劃分好的數據集保存到 JSON 文件。

    Params:
        data_dict (dict): 包含圖像路徑和標籤的字典。
        json_path (str): JSON 文件的保存路徑。
    Returns:
        None
    """
    with open(json_path, 'w') as f:
        json.dump(data_dict, f)

def train_val_test_split(img_list, labels_list, split_ratio=0.8):
    """
    修改後的數據集劃分，並存儲到 JSON 文件中。
    """
    data_dict = dict()
    train_data_dict = dict()
    val_data_dict = dict()
    test_data_dict = dict()

    for label in labels_list:
        data_dict[label] = list()
        train_data_dict[label] = list()
        val_data_dict[label] = list()
        test_data_dict[label] = list()

    # 生成每個類別的圖像路徑列表
    for img_path in img_list:
        current_label = os.path.basename(os.path.dirname(img_path))
        data_dict[current_label].append(img_path)

    # 劃分訓練集、驗證集和測試集
    for key in data_dict.keys():
        size = int(len(data_dict[key]) * split_ratio)
        val_test_size = int((len(data_dict[key]) - size) * 0.5)
        train_data_dict[key] = data_dict[key][:size]
        val_data_dict[key] = data_dict[key][size:size+val_test_size]
        test_data_dict[key] = data_dict[key][size+val_test_size:]

    return train_data_dict, val_data_dict, test_data_dict

def DataAugmentation(data_dict, dst_img_dir, up_limit=600, aug_ratio=0.3, mode='train'):
    """
    對需要增強的圖像進行增強，並將增強的圖像路徑加入到 train.json 中。
    """
    augmented_data = {}

    for key, value in data_dict.items():
        if len(value) < up_limit:
            min_generate_iter = min(up_limit - len(value), int(len(value) * aug_ratio))
            random.shuffle(value)
            print("Preparing the augmentation of {} data...".format(key))
            
            for new_image_index in tqdm(range(0, min_generate_iter)):
                org_image = Image.open(value[new_image_index])
                new_image = Image.fromarray(transform(image=np.asarray(org_image))["image"])
                new_img_dir = os.path.join(dst_img_dir, mode, key)
                CheckSavePath(new_img_dir)
                
                new_image_path = os.path.join(new_img_dir, "aug_" + os.path.basename(value[new_image_index]))
                new_image.save(new_image_path)

                # 將增強後的圖像路徑添加到 augmented_data
                if key not in augmented_data:
                    augmented_data[key] = []
                augmented_data[key].append(new_image_path)

    for key, value in augmented_data.items():
        if key in data_dict:
            data_dict[key] += value

    print("Data augmentation complete and updated in train data.")
    return data_dict

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

    train_json_file = os.path.join(dst_img_dir, 'train.json')
    val_json_file = os.path.join(dst_img_dir, 'val.json')
    test_json_file = os.path.join(dst_img_dir, 'test.json')
    
    train_data_dict, val_data_dict, test_data_dict = train_val_test_split(src_img_list, labels_list, split_ratio=split)
    
    train_data_dict = LimitSample(train_data_dict, limit_count=limit_count)
    val_data_dict = LimitSample(val_data_dict, limit_count=round(min([len(value) for value in val_data_dict.values()])*1.25))
    train_data_dict = DataAugmentation(train_data_dict, dst_img_dir, up_limit=limit_count, mode="aug")

    # 將劃分結果保存到 JSON 文件
    save_to_json(train_data_dict, train_json_file)
    save_to_json(val_data_dict, val_json_file)
    save_to_json(test_data_dict, test_json_file)

def load_data_from_json(json_path):
    """
    從 JSON 文件中加載圖像路徑和標籤。

    Params:
        json_path (str): JSON 文件的路徑。
    Returns:
        data_dict (dict): 包含圖像路徑和標籤的字典。
    """
    with open(json_path, 'r') as f:
        data_dict = json.load(f)
    return data_dict

if __name__=='__main__':

    labels_list = [label for label in os.listdir(SRC_DIR)]

    ## Prepare train/test
    # CheckSavePath(src_img_dir)
    # CheckSavePath(src_label_dir)
    CheckSavePath(DATASET_DIR)
    prepare_data(SRC_DIR, DATASET_DIR, labels_list, split=0.8, limit_count=500)
