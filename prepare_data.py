import os
import shutil
import random
from tqdm import tqdm
import cv2
import numpy as np

labels_list = ['CP00', 'CP03', 'CP08', 'CP09', 'DR02', 
               'IT03', 'IT07', 'IT08', 'IT09',
               'PASSCP06', 'PASSDIRTY', 'PASSOTHER', 'PASSOXIDATION', 'PASSSCRATCHES', 'SHORTCP06', 'SHORTOTHER']
random.seed(21)

def CheckSavePath(dir_path):
    if not os.path.isdir(dir_path):
        try:
            os.makedirs(dir_path)
        except Exception as ex:
            raise RuntimeError("Create dir error: {}".format(ex))
    return dir_path

def train_val_test_split(img_list, split_ratio=0.8):
    global labels_list
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

def ListDir(input_dir, fmt):
    if not isinstance(fmt, list):
        fmt = [fmt]
    results_list = list()
    for root, dirs, files in os.walk(input_dir):
        for f in files:
            if os.path.splitext(f)[-1] in fmt:
                results_list.append(os.path.join(root, f))
    return results_list

def LimitSample(train_data_dict, limit_count=600):
    for label in train_data_dict.keys():
        if len(train_data_dict[label]) > limit_count:
            limit_data_list = train_data_dict[label]
            random.shuffle(limit_data_list)
            train_data_dict[label] = limit_data_list[:limit_count]
    return train_data_dict

def MoveData(data_dict, dst_img_dir, dir='train'):
    for key, values in data_dict.items():
            save_dir = os.path.join(dst_img_dir, dir, key)
            CheckSavePath(save_dir)
            for file_path in values:
                filename = os.path.basename(file_path)
                save_path = os.path.join(save_dir, filename)
                # try:
                #     org_image = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
                #     org_image = (org_image // 10) * 10
                #     cv2.waitKey(300)
                # except Exception as ex:
                #     print("read {} failed. {}".format(file_path, ex))
                #     continue
                # cv2.imwrite(save_path, org_image, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
                # np.save(save_path, org_image)
                shutil.copy(file_path, save_path)
                print("Image: {} copy to {}".format(file_path, save_path))

def prepare_data(src_img_dir, dst_img_dir, split=0.8, limit_count=600):
    
    src_img_list = ListDir(src_img_dir, fmt=['.bmp', '.png', '.jpg'])
    random.shuffle(src_img_list)
    
    train_data_list, val_data_list, test_data_list = train_val_test_split(src_img_list, split_ratio=split)
    train_data_list = LimitSample(train_data_list, limit_count=limit_count)

    if len(train_data_list):
        MoveData(train_data_list, dst_img_dir, dir='train')
    if len(val_data_list):
        MoveData(val_data_list, dst_img_dir, dir='val')
    if len(test_data_list):
        MoveData(test_data_list, dst_img_dir, dir='test')

    

if __name__=='__main__':

    ## Prepare train/test
    src_img_dir = r'D:\datasets\K2_datasets\CIMS_230907'
    dst_img_dir = r'D:\datasets\K2_datasets\CIMS_230907'
    # CheckSavePath(src_img_dir)
    # CheckSavePath(src_label_dir)
    CheckSavePath(dst_img_dir)
    prepare_data(src_img_dir, dst_img_dir, split=0.8, limit_count=300)

    ## Random sample most count of category.
    # sample_count = 800
    # src_img_dir = r"D:\datasets\K2_datasets\CIMS_230829\train_passother_orgs"
    # dst_img_dir = r'D:\datasets\K2_datasets\CIMS_230829\train\PASSOTHER'
    # src_img_list = ListDir(src_img_dir, fmt=['.bmp', '.png', '.jpg'])
    # random.shuffle(src_img_list)
    # sample_img_list = src_img_list[:sample_count]
    # for img_path in sample_img_list:
    #     dst_path = os.path.join(dst_img_dir, os.path.basename(img_path))
    #     shutil.copy(img_path, dst_path)
    #     print("Sample {} to {}".format(img_path, dst_path))
