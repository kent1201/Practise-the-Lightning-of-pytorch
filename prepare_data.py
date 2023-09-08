import os
import shutil
import random
from tqdm import tqdm

labels_list = ['CP03', 'CP06', 'CP08', 'CP09', 'DR02', 
               'IT03', 'IT07', 'IT08', 'IT09', 
               'PASSCP06', 'PASSDIRTY', 'PASSOTHER', 'PASSOXDATION', 'PASSSCRATCHES', 'SHORTCP06', 'SHORTOTHER']
# labels_map = ["hit", "dirty", "crease", "scratch", "short", "open", "erosion"] # 傑精靈
# labels_map = ["scratch", "crease", "hit", "dirty", "short", "open", "erosion"]
random.seed(19592)

def CheckSavePath(dir_path):
    if not os.path.isdir(dir_path):
        try:
            os.makedirs(dir_path)
        except Exception as ex:
            raise RuntimeError("Create dir error: {}".format(ex))
    return dir_path

def train_test_split(img_list, split_ratio=0.8):
    global labels_list
    data_list = dict()
    train_data_list = dict()
    test_data_list = dict()
    for label in labels_list:
        data_list[label] = list()
        train_data_list[label] = list()
        test_data_list[label] = list()
    for img_path in img_list:
        current_label = os.path.basename(os.path.dirname(img_path))
        data_list[current_label].append(img_path)
    for key in data_list.keys():
        size = int(len(data_list[key]) * split_ratio)
        train_data_list[key] = data_list[key][:size]
        test_data_list[key] = data_list[key][size:]
    return train_data_list, test_data_list

def ListDir(input_dir, fmt):
    if not isinstance(fmt, list):
        fmt = [fmt]
    results_list = list()
    for root, dirs, files in os.walk(input_dir):
        for f in files:
            if os.path.splitext(f)[-1] in fmt:
                results_list.append(os.path.join(root, f))
    return results_list

def prepare_data(src_img_dir, dst_img_dir, split=0.8):
    
    src_img_list = ListDir(src_img_dir, fmt=['.bmp', '.png', '.jpg'])
    random.shuffle(src_img_list)
    
    train_data_list, test_data_list = train_test_split(src_img_list, split_ratio=split)
    
    if len(train_data_list):
        for key, values in train_data_list.items():
            save_dir = os.path.join(dst_img_dir, 'train', key)
            CheckSavePath(save_dir)
            for file_path in values:
                filename = os.path.basename(file_path)
                save_path = os.path.join(save_dir, filename)
                shutil.copy(file_path, save_path)
                print("Image: {} copy to {}".format(file_path, save_path))
    if len(test_data_list):
        for key, values in test_data_list.items():
            save_dir = os.path.join(dst_img_dir, 'test', key)
            CheckSavePath(save_dir)
            for file_path in values:
                filename = os.path.basename(file_path)
                save_path = os.path.join(save_dir, filename)
                shutil.copy(file_path, save_path)
                print("Image: {} copy to {}".format(file_path, save_path))
    

if __name__=='__main__':
    src_img_dir = r'D:\datasets\K2_datasets\CIMS_230829\all'
    dst_img_dir = r'D:\datasets\K2_datasets\CIMS_230829'
    # CheckSavePath(src_img_dir)
    # CheckSavePath(src_label_dir)
    CheckSavePath(dst_img_dir)
    prepare_data(src_img_dir, dst_img_dir, split=0.8)