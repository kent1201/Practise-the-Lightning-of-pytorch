import os
import shutil
from utils import GetDataPath, CheckSavePath, GetConfigs, LabelStr2Dict

def ReplacePath(src_path, dst_path):
    filename = os.path.basename(src_path)
    return os.path.join(dst_path, filename)

if __name__=='__main__':

    configs = GetConfigs(r"D:\Users\KentTsai\Documents\ViT_pytorch\K2\ModelB.ini")
    src_root_dir = configs.get("00_CreateDataset", "src_root_dir")
    dst_root_dir = configs.get("00_CreateDataset", "dst_root_dir")
    label_dict = LabelStr2Dict(configs.get("default", "labels"))
    record_file = configs.get("00_CreateDataset", "record_file")
    model_type = configs.get("default", "model_type")
    mode_type = configs.get("default", "mode_type")
    
    dst_root_dir = CheckSavePath(os.path.join(dst_root_dir, mode_type))
    
    ## Filt and select MDC.txt
    mdc_files_path = list()
    for dirPath, dirNames, fileNames in os.walk(src_root_dir):
        for f in fileNames:
            if f == record_file:
                ## 目錄個位數 22350729001 <= 3 為 modelB, 其餘為 modelA
                model_num = int(os.path.basename(dirPath)) % 100
                if model_type == "A":
                    if model_num >= 4:
                        mdc_files_path.append({"file_path": os.path.join(dirPath, f), "root_path": dirPath})
                elif model_type == "B":
                    if model_num < 4:
                        mdc_files_path.append({"file_path": os.path.join(dirPath, f), "root_path": dirPath})
    print(len(mdc_files_path))

    # Select image_path and label from files
    images_dicts = list()
    for mdc_item in mdc_files_path:
        temp_list = list()
        with open(mdc_item["file_path"], 'r') as f:
            temp_list = f.readlines()
            for line in temp_list:
                item_list = [item.strip() for item in line.split(",")]
                image_path = ReplacePath(item_list[0], mdc_item["root_path"])
                print("{} -> {}".format(item_list[0], image_path))
                label = item_list[1]
                if len(item_list) > 2:
                    site_predict = item_list[2] if item_list[2] in label_dict.values() else "Null"
                else:
                    site_predict = "Null"
                if label in label_dict.values():
                    images_dicts.append({"image_path": image_path, "label": label, "site_predict": site_predict})

    ## Move images to New Dataset structure
    move_record_list = list()
    for data in images_dicts:
        src_image_path = data["image_path"]
        src_image_path_list = src_image_path.split(os.sep)
        save_filename = "{}_{}_{}_{}_{}".format(data["site_predict"], src_image_path_list[-4], src_image_path_list[-3], src_image_path_list[-2], src_image_path_list[-1])
        data["image_path"] = src_image_path
        label_dir = data["label"]
        save_dir = CheckSavePath(os.path.join(dst_root_dir, label_dir))
        save_path = os.path.join(save_dir, save_filename)
        print(save_path)
        shutil.copyfile(src_image_path, save_path)
        move_record_list.append({"From": src_image_path, "Label": label_dir, "Site-Predict": data["site_predict"], "To": save_path})

    
            
    with open(os.path.join(dst_root_dir, "move_record.txt"), "w") as f:
        for item in move_record_list:
            print("File: {} , Label: {}, site-predict: {}, moves to {}\n".format(item["From"], item["Label"], item["Site-Predict"], item["To"]))
            f.write("File: {} , Label: {}, site-predict: {}, moves to {}\n".format(item["From"], item["Label"], item["Site-Predict"], item["To"]))
    
    label_list = list()
    for dict_item in images_dicts:
        print(dict_item["image_path"])
        print(dict_item["label"])
        label_list.append(dict_item["label"])

    label_dict = {i:label_list.count(i) for i in label_list}
    print(label_dict)
    print("total data: {}".format(len(images_dicts)))


    

