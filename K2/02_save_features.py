import os
import torch
import torch.utils.data as data
# from rich.progress import track
from tqdm import tqdm
from utils import ProcessConfig, GetDataPath, GetModel, GetTransform, GetConfigs, LabelStr2Dict, Str2List, GetCudaDevice
from PIL import Image
import numpy as np

class SimpleDataset(data.Dataset):
    def __init__(self, images_dir, transforms=None, data_fmt=[".png", ".bmp", ".jpg", ".JPG", ".JPEG"]):
        self.images_path = list()
        self.data_fmt = data_fmt
        for dirPath, dirNames, fileNames in os.walk(images_dir):
            for f in fileNames:
                if os.path.splitext(f)[1] in self.data_fmt:
                    label = dirPath.split(os.sep)[-1]
                    cams_image_dir = os.path.join(images_dir+"_CAMs", label)
                    self.images_path.append({"image_path": os.path.join(dirPath, f), "label": label, "cams_image_path": os.path.join(cams_image_dir, f)})

        self.transforms = transforms

    def __getitem__(self, index):
        image_path, label, cams_image_path =  self.images_path[index]["image_path"], self.images_path[index]["label"], self.images_path[index]["cams_image_path"]
        image = Image.open(image_path)
        if self.transforms:
            image = self.transforms(image)

        return image, label, image_path, cams_image_path
              
    def __len__(self):
        return len(self.images_path)
    
def GetFeatures(model, model_ft, loader, device, label_dict: dict()):
    features = list()
    for idx, batch in tqdm(enumerate(loader), total=len(loader)):
        image, label, image_path, cams_image_path = batch[0], batch[1], batch[2], batch[3]
        site_predict = os.path.basename(image_path[0]).split("_")[0]
        if site_predict not in label_dict.values():
            site_predict = "NULL"
        image = image.to(device)
        feature, version = None, "Null"
        try:
            version = os.path.basename(image_path[0]).split("_")[1]
        except Exception as ex:
            print("{} cannot get version. {}".format(image_path[0], ex))
        with torch.no_grad():
            output = model(image)
            confidence = float(torch.max(torch.nn.functional.softmax(output, dim=1)).data)
            pred = label_dict[int(torch.argmax(torch.nn.functional.softmax(output, dim=1), dim=1).data)]
            feature = model_ft(image).cpu().detach().numpy()
        features.append({"image_path": image_path, "cams_image_path": cams_image_path, "version": version, "label": label, "predict": pred, "site-predict": site_predict, "confidence": confidence, "feature": feature})
    return features

if __name__=='__main__':
    import gc
    
    configs = GetConfigs(r"D:\Users\KentTsai\Documents\ViT_pytorch\K2\ModelB.ini")
    model_dir = configs.get("default", "model_dir")
    model_type = configs.get("default", "model_type")
    mode_dir = configs.get("default", "mode_type")
    cuda_device = GetCudaDevice(configs.get("default", "device"))
    data_fmt = Str2List(configs.get("default", "data_fmt"))
    label_dict = LabelStr2Dict(configs.get("default", "labels"))
    root_path = configs.get("02_save_features", "root_path")

    
    model_path = GetDataPath(model_dir, data_fmt=[".pth"])[0]
    preprocess_file_path = GetDataPath(model_dir, data_fmt=[".json"])[0]
    save_file_name = "{}_features.npy".format(mode_dir)
    save_file_name = os.path.join(root_path, save_file_name)
    
    model, model_ft = GetModel(model_path=model_path, model_type=model_type, mode="features")
    
    # Get transform
    ## Load setting from json file
    process_config = ProcessConfig()
    process_config.LoadJsonFile(preprocess_file_path)
    ## Get preprocess methods from setting
    transform = GetTransform(process_config)
    
    model_ft.to(cuda_device)
    model.to(cuda_device)
    model_ft.eval()
    model.eval()
    
    
    print("="*10, "Start getting features", "="*10)
    dataset = SimpleDataset(os.path.join(root_path, mode_dir), transforms=transform, data_fmt=data_fmt)
    data_loader = data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=min(os.cpu_count(), 4))
    data_features = GetFeatures(model, model_ft, data_loader, device=cuda_device, label_dict=label_dict)
    np.save(save_file_name, data_features)
    
    del data_features
    gc.collect()
    
    data_features = np.load(save_file_name, allow_pickle=True).tolist()
    for item in data_features:
        image_path = item["image_path"][0]
        cams_image_path = item["cams_image_path"][0]
        version = item["version"]
        label = item["label"][0]
        predict = item["predict"]
        feature = item["feature"]
        confidence = item["confidence"]
        site_predict = item["site-predict"]
        print("image_path", image_path)
        print("cams_image_path", cams_image_path)
        print("version", version)
        print("predict", predict)
        print("confidence", confidence)
        print("site-predict", site_predict)
        print("label", label)
        print("feature", feature.shape)