import torch

#coding: utf-8
import os
from PIL import Image
import torch
import numpy as np
import cv2
from pathlib import Path
# from rich.progress import track
from tqdm import tqdm

from utils import ProcessConfig, GetTransform, GetDataPath, GetModel, CheckSavePath, Onnx2Pytorch, GetConfigs, Str2List, GetCudaDevice


 
def draw_CAM(model, model_ft, img, img_path, verbose=False, save_path=None):
    '''
    繪製 Class Activation Map
    :param model: 加載好權重的Pytorch model
    :param img_path: 測試圖片路徑
    :param save_path: CAM結果保存路徑
    :param verbose: 是否可視化原始heatmap(調用matplotlib)
    :return:
    :param superimposed_img: 熱力圖
    :output: 模型預測結果
    '''
    img = img.unsqueeze(0)
    
    model.eval()
    # 獲取模型最後一層(Linear)的權重 [num_classes, 2048]
    fc_weights = model.state_dict()['Gemm_output.weight'].cpu().numpy()
    # 獲取模型輸出的 score
    output = model(img)

    # 取得 feature
    features = model_ft(img).detach().cpu().numpy()
 
    # 預測得分最高的那一類對應的輸出 score
    h_x = torch.nn.functional.softmax(output, dim=1).data.squeeze()  # 1xN, N: num_classes
    probs, idx = h_x.sort(0, True)      # 輸出概率升序排列
    probs = probs.cpu().numpy()  # [0.9981, 0.0019, ,,,]
    idx = idx.cpu().numpy()  #[6, 2, ,,,]
    
    def __returnCAM(feature_conv, weight_softmax, class_idx):
        b, c, h, w = feature_conv.shape
        output_cam = []
        for idx in class_idx:  # 輸出每個類別的預測效果
            cam = weight_softmax[idx].dot(feature_conv.reshape((c, h*w)))  
            #(1, 2048) * (2048, 7*7) -> (1, 7*7) 
            cam = cam.reshape(h, w)
            cam_img = (cam - cam.min()) / (cam.max() - cam.min())  #Normalize
            cam_img = np.uint8(255 * cam_img)  #Format as CV_8UC1 (as applyColorMap required)
            output_cam.append(cam_img)
        return output_cam
    
    CAMs = __returnCAM(features, fc_weights, [idx[0]])
    # org_img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    org_img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), -1)
    org_img = cv2.resize(org_img, (img.shape[3], img.shape[2]))
    CAMs = cv2.resize(CAMs[0], (img.shape[3], img.shape[2]))  # 将热力图的大小调整为与原始图像相同
    CAMs = cv2.applyColorMap(CAMs, cv2.COLORMAP_JET)  # 将热力图应用于原始图像
    superimposed_img = cv2.addWeighted(CAMs, 0.3, org_img, 0.6, 0.0)

    # 可视化原始热力图
    if verbose:
        cv2.imshow("superimposed_img", superimposed_img)
        cv2.waitKey(0)
     
    if save_path:
        cv2.imwrite(save_path, superimposed_img)  # 将图像保存到硬盘
    return superimposed_img, output

if __name__=='__main__':

    configs = GetConfigs(r"D:\Users\KentTsai\Documents\ViT_pytorch\K2\ModelB.ini")
    model_dir = configs.get("default", "model_dir")
    model_type = configs.get("default", "model_type")
    mode_type = configs.get("default", "mode_type")
    data_fmt = Str2List(configs.get("default", "data_fmt"))
    root_path = configs.get("01_DrawCAM", "root_path")
    cuda_device = GetCudaDevice(configs.get("default", "device"))
    
    images_dir = os.path.join(root_path, mode_type)

    preprocess_file_path = GetDataPath(model_dir, data_fmt=[".json"])[0]
    model_path = GetDataPath(model_dir, data_fmt=[".onnx"])[0]

    pth_model_path = None
    try:
        pth_model_path = GetDataPath(model_dir, data_fmt=[".pth"])[0]
    except Exception as ex:
        print("No .pth found.")
        pth_model_path = Onnx2Pytorch(model_path)
        
    
    model, model_ft = GetModel(model_path=pth_model_path, model_type=model_type, mode="cams")

    model_ft.to(cuda_device)
    model.to(cuda_device)
    model_ft.eval()
    model.eval()
      
    src_root_dir = images_dir.split(os.sep)[-1]
    src_root_parent_dir = Path(images_dir)
    src_root_parent_dir = src_root_parent_dir.parent.absolute()
    dst_root_door = os.path.join(src_root_parent_dir, src_root_dir+"_CAMs")
    dst_root_door = CheckSavePath(dst_root_door)
    
    # Get transform
    ## Load setting from json file
    process_config = ProcessConfig()
    process_config.LoadJsonFile(preprocess_file_path)
    ## Get preprocess methods from setting
    transform = GetTransform(process_config)
    
    
    # Get images path
    images_path = GetDataPath(images_dir, data_fmt=data_fmt)

    for image_path in tqdm(images_path):
        image = Image.open(image_path)
        label = image_path.split(os.sep)[-2]
        input_tensor = transform(image)
        input_tensor = input_tensor.to(cuda_device)
        superimposed_img, _ = draw_CAM(model=model, model_ft=model_ft, img=input_tensor, img_path=image_path, verbose=False)
        
        save_dir = CheckSavePath(os.path.join(dst_root_door, label))
        save_path = os.path.join(save_dir, os.path.basename(image_path))
        
        cv2.imwrite(save_path, superimposed_img)