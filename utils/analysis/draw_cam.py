#coding: utf-8
import os
from PIL import Image
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt

import torch.autograd as autograd
 
def draw_CAM(model, img, image_path, transform=None, visual_heatmap=False, save_path=None):
    '''
    绘制 Class Activation Map
    :param model: 加载好权重的Pytorch model
    :param img_path: 测试图片路径
    :param save_path: CAM结果保存路径
    :param transform: 输入图像预处理方法
    :param visual_heatmap: 是否可视化原始heatmap（调用matplotlib）
    :return:
    '''
    # 图像加载&预处理
    # img = img[0]
    img = img.unsqueeze(0)
 
    # 获取模型输出的feature/score
    model.eval()
    features = model.forward_features(img).detach().cpu().numpy()
    # for param_tensor in model.state_dict():
    #     print(param_tensor, "\t", model.state_dict()[param_tensor].size())
    fc_weights = model.state_dict()['model.head.fc.weight'].cpu().numpy()
    # class_ = {0: "CP00", 1:"CP03", 2:"CP06" , 3:"CP08", 4:"CP09", 5:"DR02", 6:"IT03", 7:"IT08", 8:"IT09"}
    output = model(img)
 
    # 预测得分最高的那一类对应的输出score
    h_x = torch.nn.functional.softmax(output, dim=1).data.squeeze()  #每个类别对应概率([0.9981, 0.0019])
    probs, idx = h_x.sort(0, True)      #输出概率升序排列
    probs = probs.cpu().numpy()  #[0.9981, 0.0019]
    idx = idx.cpu().numpy()  #[1, 0]
    
    def returnCAM(feature_conv, weight_softmax, class_idx):
        b, c, h, w = feature_conv.shape        #1,2048,7,7
        output_cam = []
        for idx in class_idx:  #输出每个类别的预测效果
            cam = weight_softmax[idx].dot(feature_conv.reshape((c, h*w)))  
            #(1, 2048) * (2048, 7*7) -> (1, 7*7) 
            cam = cam.reshape(h, w)
            cam_img = (cam - cam.min()) / (cam.max() - cam.min())  #Normalize
            cam_img = np.uint8(255 * cam_img)  #Format as CV_8UC1 (as applyColorMap required)
            output_cam.append(cam_img)
        return output_cam
    
    CAMs = returnCAM(features, fc_weights, [idx[0]])
    
    org_img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    org_img = cv2.resize(org_img, (img.shape[3], img.shape[2]))
    CAMs = cv2.resize(CAMs[0], (img.shape[3], img.shape[2]))  # 将热力图的大小调整为与原始图像相同
    CAMs = cv2.applyColorMap(CAMs, cv2.COLORMAP_JET)  # 将热力图应用于原始图像
    superimposed_img = cv2.addWeighted(CAMs, 0.3, org_img, 0.6, 0.0)

    # 可视化原始热力图
    if visual_heatmap:
        plt.matshow(superimposed_img)
        plt.show()
     
    if save_path:
        cv2.imwrite(save_path, superimposed_img)  # 将图像保存到硬盘
    return superimposed_img