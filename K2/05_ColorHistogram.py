import torch

#coding: utf-8
import os
import torch
import numpy as np
import cv2
from pathlib import Path
from rich.progress import track
from utils import ProcessConfig, GetDataPath, CheckSavePath, GetConfigs, Str2List
import albumentations as A
from skimage import exposure
from skimage.exposure import match_histograms
import matplotlib.pyplot as plt

def ImagesHistogram(images_path):
    nb_bins = 256
    count_r = np.zeros(nb_bins)
    count_g = np.zeros(nb_bins)
    count_b = np.zeros(nb_bins)
    for image_path in images_path: 
        image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED) 
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        hist_r, bin_edges_r = np.histogram(image[0], bins=nb_bins, range=[0, 255])
        hist_g, bin_edges_g = np.histogram(image[1], bins=nb_bins, range=[0, 255])
        hist_b, bin_edges_b = np.histogram(image[2], bins=nb_bins, range=[0, 255])
        count_r += hist_r
        count_g += hist_g
        count_b += hist_b
    count_r = count_r / count_r.max()
    count_g = count_g / count_g.max()
    count_b = count_b / count_b.max()

    bins = bin_edges_r
    return bins, count_r, count_g, count_b


if __name__=='__main__':
    source_images_dir = r"D:\datasets\K2_datasets\CIMS_ModelA_v8_5\Blind_test_modelA"
    reference_images_dir = r"D:\datasets\K2_datasets\CIMS_ModelA_v8_5\Train"

    source_images_path = GetDataPath(source_images_dir)
    reference_images_path = GetDataPath(reference_images_dir)
    
    s_bins, s_count_r, s_count_g, s_count_b = ImagesHistogram(source_images_path)
    r_bins, r_count_r, r_count_g, r_count_b = ImagesHistogram(reference_images_path)

    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(8, 8))

    axes[0, 0].bar(s_bins[:-1], s_count_r, color='r', alpha=0.33)
    axes[0, 1].bar(s_bins[:-1], s_count_g, color='g', alpha=0.33)
    axes[0, 2].bar(s_bins[:-1], s_count_b, color='b', alpha=0.33)
    
    axes[1, 0].bar(r_bins[:-1], r_count_r, color='r', alpha=0.33)
    axes[1, 1].bar(r_bins[:-1], r_count_g, color='g', alpha=0.33)
    axes[1, 2].bar(r_bins[:-1], r_count_b, color='b', alpha=0.33)
    
    axes[0, 0].set_ylabel("Source images")
    axes[1, 0].set_ylabel("Reference images")
    axes[0, 0].set_title('Red')
    axes[0, 1].set_title('Green')
    axes[0, 2].set_title('Blue')
    plt.tight_layout()
    plt.show()




