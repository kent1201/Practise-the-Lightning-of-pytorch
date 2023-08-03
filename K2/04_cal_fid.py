import os
import numpy as np
from numpy import cov
from scipy import linalg
from PIL import Image
from scipy.linalg import sqrtm
from numpy import trace
from numpy import iscomplexobj
from utils import GetDataPath, GetConfigs, Str2List
    
def FID_Score(feature1, feature2):
    # calculate mean and covariance statistics
    mu1, sigma1 = feature1.mean(axis=0), cov(feature1, rowvar=False)
    mu2, sigma2 = feature2.mean(axis=0), cov(feature2, rowvar=False)
    eps = 1e-6

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'
    
    # calculate sum squared difference between means
    ssdiff = mu1 - mu2
    
    # Product might be almost singular
    # calculate sqrt of product between cov
    covmean = linalg.sqrtm(sigma1.dot(sigma2))
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
    
    # Numerical error might give slight imaginary component
    # check and correct imaginary numbers from sqrt
    if iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real
    
    # calculate score
    tr_covmean = np.trace(covmean)
    fid = (ssdiff.dot(ssdiff) + trace(sigma1)+ trace(sigma2) - 2 * tr_covmean)
    return fid

def GetFeatures(file_path):
    features_list = list()
    features_info = np.load(file_path, allow_pickle=True).tolist()
    for item in features_info:
        features_list.append(item["feature"])
    return np.asarray(features_list).squeeze()

if __name__=='__main__':

    import gc

    configs = GetConfigs(r"D:\Users\KentTsai\Documents\ViT_pytorch\K2\ModelA.ini")
    model_dir = configs.get("default", "model_dir")
    model_type = configs.get("default", "model_type")
    data_fmt = Str2List(configs.get("default", "data_fmt"))
    root_path = configs.get("04_cal_fid", "root_path")
    mode_type1 = configs.get("04_cal_fid", "mode_type1")
    mode_type2 = configs.get("04_cal_fid", "mode_type2")

    feature1_path = os.path.join(root_path, "{}_features.npy".format(mode_type1))
    feature2_path = os.path.join(root_path, "{}_features.npy".format(mode_type2))
    
    
    print("="*10, "Start getting feature 1", "="*10)
    features1 = GetFeatures(feature1_path)

    print("="*10, "Start getting feature 2", "="*10)
    features2 = GetFeatures(feature2_path)
    
    print("="*10, "Start calculating FID Score", "="*10)
    fid_score = FID_Score(features1, features2)
    print("FID score: {}".format(fid_score))
