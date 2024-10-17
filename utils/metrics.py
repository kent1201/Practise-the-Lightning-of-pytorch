import torch.nn as nn
import numpy as np
from numpy import cov
from scipy.linalg import sqrtm
from numpy import trace
from numpy import iscomplexobj
import torchmetrics

class ClassificationMetrics(nn.Module):
    def __init__(self, num_classes, task="multiclass"):
        super().__init__()
        self.acc = torchmetrics.classification.Accuracy(task=task, num_classes=num_classes)
        self.precision = torchmetrics.classification.Precision(task=task, num_classes=num_classes)
        self.recall = torchmetrics.classification.Recall(task=task, num_classes=num_classes)
    
    def forward(self, preds, target):
        self.acc.update(preds, target)
        self.precision.update(preds, target)
        self.recall.update(preds, target)
        return {"acc": self.acc, "precision": self.precision, "recall": self.recall}
    
    def Compute(self):
        acc_all = self.acc.compute()
        precision_all = self.precision.compute()
        recall_all = self.recall.compute()
        return {"acc": acc_all, "precision": precision_all, "recall": recall_all}
    
def FID_Score(feature1, feature2):
    # calculate mean and covariance statistics
    mu1, sigma1 = feature1.mean(axis=0), cov(feature1, rowvar=False)
    mu2, sigma2 = feature2.mean(axis=0), cov(feature2, rowvar=False)
    # calculate sum squared difference between means
    ssdiff = np.sum((mu1 - mu2)**2.0)
    # calculate sqrt of product between cov
    covmean = sqrtm(sigma1.dot(sigma2))
    # check and correct imaginary numbers from sqrt
    if iscomplexobj(covmean):
        covmean = covmean.real
    # calculate score
    fid = ssdiff + trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid
