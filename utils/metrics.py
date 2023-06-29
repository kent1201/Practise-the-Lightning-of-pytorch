import torch.nn as nn
# import our library
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
