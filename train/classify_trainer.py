import os
import sys
import torch
import torch.nn.functional as F
from torch import optim, nn
import lightning.pytorch as pl
from utils.optimizers import LionOptimizer
# from timm_vis.methods import grad_cam
import torchmetrics
from models import *


# define the LightningModule
class Classifier(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.batch_size = self.args.batch_size
        self.lr = self.args.lr
        self.model = self.args.model
        self.label_count = len(self.args.labels.split(","))
        self.criterion = ClassificationLoss(args=args)
        self.save_name = 0
        self.flooding = float(self.args.flooding)

        
        self.model = CLASSIFICATIONMODELLIST[self.model](args=args)
        

         ## Example of input, willn be used in deploy onnx model, if save_onnx is used. 
        self.example_input_array = torch.Tensor(1, 3, self.args.image_size, self.args.image_size)

        ## close automatic_optimization if you have many optimizers need to process by yourself.
        # self.automatic_optimization = False
        
        # self.train_metrics = ClassificationMetrics(num_classes=self.label_count, task="multiclass")
        # self.val_metrics = ClassificationMetrics(num_classes=self.label_count, task="multiclass")
        # self.test_metrics = ClassificationMetrics(num_classes=self.label_count, task="multiclass")
        self.train_precision = torchmetrics.Precision(task="multiclass", average='micro', num_classes=self.label_count, num_labels=self.label_count, threshold=0.5)
        self.train_recall = torchmetrics.Recall(task="multiclass", average='micro', num_classes=self.label_count, num_labels=self.label_count, threshold=0.5)
        self.val_precision = torchmetrics.Precision(task="multiclass", average='micro', num_classes=self.label_count, num_labels=self.label_count, threshold=0.5)
        self.val_recall = torchmetrics.Recall(task="multiclass", average='micro', num_classes=self.label_count, num_labels=self.label_count, threshold=0.5)
        self.test_precision = torchmetrics.Precision(task="multiclass", average='micro', num_classes=self.label_count, num_labels=self.label_count, threshold=0.5)
        self.test_recall = torchmetrics.Recall(task="multiclass", average='micro', num_classes=self.label_count, num_labels=self.label_count, threshold=0.5)
        
        self.save_hyperparameters(ignore=['model', 'train_precision', 'train_recall', 'val_precision', 'val_recall', 'test_precision', 'test_recall'])

    def forward(self, input):
        preds = self.model(input)
        return preds

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        input, target, input_path = batch
        target = torch.squeeze(target, dim=1)
        y = self.forward(input)
        loss = self.criterion(y, target.float())
        ## Flooding
        if self.flooding:
            loss = (loss-self.flooding).abs() + self.flooding
        # Logging to TensorBoard (if installed) by default
        self.log("train_loss", loss, prog_bar=True, batch_size=self.batch_size)

        preds = torch.argmax(F.softmax(y.detach(), dim=1), dim=1)
        # preds = F.softmax(y.detach(), dim=1)
        max_targets = torch.argmax(target, dim=1)
        self.train_precision(preds, max_targets)
        self.train_recall(preds, max_targets)
        self.log('train_precision', self.train_precision, on_step=True, on_epoch=True, prog_bar=True, batch_size=self.batch_size)
        self.log('train_recall', self.train_recall, on_step=True, on_epoch=True, prog_bar=True, batch_size=self.batch_size)

        return {"loss": loss, "preds": y, "target": target}

    # def on_train_epoch_end(self):
    #     ## Calculate metrics
    #     self.log('train_acc_epoch', self.acc.compute())
    #     self.log('train_precision_epoch', self.precision.compute())
    #     self.log('train_recall_epoch', self.recall.compute())
    
    def validation_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        input, target, input_path = batch
        target = torch.squeeze(target, dim=1)
        y = self.forward(input)
        loss = self.criterion(y, target.float())
        # Logging to TensorBoard (if installed) by default
        self.log("val_loss", loss, prog_bar=True, batch_size=self.batch_size)

        preds = torch.argmax(F.softmax(y.detach(), dim=1), dim=1)
        # preds = F.softmax(y.detach(), dim=1)
        max_targets = torch.argmax(target, dim=1)
        self.val_precision(preds, max_targets)
        self.val_recall(preds, max_targets)
        self.log('val_precision', self.val_precision, on_step=False, on_epoch=True, prog_bar=True, batch_size=self.batch_size)
        self.log('val_recall', self.val_recall, on_step=False, on_epoch=True, prog_bar=True, batch_size=self.batch_size)
        self.log('lr', self.lr, on_step=False, on_epoch=True, prog_bar=True, batch_size=self.batch_size)

        return {"loss": loss, "preds": y, "target": target}
    
    # def validation_step_end(self, outputs):
    #     ## Calculate metrics
    #     results = self.val_metrics(outputs["preds"], outputs["target"])
    #     self.log('val_acc', results["acc"], on_step=True, on_epoch=True)
    #     self.log('val_precision', results["precision"], on_step=True, on_epoch=True)
    #     self.log('val_recall', results["recall"], on_step=True, on_epoch=True)

    # def on_validation_epoch_end(self):
    #     ## Calculate metrics
    #     results = self.val_metrics.Compute()
    #     self.log('val_acc_epoch', results["acc"], on_step=False, on_epoch=True)
    #     self.log('val_precision_epoch', results["precision"], on_step=False, on_epoch=True)
    #     self.log('val_recall_epoch', results["recall"], on_step=False, on_epoch=True)

    def test_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        input, target, input_path = batch
        target = torch.squeeze(target, dim=1)
        y = self.forward(input)
        loss = self.criterion(y, target.float())
        # Logging to TensorBoard (if installed) by default
        self.log("test_loss", loss, batch_size=self.batch_size)

        preds = torch.argmax(F.softmax(y.detach(), dim=1), dim=1)
        # preds = F.softmax(y.detach(), dim=1)
        max_targets = torch.argmax(target, dim=1)
        self.test_precision(preds, max_targets)
        self.test_recall(preds, max_targets)
        self.log('test_precision', self.test_precision, on_step=False, on_epoch=True, prog_bar=True, batch_size=self.batch_size)
        self.log('test_recall', self.test_recall, on_step=False, on_epoch=True, prog_bar=True, batch_size=self.batch_size)

        return {"loss": loss, "preds": y, "target": target}
    
    # def test_step_end(self, outputs):
    #     ## Calculate metrics
    #     results = self.test_metrics(outputs["preds"], outputs["target"])
    #     self.log('test_acc', results["acc"], on_step=True, on_epoch=False)
    #     self.log('test_precision', results["precision"], on_step=True, on_epoch=False)
    #     self.log('test_recall', results["recall"], on_step=True, on_epoch=False)
    
    # def on_test_epoch_end(self):
    #     ## Calculate metrics
    #     results = self.test_metrics.Compute()
    #     self.log('test_acc_epoch', results["acc"], on_step=False, on_epoch=True)
    #     self.log('test_precision_epoch', results["precision"], on_step=False, on_epoch=True)
    #     self.log('test_recall_epoch', results["recall"], on_step=False, on_epoch=True)

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        # training_step defines the train loop.
        # it is independent of forward
        return self(batch)

    def configure_optimizers(self):
        optimizer = None
        lr_scheduler = None
        if self.args.optimizer == 'Adam':
            optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.args.weight_decay)
        elif self.args.optimizer == 'SGD':
            optimizer = optim.SGD(self.parameters(), lr=self.lr*10, momentum=self.args.momentum,  weight_decay=self.args.weight_decay*10)
        elif self.args.optimizer == 'AdamW':
            optimizer = optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.args.weight_decay)
        elif self.args.optimizer == 'NAdam':
            optimizer = optim.NAdam(self.parameters(), lr=self.lr, weight_decay=self.args.weight_decay)
        elif self.args.optimizer == 'Lion':
            ## Suggest the lr smaller 10x or 3x than lr of the Adam
            ## Suggest the batch size of Lion at least 64
            self.lr = self.lr * 0.1
            optimizer = LionOptimizer(self.parameters(), lr=self.lr, weight_decay=self.args.weight_decay)
        else:
            raise ValueError("Optimizer {} does not exist.".format(self.args.optimizer))
        
        if self.args.scheduler == 'ReduceLROnPlateau':
            lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=self.args.lr_factor, patience=10)
        elif self.args.scheduler == 'StepLR':
            lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=self.args.lr_step_size, gamma=self.args.lr_factor)
        elif self.args.scheduler == 'ExponentialLR':
            lr_scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=self.args.lr_factor)
        elif self.args.scheduler == 'CosineAnnealingLR':
            lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.args.lr_cycle)
        elif self.args.scheduler == 'CosineAnnealingWarmRestarts':
            lr_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=self.args.lr_cycle, T_mult=1)
        else:
            raise ValueError("LR_scheduler {} does not exist.".format(self.args.scheduler))
        return {"optimizer": optimizer, 
                "lr_scheduler": {"scheduler": lr_scheduler, 
                                 "monitor": "val_loss"}}