import os
import sys
import torch
import torch.nn.functional as F
from torch import optim, nn
from utils.optimizers import NovoGrad
import lightning.pytorch as pl
from utils.optimizers import LionOptimizer
# from timm_vis.methods import grad_cam
import torchmetrics
from torchmetrics.classification import MulticlassConfusionMatrix
from models import *
from matplotlib import pyplot as plt


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
        # print(self.model)
        

         ## Example of input, willn be used in deploy onnx model, if save_onnx is used. 
        self.example_input_array = torch.Tensor(1, self.args.num_channels, self.args.crop_size, self.args.crop_size)

        ## close automatic_optimization if you have many optimizers need to process by yourself.
        # self.automatic_optimization = False
        
        # self.train_metrics = ClassificationMetrics(num_classes=self.label_count, task="multiclass")
        # self.val_metrics = ClassificationMetrics(num_classes=self.label_count, task="multiclass")
        # self.test_metrics = ClassificationMetrics(num_classes=self.label_count, task="multiclass")
        self.train_precision = torchmetrics.Precision(task="multiclass", average='micro', num_classes=self.label_count, num_labels=self.label_count, threshold=0.5)
        self.train_recall = torchmetrics.Recall(task="multiclass", average='micro', num_classes=self.label_count, num_labels=self.label_count, threshold=0.5)
        self.val_precision = torchmetrics.Precision(task="multiclass", average='micro', num_classes=self.label_count, num_labels=self.label_count, threshold=0.5)
        self.val_recall = torchmetrics.Recall(task="multiclass", average='micro', num_classes=self.label_count, num_labels=self.label_count, threshold=0.5)
        self.val_confusion_matrix = MulticlassConfusionMatrix(num_classes=self.label_count)
        self.test_precision = torchmetrics.Precision(task="multiclass", average='micro', num_classes=self.label_count, num_labels=self.label_count, threshold=0.5)
        self.test_recall = torchmetrics.Recall(task="multiclass", average='micro', num_classes=self.label_count, num_labels=self.label_count, threshold=0.5)
        self.test_confusion_matrix = MulticlassConfusionMatrix(num_classes=self.label_count)
        
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
        self.val_confusion_matrix.update(preds, max_targets)
        self.log('val_precision', self.val_precision, on_step=False, on_epoch=True, prog_bar=True, batch_size=self.batch_size)
        self.log('val_recall', self.val_recall, on_step=False, on_epoch=True, prog_bar=True, batch_size=self.batch_size)
        self.log('lr', self.lr, on_step=False, on_epoch=True, prog_bar=True, batch_size=self.batch_size)

        return {"loss": loss, "preds": y, "target": target}

    def on_validation_epoch_end(self):
        ## Calculate metrics
        fig_, ax_ = self.val_confusion_matrix.plot(labels=self.args.labels.split(","))
        fig_.savefig(os.path.join(self.args.save_ckpt_path, "val_confusion_matrix.png"))
        self.val_confusion_matrix.reset()
        plt.close(fig_)

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
        self.test_confusion_matrix.update(preds, max_targets)
        self.log('test_precision', self.test_precision, on_step=False, on_epoch=True, prog_bar=True, batch_size=self.batch_size)
        self.log('test_recall', self.test_recall, on_step=False, on_epoch=True, prog_bar=True, batch_size=self.batch_size)
        
        return {"loss": loss, "preds": y, "target": target}
    
    def on_test_epoch_end(self):
        ## Calculate metrics
        fig_, ax_ = self.test_confusion_matrix.plot(labels=self.args.labels.split(","))
        fig_.savefig(os.path.join(self.args.save_ckpt_path, "test_confusion_matrix.png"))
        plt.close(fig_)

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        # training_step defines the train loop.
        # it is independent of forward
        return self(batch)

    def configure_optimizers(self):
        optimizer = None
        lr_scheduler = None
        group_params = self.parameters()
        if self.args.special_lr_factor:
            params = list(filter(lambda kv: 'head' in kv[0], self.model.named_parameters()))
            base_params = list(filter(lambda kv: 'head' not in kv[0], self.model.named_parameters()))
            params = [item[1] for item in params]
            base_params = [item[1] for item in base_params]
            group_params = [
                {'params': params, 'lr': self.lr*float(self.args.special_lr_factor)},
                {'params': base_params, 'lr': self.lr},
            ]
        if self.args.optimizer == 'Adam':
            if self.args.special_lr_factor:
                optimizer = optim.Adam(group_params, weight_decay=self.args.weight_decay)
            else:
                optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.args.weight_decay)
        elif self.args.optimizer == 'SGD':
            if self.args.special_lr_factor:
                for param in group_params:
                    param['lr'] = param['lr'] * 10
                optimizer = optim.SGD(group_params, momentum=self.args.momentum,  weight_decay=self.args.weight_decay*10)
            else:
                optimizer = optim.SGD(self.parameters(), lr=self.lr*10, momentum=self.args.momentum,  weight_decay=self.args.weight_decay*10)
        elif self.args.optimizer == 'AdamW':
            if self.args.special_lr_factor:
                optimizer = optim.AdamW(group_params, weight_decay=self.args.weight_decay)
            else:
                optimizer = optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.args.weight_decay)
        elif self.args.optimizer == 'NAdam':
            if self.args.special_lr_factor:
                optimizer = optim.NAdam(group_params, weight_decay=self.args.weight_decay)
            else:
                optimizer = optim.NAdam(self.parameters(), lr=self.lr, weight_decay=self.args.weight_decay)
        elif self.args.optimizer == 'Lion':
            ## Suggest the lr smaller 10x or 3x than lr of the Adam
            ## Suggest the batch size of Lion at least 64
            if self.args.special_lr_factor:
                for param in group_params:
                    param['lr'] = param['lr'] * 0.1
                optimizer = LionOptimizer(group_params, weight_decay=self.args.weight_decay)
            else:
                optimizer = LionOptimizer(self.parameters(), lr=self.lr*0.1, weight_decay=self.args.weight_decay)
        elif self.args.optimizer == 'NovoGrad':
            ## Suggest the lr smaller 10x or 3x than lr of the Adam
            ## Suggest the batch size of Lion at least 64
            if self.args.special_lr_factor:
                optimizer = NovoGrad(group_params, weight_decay=self.args.weight_decay)
            else:
                optimizer = NovoGrad(self.parameters(), lr=self.lr, betas=(0.95, 0.98), weight_decay=self.args.weight_decay*0.1)
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
