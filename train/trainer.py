import os
import torch
import torch.nn.functional as F
from torch import optim, nn
import lightning.pytorch as pl
from utils.metrics import ClassificationMetrics
from utils.optimizers import LionOptimizer
import torchmetrics

# define the LightningModule
class ClassifiedTrainer(pl.LightningModule):
    def __init__(self, model, args):
        super().__init__()
        self.save_hyperparameters(ignore=['model'])
        self.model = model
        self.args = args
        self.batch_size = self.args.batch_size
        self.lr = self.args.lr
        self.criterion = nn.CrossEntropyLoss()
        self.label_count = len(self.args.labels.split(","))

         ## Example of input, willn be used in deploy onnx model, if save_onnx is used. 
        self.example_input_array = torch.Tensor(1, 3, args.image_size, args.image_size)

        ## close automatic_optimization if you have many optimizers need to process by yourself.
        # self.automatic_optimization = False
        
        # self.train_metrics = ClassificationMetrics(num_classes=self.label_count, task="multiclass")
        # self.val_metrics = ClassificationMetrics(num_classes=self.label_count, task="multiclass")
        # self.test_metrics = ClassificationMetrics(num_classes=self.label_count, task="multiclass")
        self.train_precision = torchmetrics.Precision(task="multiclass", num_classes=self.label_count)
        self.train_recall = torchmetrics.Recall(task="multiclass", num_classes=self.label_count)
        self.val_precision = torchmetrics.Precision(task="multiclass", num_classes=self.label_count)
        self.val_recall = torchmetrics.Recall(task="multiclass", num_classes=self.label_count)
        self.test_precision = torchmetrics.Precision(task="multiclass", num_classes=self.label_count)
        self.test_recall = torchmetrics.Recall(task="multiclass", num_classes=self.label_count)
        

    def forward(self, input):
        preds = self.model(input)
        return preds

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        input, target = batch
        target = torch.squeeze(target, dim=1)
        y = self.forward(input)
        loss = self.criterion(y, target.float())
        # Logging to TensorBoard (if installed) by default
        self.log("train_loss", loss, prog_bar=True)

        preds = torch.argmax(F.softmax(y.detach(), dim=1), dim=1)
        max_targets = torch.argmax(target, dim=1)
        self.train_precision(preds, max_targets)
        self.train_recall(preds, max_targets)
        self.log('train_precision', self.train_precision, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_recall', self.train_recall, on_step=True, on_epoch=True, prog_bar=True)

        return {"loss": loss, "preds": y, "target": target}

    # def on_train_epoch_end(self):
    #     ## Calculate metrics
    #     self.log('train_acc_epoch', self.acc.compute())
    #     self.log('train_precision_epoch', self.precision.compute())
    #     self.log('train_recall_epoch', self.recall.compute())
    
    def validation_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        input, target = batch
        target = torch.squeeze(target, dim=1)
        y = self.forward(input)
        loss = self.criterion(y, target.float())
        # Logging to TensorBoard (if installed) by default
        self.log("val_loss", loss, prog_bar=True)

        preds = torch.argmax(F.softmax(y.detach(), dim=1), dim=1)
        max_targets = torch.argmax(target, dim=1)
        self.val_precision(preds, max_targets)
        self.val_recall(preds, max_targets)
        self.log('val_precision', self.val_precision, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_recall', self.val_recall, on_step=False, on_epoch=True, prog_bar=True)

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
        input, target = batch
        target = torch.squeeze(target, dim=1)
        y = self.forward(input)
        loss = self.criterion(y, target.float())
        # Logging to TensorBoard (if installed) by default
        self.log("test_loss", loss)

        preds = torch.argmax(F.softmax(y.detach(), dim=1), dim=1)
        max_targets = torch.argmax(target, dim=1)
        self.test_precision(preds, max_targets)
        self.test_recall(preds, max_targets)
        self.log('test_precision', self.test_precision, on_step=False, on_epoch=True, prog_bar=True)
        self.log('test_recall', self.test_recall, on_step=False, on_epoch=True, prog_bar=True)

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
        if self.args.optimizer == 'Adam':
            optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.args.weight_decay)
        elif self.args.optimizer == 'AdamW':
            optimizer = optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.args.weight_decay)
        elif self.args.optimizer == 'Lion':
            ## Suggest the lr smaller 10x or 3x than lr of the Adam
            ## Suggest the batch size of Lion at least 64
            self.lr = self.lr * 0.1
            optimizer = LionOptimizer(self.parameters(), lr=self.lr, weight_decay=self.args.weight_decay)
        else:
            raise ValueError("Optimizer {} does not exist.".format(self.args.optimizer))
        lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.8, patience=10)
        return {"optimizer": optimizer, 
                "lr_scheduler": {"scheduler": lr_scheduler, 
                                 "monitor": "val_loss"}}