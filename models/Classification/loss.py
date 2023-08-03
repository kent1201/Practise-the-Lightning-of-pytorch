import torch
import torch.nn as nn
import torch.nn.functional as F

class ClassificationLoss(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.loss = args.loss
        self.criterion = None
        if self.loss == "CrossEntropy":
            self.criterion = nn.CrossEntropyLoss()
        elif self.loss == "Focal":
            self.criterion = FocalLoss()
        elif self.loss == "SigmoidFocal":
            self.criterion = SigmoidFocalLoss(reduction="mean")
        else:
            raise ValueError("{} not superised".format(self.loss))
    
    def forward(self, input, target):
        output = self.criterion(input, target)
        return output
    
class SmoothCrossEntropy(torch.nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, pred, gold, smoothing=0.1):
        n_class = pred.size(1)

        one_hot = torch.full_like(pred, fill_value=smoothing / (n_class - 1))
        one_hot.scatter_(dim=1, index=gold.unsqueeze(1), value=1.0 - smoothing)
        log_prob = F.log_softmax(pred, dim=1)

        return F.kl_div(input=log_prob, target=one_hot, reduction='none').sum(-1).mean()


class FocalLoss(nn.Module):
    def __init__(self, gama=1.5, alpha=0.25, weight=None, reduction="mean") -> None:
        super().__init__() 
        self.loss_fcn = nn.CrossEntropyLoss(weight=weight, reduction=reduction)
        self.gama = gama 
        self.alpha = alpha 
        self.reduction = reduction

    def forward(self, pre, target):
        logp = self.loss_fcn(pre, target)
        p = torch.exp(-logp) 
        loss = (1-p)**self.gama * self.alpha * logp
        return loss.mean()
    
class SigmoidFocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=0.25, reduction="none") -> torch.Tensor:
        super().__init__() 
        self.loss_fcn = nn.BCEWithLogitsLoss(reduction="none")
        self.sigmoid = torch.nn.Sigmoid()
        self.gamma = gamma 
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, input, target):
        p = self.sigmoid(input)
        ce_loss = self.loss_fcn(input, target)
        p_t = p * target + (1 - p) * (1 - target)
        loss = ce_loss * ((1 - p_t) ** self.gamma)
        if self.alpha > 0:
            alpha_t = self.alpha * target + (1 - self.alpha) * (1 - target)
            loss = alpha_t * loss
        if self.reduction == "none":
            pass
        elif self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()
        else:
            raise ValueError(
                f"Invalid Value for arg 'reduction': '{self.reduction} \n Supported reduction modes: 'none', 'mean', 'sum'"
            )
        return loss
