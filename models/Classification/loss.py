import torch
import torch.nn as nn
import torch.nn.functional as F

class ClassificationLoss(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.loss = args.loss
        self.criterion = None
        self.class_weights = args.loss_with_cls_weight
        if self.loss == "CrossEntropy":
            if self.class_weights:
                self.class_weights = [float(item) for item in self.class_weights.split(",")]
                self.criterion = nn.CrossEntropyLoss(weight=torch.Tensor(self.class_weights))
            else:
                self.criterion = nn.CrossEntropyLoss()
        elif self.loss == "Poly1CrossEntropyLoss":
            if self.class_weights:
                self.class_weights = [float(item) for item in self.class_weights.split(",")]
                self.criterion = Poly1CrossEntropyLoss(num_classes=args.num_classes, reduction="mean", weight=torch.Tensor(self.class_weights))
            else:
                self.criterion = Poly1CrossEntropyLoss(num_classes=args.num_classes, reduction="mean")
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
    
## https://github.com/abhuse/polyloss-pytorch/tree/main
class Poly1CrossEntropyLoss(nn.Module):
    def __init__(self,
                 num_classes: int,
                 epsilon: float = 1.0,
                 reduction: str = "none",
                 weight: torch.Tensor = None):
        """
        Create instance of Poly1CrossEntropyLoss
        :param num_classes:
        :param epsilon:
        :param reduction: one of none|sum|mean, apply reduction to final loss tensor
        :param weight: manual rescaling weight for each class, passed to Cross-Entropy loss
        """
        super(Poly1CrossEntropyLoss, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.reduction = reduction
        self.weight = weight
        return

    def forward(self, logits, labels):
        """
        Forward pass
        :param logits: tensor of shape [N, num_classes]
        :param labels: tensor of shape [N]
        :return: poly cross-entropy loss
        """
        labels_onehot = F.one_hot(labels, num_classes=self.num_classes).to(device=logits.device,
                                                                           dtype=logits.dtype)
        pt = torch.sum(labels_onehot * F.softmax(logits, dim=-1), dim=-1)
        CE = F.cross_entropy(input=logits,
                             target=labels,
                             reduction='none',
                             weight=self.weight)
        poly1 = CE + self.epsilon * (1 - pt)
        if self.reduction == "mean":
            poly1 = poly1.mean()
        elif self.reduction == "sum":
            poly1 = poly1.sum()
        return poly1


class Poly1FocalLoss(nn.Module):
    def __init__(self,
                 num_classes: int,
                 epsilon: float = 1.0,
                 alpha: float = 0.25,
                 gamma: float = 2.0,
                 reduction: str = "none",
                 weight: torch.Tensor = None,
                 pos_weight: torch.Tensor = None,
                 label_is_onehot: bool = False):
        """
        Create instance of Poly1FocalLoss
        :param num_classes: number of classes
        :param epsilon: poly loss epsilon
        :param alpha: focal loss alpha
        :param gamma: focal loss gamma
        :param reduction: one of none|sum|mean, apply reduction to final loss tensor
        :param weight: manual rescaling weight for each class, passed to binary Cross-Entropy loss
        :param label_is_onehot: set to True if labels are one-hot encoded
        """
        super(Poly1FocalLoss, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.weight = weight
        self.pos_weight = pos_weight
        self.label_is_onehot = label_is_onehot
        return

    def forward(self, logits, labels):
        """
        Forward pass
        :param logits: output of neural netwrok of shape [N, num_classes] or [N, num_classes, ...]
        :param labels: ground truth tensor of shape [N] or [N, ...] with class ids if label_is_onehot was set to False, otherwise 
            one-hot encoded tensor of same shape as logits
        :return: poly focal loss
        """
        # focal loss implementation taken from
        # https://github.com/facebookresearch/fvcore/blob/main/fvcore/nn/focal_loss.py

        p = torch.sigmoid(logits)

        if not self.label_is_onehot:
            # if labels are of shape [N]
            # convert to one-hot tensor of shape [N, num_classes]
            if labels.ndim == 1:
                labels = F.one_hot(labels, num_classes=self.num_classes)

            # if labels are of shape [N, ...] e.g. segmentation task
            # convert to one-hot tensor of shape [N, num_classes, ...]
            else:
                labels = F.one_hot(labels.unsqueeze(1), self.num_classes).transpose(1, -1).squeeze_(-1)

        labels = labels.to(device=logits.device,
                           dtype=logits.dtype)

        ce_loss = F.binary_cross_entropy_with_logits(input=logits,
                                                     target=labels,
                                                     reduction="none",
                                                     weight=self.weight,
                                                     pos_weight=self.pos_weight)
        pt = labels * p + (1 - labels) * (1 - p)
        FL = ce_loss * ((1 - pt) ** self.gamma)

        if self.alpha >= 0:
            alpha_t = self.alpha * labels + (1 - self.alpha) * (1 - labels)
            FL = alpha_t * FL

        poly1 = FL + self.epsilon * torch.pow(1 - pt, self.gamma + 1)

        if self.reduction == "mean":
            poly1 = poly1.mean()
        elif self.reduction == "sum":
            poly1 = poly1.sum()

        return poly1
