from models.Classification.loss import ClassificationLoss
from models.Classification.timm_vit import Timm_Vit
from models.Classification.sala import SALA
from models.Classification.resnet import ResNet18, ResNet50, ResNet101, ResNet152

CLASSIFICATIONMODELLIST = {"Timm_Vit": Timm_Vit, "SALA": SALA, "resnet18": ResNet18, "resnet50": ResNet50, "resnet101": ResNet101, "resnet152": ResNet152}