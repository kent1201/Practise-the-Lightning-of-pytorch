from models.Classification.loss import ClassificationLoss
from models.Classification.timm_vit import Timm_Vit
from models.Classification.deep_vit import DeepViT
# from models.Classification.dino_vit import Dino
from models.Classification.simple_vit import SimpleVit
from models.Classification.small_dataset_vit import SmallDataVit
from models.Classification.sala import SALA

CLASSIFICATIONMODELLIST = {"Timm_Vit": Timm_Vit, "DeepViT": DeepViT, "SimpleVit": SimpleVit, "SmallDataVit": SmallDataVit, "SALA": SALA}