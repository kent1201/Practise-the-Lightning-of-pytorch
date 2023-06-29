import torch
from vit_pytorch import ViT, SimpleViT
from vit_pytorch.distill import DistillWrapper

def BaseVit(
    image_size = 256,
    patch_size = 32,
    num_classes = 1000,
    dim = 1024,
    depth = 6,
    heads = 16,
    mlp_dim = 2048,
    dropout = 0.1,
    emb_dropout = 0.1
):
    model = ViT(
        image_size = image_size,
        patch_size = patch_size,
        num_classes = num_classes,
        dim = dim,
        depth = depth,
        heads = heads,
        mlp_dim = mlp_dim,
        dropout = dropout,
        emb_dropout = emb_dropout
    )
    return model

def SimpleVit(
    image_size = 256,
    patch_size = 32,
    num_classes = 1000,
    dim = 1024,
    depth = 6,
    heads = 16,
    mlp_dim = 2048
):
    model = ViT(
        image_size = image_size,
        patch_size = patch_size,
        num_classes = num_classes,
        dim = dim,
        depth = depth,
        heads = heads,
        mlp_dim = mlp_dim
    )
    return model

def Distillation(
    teacher,
    student,
    temperature = 3,           # temperature of distillation
    alpha = 0.5,               # trade between main loss and distillation loss
    hard = False               # whether to use soft or hard distillation
):
    models = DistillWrapper(
        student = student,
        teacher = teacher,
        temperature = temperature,           # temperature of distillation
        alpha = alpha,               # trade between main loss and distillation loss
        hard = hard               # whether to use soft or hard distillation
    )
    return models

if __name__=='__main__':

    v = SimpleVit(
        image_size = 256,
        patch_size = 32,
        num_classes = 1000,
        dim = 1024,
        depth = 6,
        heads = 16,
        mlp_dim = 2048,
        dropout = 0.1,
        emb_dropout = 0.1
    )
    img = torch.randn(1, 3, 256, 256)

    preds = v(img) # (1, 1000)