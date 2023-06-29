from torch import nn
from vit_pytorch.vit_for_small_dataset import ViT

class SmallDataVit(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.model = ViT(
            image_size = args.image_size,
            patch_size = args.patch_size,
            num_classes = args.num_classes,
            dim = args.dim,
            depth = args.depth,
            heads = args.heads,
            mlp_dim = args.mlp_dim,
            dropout = 0.3,
            emb_dropout = 0.3
        )
    def forward(self, x):
        y = self.model(x)
        return y
    
