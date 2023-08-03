from torch import nn
from vit_pytorch.deepvit import DeepViT

class DeepVit(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.model = DeepViT(
            image_size = args.image_size,
            patch_size = args.patch_size,
            num_classes = args.num_classes,
            dim = args.dim,
            depth = args.depth,
            heads = args.heads,
            mlp_dim = args.mlp_dim,
            dropout = 0.1,
            emb_dropout = 0.1
        )
    def forward(self, x):
        y = self.model(x)
        return y