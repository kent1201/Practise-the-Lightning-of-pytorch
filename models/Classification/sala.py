import torch
import torch.nn as nn


class SALA(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.model_path = args.load_sala_ckpt_path
        self.num_classes = args.num_classes
        assert self.model_path != None, "[SALA] load_ckpt_path is None!!"
        self.model = torch.load(self.model_path)
    
    def forward(self, input):
        output = self.model(input)
        # print("SALA output shape: {}".format(output.shape))
        # print("SALA output 0: {}".format(output[0]))
        return output[:, 0:self.num_classes]

if __name__=='__main__':
    import torch
    sala_model = SALA(r"D:\Users\KentTsai\Documents\ViT_pytorch\AoiColorModel_ver2_2_0.pth")
    # ListModels(r"/mnt/d/Users/KentTsai/Documents/ViT_pytorch/timm_model_list.txt")
    # model = Timm_Vit()
    input = torch.randn([1, 3, 224, 224])
    output = sala_model(input)
    print(output)