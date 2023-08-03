import timm
import torch.nn as nn

def ListModels(output_file_path=""):
    model_list = timm.list_models(pretrained=True)
    for item in model_list:
        print(item)
    if output_file_path:
        with open(output_file_path, 'w') as f:
            f.writelines("{}\n".format(item) for item in model_list)
    return model_list


class SALA(nn.Module):
    def __init__(self, model_path):
        super().__init__()
        self.model_path = model_path
        self.model = torch.load(self.model_path)
    
    def forward(self, input):
        output = self.model(input)
        return output

if __name__=='__main__':
    import torch
    sala_model = SALA(r"D:\Users\KentTsai\Documents\ViT_pytorch\AoiColorModel_ver2_2_0.pth")
    # ListModels(r"/mnt/d/Users/KentTsai/Documents/ViT_pytorch/timm_model_list.txt")
    # model = Timm_Vit()
    input = torch.randn([1, 3, 224, 224])
    output = sala_model(input)
    print(output)