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


class Timm_Vit(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.model_name = self.args.timm_model
        self.model = timm.create_model(self.model_name, pretrained=True, num_classes=args.num_classes)
    
    def forward_features(self, x):
        return self.model.forward_features(x)
    
    def forward(self, input):
        output = self.model(input)
        return output

if __name__=='__main__':
    import torch
    ListModels(r"/mnt/d/Users/KentTsai/Documents/ViT_pytorch/timm_model_list.txt")
    # model = Timm_Vit()
    # input = torch.randn([1, 3, 224, 224])
    # output = model(input)
    # print(output.shape)