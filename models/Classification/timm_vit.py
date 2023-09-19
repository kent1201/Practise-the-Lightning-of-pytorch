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
    def __init__(self, args=None):
        super().__init__()
        self.args = args
        self.model_name = self.args.timm_model
        self.dropout_rate = 0.0
        if self.args.dropout:
            self.dropout_rate = float(self.args.dropout)
        self.model = timm.create_model(self.model_name, 
                                       pretrained=True, 
                                       num_classes=args.num_classes, 
                                       drop_rate=self.dropout_rate)
        # if self.model_name == "convnextv2_tiny.fcmae_ft_in1k":
        #     self.model.head.fc = nn.Linear(in_features=768, out_features=384, bias=True)
        #     self.relu = nn.ReLU(inplace=True)
        #     self.fc2 = nn.Linear(in_features=384, out_features=args.num_classes, bias=True)

    
    def forward_features(self, x):
        return self.model.forward_features(x)
    
    def forward(self, input):
        output = self.model(input)
        # if self.model_name == "convnextv2_tiny.fcmae_ft_in1k":
        #     output = self.relu(output)
        #     output = self.fc2(output)
        return output

if __name__=='__main__':
    import torch
    ListModels(r"D:\Users\KentTsai\Documents\ViT_pytorch\timm_model_list.txt")
    # model = Timm_Vit()
    
    # print(model)
    # input = torch.randn([1, 3, 224, 224])
    # output = model(input)
    # print(output.shape)