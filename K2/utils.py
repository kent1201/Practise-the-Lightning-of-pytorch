import os
import copy
import json
import configparser
import torch
import torch.utils.data as data
from PIL import Image
from torchvision import transforms as T
import onnx
from onnx2pytorch import ConvertModel
from datetime import datetime
# import albumentations as A 

def GetNowTime_yyyymmddhhMMss():
    now_time = datetime.now().strftime("%Y%m%d%H%M%S%f")[:-3]
    return now_time    

def GetCudaDevice(cuda_id):
    return "cuda:{}".format(cuda_id if torch.cuda.is_available() else -1)

def GetConfigs(congig_file_path):
    config = configparser.ConfigParser()
    config.read(congig_file_path, encoding='UTF-8')
    return config

def LabelStr2Dict(str, sep=","):
    label_list = Str2List(str, sep)
    label_dict = dict()
    for index, item in enumerate(label_list):
        label_dict[index] = item
    return label_dict

def Str2List(str, sep=","):
    return [item.strip() for item in str.split(sep)]


def CheckSavePath(dir_path):
    if not os.path.isdir(dir_path):
        try:
            os.makedirs(dir_path)
        except Exception as ex:
            print("Create dir error: {}".format(ex))
    return dir_path

def GetDataPath(data_dir, data_fmt=[".png", ".bmp", ".jpg", ".JPG", ".JPEG"]):
    """
    data_dir(str): images directory path
    data_fmt(list): the data format want to get. default: [".png", ".bmp", ".jpg", ".JPG", ".JPEG"]
    """
    data_path = list()
    for dirPath, dirNames, fileNames in os.walk(data_dir):
        for f in fileNames:
            if os.path.splitext(f)[1] in data_fmt:
                data_path.append(os.path.join(dirPath, f))
    return data_path

class InferenceDataset(data.Dataset):
    def __init__(self, images_dir, transforms=None):
        self.images_path = list()
        self.data_fmt = [".png", ".bmp", ".jpg", ".JPG", ".JPEG"]
        for dirPath, dirNames, fileNames in os.walk(images_dir):
            for f in fileNames:
                if os.path.splitext(f)[1] in self.data_fmt:
                    self.images_path.append(os.path.join(dirPath, f))

        self.transforms = transforms

    def __getitem__(self, index):
        image_path =  self.images_path[index]
        image = Image.open(image_path)
        if self.transforms:
            # image = self.transforms(image=np.asarray(image))["image"]
            image = self.transforms(image)

        return image
              
    def __len__(self):
        return len(self.images_path)
    
class PreprocessPara:
    def __init__(self):
        self.normalize    = {"switch": False, "mode": 'ImageNet', "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}
        self.resize       = {"switch": False, "imageSize":[224, 224], "interpolation": 'BILINEAR'}
        self.centerCrop   = {"switch": False, "size": [1, 1]}
        self.pad          = {"switch": False, "padding": [0, 0, 0, 0], "fill": [0, 0, 0], "paddingMode": 'constant'}
        self.gaussianBlur = {"switch": False, "kernelSize": [1, 1], "sigma": 1}
        self.brightness   = {"switch": False, "brightness": 1}
        self.contrast     = {"switch": False, "contrast": 1}
        self.saturation   = {"switch": False, "saturation": 1}
        self.hue          = {"switch": False, "hue": 0}
        self.batchSize    = {"switch": False, "batchSize": 1}
        self.para_map = {'normalize': self.normalize, 
                            'resize': self.resize,
                            'centerCrop': self.centerCrop,
                            'pad': self.pad,
                            'gaussianBlur': self.gaussianBlur,
                            'brightness': self.brightness,
                            'contrast': self.contrast,
                            'saturation': self.saturation,
                            'hue': self.hue,
                            'batchSize': self.batchSize}
        
        self.normalization_map = self.__NormalMap()
    
    def __NormalMap(self):
        return {'ImageNet': {'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]},
                # 'CIFAR10': {'mean': [0.49139968, 0.48215827, 0.44653124], 'std': [0.24703233, 0.24348505, 0.26158768]},
                'CIFAR10': {'mean': [0.4914, 0.4822, 0.4465], 'std': [0.2023, 0.1994, 0.201]},
                'MNIST': {'mean': [0.1307, 0.1307, 0.1307], 'std': [0.3081, 0.3081, 0.3081]},
                'CalculateFromData': {'mean': [0.5, 0.5, 0.5], 'std': [0.5, 0.5, 0.5]},
                'UserInput': {'mean': [0.5, 0.5, 0.5], 'std': [0.5, 0.5, 0.5]}
                }


class ProcessConfig:
    def __init__(self, logger=None):
        self.Data = None
        self.preprocess_paras = PreprocessPara()
        self.logger = logger
        
    
    def LoadJsonFile(self, json_file):
        with open(json_file, 'r') as f:
            self.Data = json.load(f)
    
    def GetPreprocessMethods(self):
        for item in self.Data['ConfigPreprocess']['PreprocessPara']:
            # print("Before: {}\t{}".format(item, self.preprocess_paras.para_map[item]))
            for k, v in self.Data['ConfigPreprocess']['PreprocessPara'][item].items():
                self.preprocess_paras.para_map[item][k] = v
            if self.logger:
                self.logger.info("{}\t{}".format(item, self.preprocess_paras.para_map[item]))
            else:
                print("{}\t{}".format(item, self.preprocess_paras.para_map[item]))

        if self.preprocess_paras.para_map['normalize']['mode'] in ['ImageNet', 'CIFAR 10', 'MNIST']:
                self.preprocess_paras.para_map['normalize']['mean'] == self.preprocess_paras.normalization_map[self.preprocess_paras.para_map['normalize']['mode']]['mean']
                self.preprocess_paras.para_map['normalize']['std'] == self.preprocess_paras.normalization_map[self.preprocess_paras.para_map['normalize']['mode']]['std']
            


def GetTransform(process_config):

    def RESIZE_MAP():
        return {'BILINEAR': T.InterpolationMode.BILINEAR,
            'NEAREST': T.InterpolationMode.NEAREST,
            'BICUBIC': T.InterpolationMode.BICUBIC,
            'BOX': T.InterpolationMode.BOX,
            'HAMMING': T.InterpolationMode.HAMMING,
            'LANCZOS': T.InterpolationMode.LANCZOS
            }
    
    reszie_map = RESIZE_MAP()

    transforms_images = None

    ## Get config from json data
    process_config.GetPreprocessMethods()

    ## According to the settings, add each setting into module
    modules = []
    if process_config.preprocess_paras.resize['switch']:
        modules.append(T.Resize(size=process_config.preprocess_paras.resize['imageSize'], interpolation=reszie_map[process_config.preprocess_paras.resize['interpolation']]))
    if process_config.preprocess_paras.centerCrop['switch']:
        modules.append(T.CenterCrop(size=process_config.preprocess_paras.centerCrop['size']))
    if process_config.preprocess_paras.pad['switch']:
        # print("{}\t{}".format(type(process_config.preprocess_paras.pad['paddingMode']), process_config.preprocess_paras.pad['paddingMode']))
        modules.append(T.Pad(padding=process_config.preprocess_paras.pad['padding'], fill=process_config.preprocess_paras.pad['fill'][0], padding_mode=process_config.preprocess_paras.pad['paddingMode']))
    if process_config.preprocess_paras.brightness['switch'] or process_config.preprocess_paras.contrast['switch'] or process_config.preprocess_paras.saturation['switch'] or process_config.preprocess_paras.hue['switch']:
        modules.append(T.ColorJitter(brightness=(process_config.preprocess_paras.brightness['brightness'], process_config.preprocess_paras.brightness['brightness']), 
                                     contrast=(process_config.preprocess_paras.contrast['contrast'], process_config.preprocess_paras.contrast['contrast']), 
                                     saturation=(process_config.preprocess_paras.saturation['saturation'], process_config.preprocess_paras.saturation['saturation']), 
                                     hue=(process_config.preprocess_paras.hue['hue'], process_config.preprocess_paras.hue['hue'])))
    if process_config.preprocess_paras.gaussianBlur['switch']:
        modules.append(T.GaussianBlur(kernel_size=process_config.preprocess_paras.gaussianBlur['kernelSize'], sigma=process_config.preprocess_paras.gaussianBlur['sigma']))
    if process_config.preprocess_paras.normalize['switch']:
        # modules.append(T.ConvertImageDtype(tr_float))
        modules.append(T.ToTensor())
        modules.append(T.Normalize(mean=process_config.preprocess_paras.normalize['mean'], std=process_config.preprocess_paras.normalize['std']))
    
    # transforms_images = nn.Sequential(*modules)
    transforms_images = T.Compose(modules)
    
    return transforms_images

def Onnx2Pytorch(onnx_file_path: str) -> str:
    mp = onnx.load_model(onnx_file_path)
    mp.graph.input[0].type.tensor_type.shape.dim[0].dim_param = 'None'
    mp.graph.output[0].type.tensor_type.shape.dim[0].dim_param = 'None'
    pytorch_model = ConvertModel(mp, experimental=True)
    save_filename = os.path.splitext(os.path.basename(onnx_file_path))[0]+".pth"
    save_dir = os.path.abspath(os.path.join(onnx_file_path, os.pardir))
    save_path = os.path.join(save_dir, save_filename)
    torch.save(pytorch_model, save_path)
    return save_path


def _set_module(model, submodule_key, module):
    tokens = submodule_key.split('.')
    sub_tokens = tokens[:-1]
    cur_mod = model
    for s in sub_tokens:
        cur_mod = getattr(cur_mod, s)
    setattr(cur_mod, tokens[-1], module)

def GetModel(model_path, model_type="B", mode="cams"):
    assert mode in ["cams", "features"], "mode in [cams|features]"
    model = torch.load(model_path)
    model_ft = copy.deepcopy(model)
    for name, module in model_ft.named_modules():
        # if model_type == "B":
        #     if mode == "cams":
        #         if name in ['GlobalAveragePool_onnx::Flatten_493', 'Flatten_onnx::Gemm_494', 'Gemm_output']:
        #             _set_module(model_ft, name, torch.nn.Identity())
        #     elif mode == "features":
        #         if name in ['Gemm_output']:
        #             _set_module(model_ft, name, torch.nn.Identity())
        # elif model_type == "A":
        #     if mode == "cams":
        #         if name in ['GlobalAveragePool_onnx::Flatten_652', 'Flatten_onnx::Gemm_653', 'Gemm_output']:
        #             _set_module(model_ft, name, torch.nn.Identity())
        #     elif mode == "features":
        #         if name in ['Gemm_output']:
        #             _set_module(model_ft, name, torch.nn.Identity())
        if mode == "cams":
            if name in ['GlobalAveragePool_onnx::Flatten_493', 'Flatten_onnx::Gemm_494', 'GlobalAveragePool_onnx::Flatten_652', 'Flatten_onnx::Gemm_653', 'Gemm_output']:
                _set_module(model_ft, name, torch.nn.Identity())
        elif mode == "features":
            if name in ['Gemm_output']:
                _set_module(model_ft, name, torch.nn.Identity())
    return model, model_ft

if __name__=='__main__':

    model_path = r"D:\Users\KentTsai\Documents\ViT_pytorch\K2\AI_Model\Model_A_v8.5\BestOnnx.onnx"
    # model_path = r"D:\Users\KentTsai\Documents\ViT_pytorch\K2\AI_Model\Model_B_v2.2\BestOnnx.onnx"
    pth_model_path = Onnx2Pytorch(model_path)
    

    model = torch.load(pth_model_path)

    ## optional (show each module name in model)
    for name, module in model.named_modules():
        print(name, '\t', module)

    
