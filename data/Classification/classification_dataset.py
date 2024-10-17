import os
import torch
import lightning.pytorch as pl

from torch.utils.data.dataset import Subset, ConcatDataset

from sklearn.model_selection import KFold
from data.Classification import compute_mean_std, DATASETS

# class CustomSubset(Subset):
#     def __init__(self, dataset, indices):
#         super().__init__(dataset, indices)

#     def __getitem__(self, idx): #同时支持索引访问操作
#         image, target = self.dataset[self.indices[idx]]      
#         return image, target

#     def __len__(self): # 同时支持取长度操作
#         return len(self.indices)



class ClassificationDataset(pl.LightningDataModule):

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.k_folds = int(self.args.k_folds) if isinstance(self.args.k_folds, str) else None
        self.num_splits = int(self.args.num_splits) if isinstance(self.args.num_splits, str) else None
        if self.k_folds and self.num_splits:
            assert 1 <= self.k_folds <= self.num_splits, "incorrect fold number"
            self.kf = KFold(n_splits=self.num_splits, shuffle=True) # , random_state=self.args.random_seed)
        
        self.data_dir = self.args.root_path
        self.batch_size = self.args.batch_size
        self.project = self.args.project
        self.data_fmt = [".png", ".bmp", ".jpg", ".JPG", ".JPEG"]
        # if self.args.compute_mean_std:
        #     self.mean, self.std = compute_mean_std(os.path.join(self.data_dir, 'train'), self.args.load_size, self.data_fmt)
        # else:
        self.mean = [0.5667182803153992, 0.5667171478271484, 0.5666833519935608]
        self.std = [0.15075330436229706, 0.14988024532794952, 0.1496531218290329]
        self.save_hyperparameters()
        
    
    ## Setup is prepare for every GPU
    def setup(self, stage='fit'):
        if stage == "fit":
            self.train_dataset = DATASETS[self.project](self.args, "train")
            self.val_dataset = DATASETS[self.project](self.args, "val")
        elif stage == "train":
            self.train_dataset = DATASETS[self.project](self.args, "train")
        elif stage == "valid":
            self.val_dataset = DATASETS[self.project](self.args, "val")
        elif stage == "test":
            self.test_dataset = DATASETS[self.project](self.args, "test")
        
        self.train_fold, self.val_fold = None, None
        if self.k_folds and stage !="test":
            self.concat_train_dataset = ConcatDataset([self.train_dataset, self.val_dataset])
            all_splits = [k for k in self.kf.split(self.concat_train_dataset)]
            train_indexes, val_indexes = all_splits[self.k_folds]
            train_indexes, val_indexes = train_indexes.tolist(), val_indexes.tolist()
            self.train_fold, self.val_fold = Subset(self.concat_train_dataset, train_indexes), Subset(self.concat_train_dataset, val_indexes)

            train_data_dict, val_data_dict = self.train_dataset.data_dict, self.val_dataset.data_dict
            concat_train_data_dict = train_data_dict + val_data_dict
            self.train_fold_data_list = [concat_train_data_dict[index]["image_path"] for index in train_indexes]
            self.val_fold_data_list = [concat_train_data_dict[index]["image_path"] for index in val_indexes]

    def train_dataloader(self):
        if not self.train_fold:
            return torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.args.num_workers)
        else:
            return torch.utils.data.DataLoader(self.train_fold, batch_size=self.batch_size, shuffle=True, num_workers=self.args.num_workers)
    
    def val_dataloader(self):
        if not self.val_fold:
            return torch.utils.data.DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.args.num_workers)
        else:
            return torch.utils.data.DataLoader(self.val_fold, batch_size=self.batch_size, shuffle=False, num_workers=self.args.num_workers)
    
    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.args.num_workers)
    

if __name__=="__main__":
    # base_dataset = K2Dataset(root=r"/mnt/d/datasets/K2_data_0417/9/train")
    # print(base_dataset.labels_map)
    # print(len(base_dataset))
    pass