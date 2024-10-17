import os
import random
import torch
from argparse import ArgumentParser

from train import CreateModel
from data import CreateDataset
# from data.Classification.classification_dataset import ClassificationDataset #, DataLoader
from utils.finetune import FineTuneBatchSizeFinder, FineTuneLearningRateFinder
from utils.prune import compute_amount
from utils.data_visualization import DataVisualization
from utils.utils import CheckSavePath, ListDir
from utils.ema import EMA

import lightning.pytorch as pl
from lightning.pytorch.utilities.model_summary import ModelSummary
from lightning.pytorch.profilers import AdvancedProfiler
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks import StochasticWeightAveraging
from lightning.pytorch.tuner import Tuner
from lightning.pytorch.callbacks import ModelPruning
from lightning.pytorch.callbacks import DeviceStatsMonitor
from lightning.pytorch.callbacks import RichProgressBar
from lightning.pytorch.callbacks.progress.rich_progress import RichProgressBarTheme

import matplotlib.pyplot as plt


## Only for RTX 40 Series that has tensor cores
torch.set_float32_matmul_precision('high')

random.seed(42)


def ArgumentParsers():
    parser = ArgumentParser()
    ## base
    parser.add_argument("--devices", type=int, default=1)
    parser.add_argument("--task", type=str, default="classification", help="[classification|defect_gan]")
    parser.add_argument("--mode", type=str, default="fit", help="[fit|train|test]")
    parser.add_argument("--project", type=str, help=r"decide the project (dataset)")
    ## dataset
    parser.add_argument("--root_path", type=str, default=r"/workspace/Datasets/animals_64classes/Ver_001_20241015")
    parser.add_argument("--num_classes", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--accumulate_grad_batches", type=int, default=1, help="help divide the big batchsize to small K batchsize to avoid memory overhead.")
    parser.add_argument("--load_size", type=int, default=512)
    parser.add_argument("--crop_size", type=int, default=512)
    parser.add_argument("--num_channels", type=int, default=3)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--num_splits", nargs='?', const=True, default=False, help='number of folds you want to split, only used when k-fold cross validation')
    parser.add_argument("--k_folds", nargs='?', const=True, default=False, help='number of ration of train and val, only used when k-fold cross validation')
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument("--mixup", action='store_true', help="If called, images would be mixed up.")
    parser.add_argument("--mixup_interval", type=int, default=5, help="Only used when --mixup")
    parser.add_argument("--mixup_alpha", type=float, default=0.2, help="Only used when --mixup")
    parser.add_argument("--compute_mean_std", action='store_true', help="calculate the mean and std of the dataset.")
    ## Models (For Vit-pytorch)
    # parser.add_argument("--image_size", type=int, default=224, help="image size for Vit-pytorch, not for timm model")
    # parser.add_argument("--patch_size", type=int, default=32)
    # parser.add_argument("--dim", type=int, default=1024)
    # parser.add_argument("--depth", type=int, default=6)
    # parser.add_argument("--heads", type=int, default=16)
    # parser.add_argument("--mlp_dim", type=int, default=2048)

    parser.add_argument("--save_ckpt_path", type=str, default=r"./Exp/test001_241017")
    parser.add_argument("--load_ckpt_path", type=str)
    parser.add_argument("--load_sala_ckpt_path", type=str)
    parser.add_argument('--model', type=str, default="Timm_Vit", help="[Timm_Vit|DeepViT|SimpleVit|SmallDataVit|SALA]")
    parser.add_argument("--timm_model", type=str, default=r"eva02_small_patch14_336.mim_in22k_ft_in1k", help="Only used when model_name is Timm_Vit")
    parser.add_argument("--loss", type=str, default="CrossEntropy", help="[CrossEntropy|Focal|SigmoidFocal|Poly1CrossEntropyLoss]")
    parser.add_argument("--loss_with_cls_weight", nargs='?', const=True, default=False, help="the weight in different classes, now only support for Crossentropy loss. ex: 1.0,2.0,0.5,...")

    ## Train
    parser.add_argument("--dev", action='store_true', help='Help you fast run a loop of your train schedule')
    parser.add_argument("--optimizer", type=str, default="Adam", help="[Adam|SGD|AdamW|NAdam|Lion|NovoGrad]")
    parser.add_argument("--flooding", nargs='?', const=True, default=False, help="Suggest: val_loss * 0.5. Do We Need Zero Training Loss After Achieving Zero Training Error? ICML, 2020")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--special_lr_factor", nargs='?', const=True, default=False, help="special learning rate factor (factor * --lr) for model.head.parameters, only support for timm model.")
    parser.add_argument("--dropout", nargs='?', const=True, default=False, help="dropout range: 0.2~0.8")
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--max_epochs", type=int, default=100)
    parser.add_argument("--resume", nargs='?', const=True, default=False, help='resume most recent training')
    parser.add_argument("--precision", type=str, default="32", help="Mixed Precision of the model, input: [16-mixed|32]")
    parser.add_argument("--patience", type=int, default=10, help='Early stopping patience')
    parser.add_argument("--scheduler", type=str, default="StepLR", help="[StepLR|ReduceLROnPlateau|ExponentialLR|CosineAnnealingLR|CosineAnnealingWarmRestarts]")
    parser.add_argument("--lr_factor", type=float, default=0.1, help="learning rate gamma of scheduler")
    parser.add_argument("--lr_step_size", type=int, default=20, help="learning rate step size of scheduler")
    parser.add_argument("--lr_cycle", type=int, default=20, help="learning rate cycle of scheduler")
    parser.add_argument("--stochastic_weight_averaging", action='store_true', default=False, help='swa help to generalize model')
    parser.add_argument("--exponential_moving_average", nargs='?', const=True, default=False, help='ema help to generalize model')
    parser.add_argument("--swa_lr", type=float, default=1e-4, help='lr of swa help to generalize model, only used when --stochastic_weight_averaging')
    parser.add_argument("--gradient_clip_val", type=float, default=0, help='gradient_clip to avoid exploding gradients. 0 means no clipped.')
    parser.add_argument("--tune", action='store_true', help='if called, Trainer will finetune the lr and batch size automatically. It\'s suitable for training model.')
    parser.add_argument("--finetune", action='store_true', help='if called, Trainer will finetune the lr and batch size each epoch automatically. It\'s suitable for finetuning model.')

    ## Deploy
    parser.add_argument("--save_onnx", action='store_true', help="Save onnx model")
    parser.add_argument("--save_torch", action='store_true', help="Save torch model")
    parser.add_argument("--pruning", action='store_true', help="Enable pruning during training to monomize the model.")

    ## Others
    parser.add_argument("--device_monitor", action='store_true', help="Set the monitor of device")
    parser.add_argument("--analysis_method", type=str, default="none", help="[none|cams|pca|tsne|kmeans|all]")

    ## Parse the user inputs and defaults (returns a argparse.Namespace)
    args = parser.parse_args()
    
    return args

def CallBackDict(args):
    call_back_dict = dict()
    k_folds = int(args.k_folds) if isinstance(args.k_folds, str) else None

    if args.stochastic_weight_averaging:
        SWA_callback = StochasticWeightAveraging(swa_lrs=args.swa_lr)
        call_back_dict["stochastic_weight_averaging"] = SWA_callback

    if args.exponential_moving_average:
        call_back_dict["exponential_moving_average"] = EMA(decay=float(args.exponential_moving_average))

    # Finetune
    if args.finetune and not args.tune:
        call_back_dict["FineTuneLearningRateFinder"] = FineTuneLearningRateFinder(milestones=(5, 10))
        call_back_dict["FineTuneBatchSizeFinder"] = FineTuneBatchSizeFinder(milestones=(5, 10))

    # Set factor of Early Stop
    if not args.pruning:
        call_back_dict["EarlyStopping"] = EarlyStopping(monitor="val_precision", min_delta=0.00, patience=args.patience, verbose=False, mode="max")
    
    # Set prune of minize the model
    if args.pruning:
        call_back_dict["ModelPruning"] = ModelPruning("l1_unstructured", amount=compute_amount)

    
    call_back_dict["RichProgressBar"] = RichProgressBar()

    # Set the monitor of device
    if args.device_monitor:
        call_back_dict["DeviceStatsMonitor"] = DeviceStatsMonitor()

    # (Must be last) Set factor of Saveing best result
    if not k_folds:
        call_back_dict["ModelCheckpoint"] = ModelCheckpoint(dirpath=args.save_ckpt_path, filename='{epoch}-{val_loss:.2f}-{val_precision:.2f}', monitor="val_precision", mode="max", save_last=True)

    return call_back_dict

def train(args):

    ## Create Datasets
    dataset = None
    dataset = CreateDataset(args=args)

    ## Create Model
    model = CreateModel(args=args)
    # Print summary of model
    summary = ModelSummary(model, max_depth=-1)
    print(summary)
    # Load exist check point (Not resume))
    if args.load_ckpt_path:
        model.load_from_checkpoint(args.load_ckpt_path)
    # optimizer = classifier.configure_optimizers()

    ## Train and valid model
    # load specific checkpoints if resume
    resume_ckpt = args.resume if isinstance(args.resume, str) else None

    ## Creat Trainer
    callback_dict = CallBackDict(args)
    profiler = AdvancedProfiler(dirpath=args.save_ckpt_path, filename="perf_logs")
    # Set the details of the trainer
    trainer = pl.Trainer(fast_dev_run=args.dev,
                        profiler=profiler,
                        accumulate_grad_batches=args.accumulate_grad_batches,
                        gradient_clip_val=args.gradient_clip_val,
                        accelerator="auto", 
                        devices=args.devices, 
                        max_epochs=args.max_epochs, 
                        default_root_dir=args.save_ckpt_path, 
                        callbacks=list(callback_dict.values()),
                        precision=args.precision
                    )
    
    ## Tune parameters
    if args.tune and not args.finetune:
        tuner = Tuner(trainer)
        # Auto-scale batch size by growing it exponentially (default)
        # tuner.scale_batch_size(model, mode="power")
        # Auto-scale batch size with binary search
        tuner.scale_batch_size(model, mode="binsearch")
        tuner.lr_find(model)

    trainer.fit(model=model,
                datamodule=dataset,
                ckpt_path=resume_ckpt
            )
    ## Testing
    dataset.setup("test")
    trainer.test(model=model, datamodule=dataset)

    return trainer, model, dataset
    

def test(args):
    ## Create Datasets
    dataset = None
    dataset = CreateDataset(args=args)

    ## Create Model
    model = CreateModel(args=args)
    if not args.load_ckpt_path:
        load_ckpt_path = os.path.join(args.save_ckpt_path, "last.ckpt")
        model.load_from_checkpoint(load_ckpt_path)
    if args.load_ckpt_path:
        model.load_from_checkpoint(args.load_ckpt_path)
    # Print summary of model
    # summary = ModelSummary(model, max_depth=-1)
    # print(summary)

    ## Creat Trainer
    callback_dict = CallBackDict(args)
    profiler = AdvancedProfiler(dirpath=args.save_ckpt_path, filename="perf_logs")
    # Set the details of the trainer
    trainer = pl.Trainer(fast_dev_run=args.dev,
                        profiler=profiler,
                        accumulate_grad_batches=args.accumulate_grad_batches,
                        gradient_clip_val=args.gradient_clip_val,
                        accelerator="auto", 
                        devices=args.devices, 
                        max_epochs=args.max_epochs, 
                        default_root_dir=args.save_ckpt_path, 
                        callbacks=list(callback_dict.values()),
                        precision=args.precision
                    )
    dataset.setup("test")
    trainer.test(model=model, datamodule=dataset)

    return trainer, model, dataset
    
    

if __name__=='__main__':

    from common_data import PROJECT

    args = ArgumentParsers()

    num_splits = int(args.num_splits) if isinstance(args.num_splits, str) else None
    k_folds = int(args.k_folds) if isinstance(args.k_folds, str) else None

    CheckSavePath(args.save_ckpt_path)

    if args.project in PROJECT:
        setattr(args, "labels", ",".join(PROJECT[args.project]["labels"]))

    
    trainer, model, dataset = None, None, None
    ## Pass model and inputs and start Training 
    if args.mode == "train" or args.mode == "fit":
        if not k_folds:
            trainer, model, dataset = train(args=args)
            # test(args=args)
        else: ## K-Fold training
            default_save_ckpt_path = args.save_ckpt_path
            for k in range(num_splits):
                print("="*10, "start {} training".format(k), "="*10)
                
                dst_file_path = os.path.join(default_save_ckpt_path, "k_{}".format(k))
                CheckSavePath(dst_file_path)
                args.save_ckpt_path = dst_file_path
                
                trainer, dataset = train(args=args)
                # test(args=args)
                
                with open(os.path.join(dst_file_path, "train.txt"), "w") as f:
                    for item in dataset.train_fold_data_list:
                        f.write("{}\n".format(item))
                with open(os.path.join(dst_file_path, "val.txt"), "w") as f:
                    for item in dataset.val_fold_data_list:
                        f.write("{}\n".format(item))

        ## Deploy model
        # Save ONNX
        if args.save_onnx:
            trainer.model.to_onnx(os.path.join(args.save_ckpt_path, "best_model.onnx"), export_params=True)
        # Save Torch Script
        if args.save_torch:
            torch_script = trainer.to_torchscript()
            torch.jit.save(torch_script, os.path.join(args.save_ckpt_path, "best_model.pt"))
    
    elif args.mode == "test":
        trainer, model, dataset = test(args=args)


    if args.analysis_method != "none":
        dataset.setup("test")
        data_visualizer = DataVisualization(args=args, model=model.model, dataset=dataset)
        if args.analysis_method == "all" or args.analysis_method == "cams":
            data_visualizer.DrawCAM()
        # data_X, data_Y = data_visualizer.dataTransform()
        # data_visualizer.visualization(data_X=data_X, data_Y=data_Y)
        
        
    

    """
    import onnxruntime

    ort_session = onnxruntime.InferenceSession(filepath)
    input_name = ort_session.get_inputs()[0].name
    ort_inputs = {input_name: np.random.randn(1, 64)}
    ort_outs = ort_session.run(None, ort_inputs)
    """


        