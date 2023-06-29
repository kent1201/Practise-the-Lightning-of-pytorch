import os
import torch
from argparse import ArgumentParser
from models.deep_vit import DeepVit
from models.simple_vit import SimpleVit
from models.small_dataset_vit import SmallDataVit
from models.timm_vit import Timm_Vit
from train.trainer import ClassifiedTrainer
from data.classification_dataset import ClassificationDataset, DataLoader
from utils.finetune import FineTuneBatchSizeFinder, FineTuneLearningRateFinder
from utils.prune import compute_amount


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

## Only for RTX 40 Series that has tensor cores
torch.set_float32_matmul_precision('high')


def ArgumentParsers():
    parser = ArgumentParser()
    ## base
    parser.add_argument("--devices", type=int, default=1)
    ## dataset
    parser.add_argument("--root_path", type=str, default=r"/mnt/d/datasets/K2_datasets/9")
    parser.add_argument("--labels", type=str, default="CP00,CP03,CP06,CP08,CP09,DR02,IT03,IT08,IT09")
    parser.add_argument("--num_classes", type=int, default=9)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--accumulate_grad_batches", type=int, default=1, help="help divide the big batchsize to small K batchsize to avoid memory overhead.")
    parser.add_argument("--load_size", type=int, default=224)
    parser.add_argument("--crop_size", type=int, default=224)
    parser.add_argument("--num_workers", type=int, default=2)
    ## Models
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--patch_size", type=int, default=32)
    parser.add_argument("--dim", type=int, default=1024)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--heads", type=int, default=16)
    parser.add_argument("--mlp_dim", type=int, default=2048)
    parser.add_argument("--save_ckpt_path", type=str, default="./Exp/swinv2_cr_small_224.sw_in1k")
    parser.add_argument("--load_ckpt_path", type=str, default="")
    parser.add_argument("--timm_model", type=str, default=r"swinv2_cr_small_224.sw_in1k")
    ## Train
    parser.add_argument("--dev", action='store_true', help='Help you fast run a loop of your train schedule')
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--max_epochs", type=int, default=200)
    parser.add_argument("--resume", nargs='?', const=True, default=False, help='resume most recent training')
    parser.add_argument("--precision", type=int, default=32, help="Mixed Precision of the model, input: [16|32]")
    parser.add_argument("--patience", type=int, default=10, help='Early stopping patience')
    parser.add_argument("--optimizer", type=str, default="Adam", help="[Adam|AdamW|Lion|]")
    parser.add_argument("--stochastic_weight_averaging", action='store_true', default=False, help='swa help to generalize model')
    parser.add_argument("--swa_lr", type=float, default=1e-2, help='lr of swa help to generalize model, only used when --stochastic_weight_averaging')
    parser.add_argument("--gradient_clip_val", type=float, default=0.5, help='gradient_clip to avoid exploding gradients. 0 means no clipped.')
    parser.add_argument("--tune", action='store_true', help='if called, Trainer will finetune the lr and batch size automatically. It\'s suitable for training model.')
    parser.add_argument("--finetune", action='store_true', help='if called, Trainer will finetune the lr and batch size each epoch automatically. It\'s suitable for finetuning model.')
    ## Validation

    ## Test

    ## Deploy
    parser.add_argument("--save_onnx", action='store_true', help="Save onnx model")
    parser.add_argument("--save_torch", action='store_true', help="Save onnx model")
    parser.add_argument("--pruning", action='store_true', help="Enable pruning during training to monomize the model.")

    ## Others
    parser.add_argument("--device_monitor", action='store_true', help="Set the monitor of device")

    ## Parse the user inputs and defaults (returns a argparse.Namespace)
    args = parser.parse_args()
    return args

if __name__=='__main__':
    args = ArgumentParsers()
    
    
    ## Datasets
    train_dataset = ClassificationDataset(args=args, mode='train')
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_dataset = ClassificationDataset(args=args, mode='val')
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_dataset = ClassificationDataset(args=args, mode='test')
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)


    ## Model
    # Using specific algorithm
    model = Timm_Vit(args=args)
    # Using specific kind of trainer
    classifier = ClassifiedTrainer(model=model, args=args)
    # Print summary of model
    summary = ModelSummary(classifier, max_depth=-1)
    print(summary)
    # Load exist check point (Not resume))
    if args.load_ckpt_path:
        classifier.load_from_checkpoint(args.load_ckpt_path)
    optimizer = classifier.configure_optimizers()
    
    ## Creat Trainer
    callback_list = []
    if args.stochastic_weight_averaging:
        SWA_callback = StochasticWeightAveraging(swa_lrs=args.swa_lr)
        callback_list.append(SWA_callback)
    # Finetune
    if args.finetune and not args.tune:
        callback_list.append(FineTuneLearningRateFinder(milestones=(5, 10)))
        # callback_list.append(FineTuneBatchSizeFinder(milestones=(5, 10)))
    # Set factor of Early Stop 
    early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=0.00, patience=args.patience, verbose=False, mode="min")
    callback_list.append(early_stop_callback)
    # Set factor of Saveing best result
    checkpoint_callback = ModelCheckpoint(dirpath=args.save_ckpt_path, filename='{epoch}-{val_loss:.2f}-{val_precision:.2f}', monitor="val_loss", mode="min", save_last=True)
    callback_list.append(checkpoint_callback)
    # Set prune of minize the model
    callback_list.append(ModelPruning("l1_unstructured", amount=compute_amount))
    # Check the bottleneck of the trainer
    profiler = AdvancedProfiler(dirpath=args.save_ckpt_path, filename="perf_logs")
    # Set the monitor of device
    if args.device_monitor:
        callback_list.append(DeviceStatsMonitor())
    # Set progress bar
    # progress_bar = RichProgressBar(
    #     theme=RichProgressBarTheme(
    #         description="green_yellow",
    #         progress_bar="green1",
    #         progress_bar_finished="green1",
    #         progress_bar_pulse="#6206E0",
    #         batch_progress="green_yellow",
    #         time="grey82",
    #         processing_speed="grey82",
    #         metrics="grey82",
    #     )
    # )
    # callback_list.append(progress_bar)
    callback_list.append(RichProgressBar())
    # Set the details of the trainer
    trainer = pl.Trainer(fast_dev_run=args.dev,
                         profiler=profiler,
                         accumulate_grad_batches=args.accumulate_grad_batches,
                         gradient_clip_val=args.gradient_clip_val,
                         accelerator="auto", 
                         devices=args.devices, 
                         max_epochs=args.max_epochs, 
                         default_root_dir=args.save_ckpt_path, 
                         callbacks=callback_list,
                         precision=args.precision)
    

    ## Tune parameters
    if args.tune and not args.finetune:
        tuner = Tuner(trainer)
        # Auto-scale batch size by growing it exponentially (default)
        # tuner.scale_batch_size(model, mode="power")
        # Auto-scale batch size with binary search
        # tuner.scale_batch_size(model, mode="binsearch")
        tuner.lr_find(classifier)

    ## Train and valid model
    # load specific checkpoints if resume
    resume_ckpt = args.resume if isinstance(args.resume, str) else None
    ## Pass model and inputs and start Training 
    trainer.fit(model=classifier, 
                train_dataloaders=train_dataloader, 
                val_dataloaders=val_dataloader,
                ckpt_path=resume_ckpt
                )
    # for epoch in range(args.max_epochs):
    #     for batch_idx, batch in enumerate(train_dataloader):
    #         train_loss = classifier.training_step(batch, batch_idx)
    #         optimizer.zero_grad()
    #         train_loss.backward()
    #         optimizer.step()
    #     trainer.valid(classifier, dataloaders=val_dataloader)
    
    ## Test model
    trainer.test(classifier, dataloaders=test_dataloader)

    ## Deploy model
    # Save ONNX
    if args.save_onnx:
        classifier.to_onnx(os.path.join(args.save_ckpt_path, "best_model.onnx"), export_params=True)
    # Save Torch Script
    if args.save_torch:
        torch_script = model.to_torchscript()
        torch.jit.save(torch_script, os.path.join(args.save_ckpt_path, "best_model.pt"))
    

    """
    import onnxruntime

    ort_session = onnxruntime.InferenceSession(filepath)
    input_name = ort_session.get_inputs()[0].name
    ort_inputs = {input_name: np.random.randn(1, 64)}
    ort_outs = ort_session.run(None, ort_inputs)
    """


        