import os
import yaml
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
import wandb
import torchmetrics

# Import model and dataset classes from local files
from model import ViTBackboneNet, Classifier, SingleScanModelBP, SingleScanModelQuad
from dataset import DualImageDataset, get_default_transform_dual, get_validation_transform_dual, dual_image_collate_fn
from train_lightning_os import QuadInputBinaryClassificationLightningModule, QuadInputBinaryClassificationDataModule


if __name__ == "__main__":
 
    if torch.cuda.is_available():
        print(f"Current CUDA device:     {torch.cuda.current_device()}")
        print(f"CUDA device name:        {torch.cuda.get_device_name(torch.cuda.current_device())}")
    print("-------------------------")

    torch.set_float32_matmul_precision('medium')
    import argparse
    parser = argparse.ArgumentParser()
    # Update default config path if needed for dual input specific config
    parser.add_argument('--config', type=str, default="config_finetune.yml") 
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    os.environ['CUDA_VISIBLE_DEVICES'] = config['gpu']['visible_device']

    wandb_logger = WandbLogger(
        project=config['logger']['project_name'],
        name=config['logger']['run_name'],
        config=config
    )

    data_module = QuadInputBinaryClassificationDataModule(config)
    model = QuadInputBinaryClassificationLightningModule(config)

    checkpoint_callback = ModelCheckpoint(
        dirpath=config['logger']['save_dir'],
        filename=config['logger']['save_name'],  
        monitor='val_auroc',
        mode='max',
        save_top_k=config['train'].get('save_top_k', 5)
    )
    lr_monitor = LearningRateMonitor(logging_interval='epoch')

    trainer = pl.Trainer(
        max_epochs=config['model']['max_epochs'],
        logger=wandb_logger,
        callbacks=[checkpoint_callback, lr_monitor],
        accelerator=config['train'].get('accelerator', 'auto'),
        devices=config['train'].get('devices', 'auto'),
        precision=config['train'].get('precision', "16-mixed")
    )

    trainer.fit(model, datamodule=data_module) 