from src.utils import load_config
from src.callbacks import CallbackFactory
from src.loggers import LoggerFactory
from src.datamodules import VLLMDataModule
from src.modules import VLLMClassifierModule

import os
import argparse

import pytorch_lightning as pl
from pytorch_lightning import seed_everything

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=str)
    parser.add_argument("--run", type=str)
    parser.add_argument("--devices", type=int, default=None)
    return parser.parse_args()

def train(project, run, devices=None):
    default_root_dir = f"runs/{project}/{run}"
    config = load_config(os.path.join(default_root_dir, "config.yaml"))

    seed_everything(config.training_config.seed)

    callbacks = CallbackFactory(config)()
    logger = LoggerFactory(config)()
    trainer = pl.Trainer(
        max_epochs=config.training_config.max_epochs,
        deterministic=config.training_config.deterministic,
        devices=[config.training_config.devices] if devices is None else [devices],
        accumulate_grad_batches=config.training_config.accumulate_grad_batches,
        logger=logger,
        callbacks=callbacks,
        default_root_dir=default_root_dir,
        precision=config.training_config.precision,
        val_check_interval=config.training_config.val_check_interval,
    )

    data_module = VLLMDataModule(config)
    module = VLLMClassifierModule(config, training_steps=data_module.epoch_steps * config.training_config.max_epochs)

    trainer.fit(module, data_module)

    best_model_path = os.path.join(default_root_dir, f"{config.run}.ckpt")
    best_model = VLLMClassifierModule.load_from_checkpoint(best_model_path, config=config)
    trainer.test(best_model, data_module)

if __name__ == "__main__":
    args = parse_args()
    train(args.project, args.run, args.devices)