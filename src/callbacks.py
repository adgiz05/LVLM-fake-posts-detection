import os

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor

class PostTrainingCleaner(pl.Callback):
    def on_test_end(self, trainer, pl_module):
        save_dir = trainer.default_root_dir
        last_ckpt = os.path.join(save_dir, "last.ckpt")
        # Remove last checkpoint
        if os.path.exists(last_ckpt):
            os.remove(last_ckpt)

        return super().on_test_end(trainer, pl_module)

class CallbackFactory:
    def __init__(self, config):
        self.config = config
        self.callback_config = config.callback_config

    def __call__(self):
        save_dir = f"runs/{self.config.project}/{self.config.run}"
        callbacks = []

        if "model_checkpoint" in self.callback_config.callbacks:
            callbacks.append(ModelCheckpoint(
                monitor=self.callback_config.monitor,
                mode=self.callback_config.mode,
                save_top_k=self.callback_config.save_top_k,
                dirpath=save_dir,
                filename=self.config.run,
                save_last=self.callback_config.save_last
            ))

        if "early_stopping" in self.callback_config.callbacks:
            callbacks.append(EarlyStopping(
                monitor=self.callback_config.monitor,
                patience=self.callback_config.patience,
                mode=self.callback_config.mode
            ))
        
        if "lr_monitor" in self.callback_config.callbacks:
            callbacks.append(LearningRateMonitor(logging_interval=self.callback_config.logging_interval))
        
        if "post_training_cleaner" in self.callback_config.callbacks:
            callbacks.append(PostTrainingCleaner())

        return callbacks