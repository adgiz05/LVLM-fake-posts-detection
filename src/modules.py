from src.optimizers import OptimizerFactory
from src.models import VLLMClassifier

import torch
import pytorch_lightning as pl
from torchmetrics import Accuracy, F1Score

class VLLMClassifierModule(pl.LightningModule):
    def __init__(self, config, training_steps=None):
        super().__init__()

        self.save_hyperparameters(config.dict)
        model_config = config.multimodal_module_config
        data_config = config.data_config

        self.model = VLLMClassifier(
            model_id=model_config.model_id,
            num_classes=config.data_config.num_classes,
            dropout=model_config.dropout,
            lora_config=model_config.lora_config.dict,
        )

        self.loss_fn = torch.nn.CrossEntropyLoss()

        self.acc = Accuracy(task=data_config.task, num_classes=data_config.num_classes)
        self.f1 = F1Score(task="multiclass", num_classes=config.data_config.num_classes, average="macro")

        self.optimizer_config = config.optimizer_config

        self.training_steps = training_steps

    def forward(self, **inputs):
        return self.model(**inputs)
    
    def _step(self, batch, batch_idx):
        inputs, labels = batch
        logits = self.model(**inputs)
        loss = self.loss_fn(logits, labels)
        preds = torch.argmax(logits, dim=-1)

        acc = self.acc(preds, labels)
        f1 = self.f1(preds, labels)

        return loss, acc, f1
    
    def training_step(self, batch, batch_idx):
        loss, acc, f1 = self._step(batch, batch_idx)
        self.log_dict({
            "train_loss": loss,
            "train_acc": acc,
            "train_f1": f1,
        }, on_step=True, on_epoch=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        loss, acc, f1 = self._step(batch, batch_idx)
        self.log_dict({
            "val_loss": loss,
            "val_acc": acc,
            "val_f1": f1,
        }, on_step=False, on_epoch=True)
        return loss
    
    def test_step(self, batch, batch_idx):
        loss, acc, f1 = self._step(batch, batch_idx)
        self.log_dict({
            "test_loss": loss,
            "test_acc": acc,
            "test_f1": f1,
        }, on_step=False, on_epoch=True)
        return loss
    
    def configure_optimizers(self):
        return OptimizerFactory(self.optimizer_config)(
            model=self.model,
            training_steps=self.training_steps,
        )