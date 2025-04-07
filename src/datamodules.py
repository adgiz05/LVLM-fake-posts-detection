from src.adapters import GraphPruner
from src.collators import VLLMCollator
from src.datasets import VLLMDataset

import torch
import pickle
import pandas as pd
import pytorch_lightning as pl

class VLLMDataModule(pl.LightningDataModule):
    def __init__(self, config, collator=None):
        super().__init__()
        # Comments preprocessing
        self.comment_format = config.multimodal_module_config.comments_config.comment_format
        self.max_nodes_per_graph = config.multimodal_module_config.comments_config.max_nodes_per_graph
        self.max_comment_words = config.multimodal_module_config.comments_config.max_comment_words

        # Sampling
        self.label_str = config.data_config.label_str
        self.num_train_samples = config.data_config.num_train_samples
        self.num_val_samples = config.data_config.num_val_samples
        self.seed = config.training_config.seed

        # DataLoader
        self.batch_size = config.data_config.batch_size
        self.num_workers = config.data_config.num_workers
        self.prefetch_factor = config.data_config.prefetch_factor
        self.pin_memory = config.data_config.pin_memory

        # Collator
        self.collator = collator if collator is not None else VLLMCollator(config.multimodal_module_config.model_id)

    def _setup_stage(self, stage):
        data = pd.read_csv(f"dataset/{stage}.csv")
        comments = pickle.load(open(f"dataset/comments/{stage}_comment_trees.pkl", "rb"))

        if self.max_nodes_per_graph is not None:
            print(f"Pruning graphs to maximum {self.max_nodes_per_graph} nodes...")
            graph_pruner = GraphPruner(self.max_nodes_per_graph)
            comments = {k: graph_pruner(v) for k, v in comments.items()}
        
        if stage == 'train' and self.num_train_samples is not None:
            groups = data.groupby(self.label_str)
            data = groups.apply(lambda x: x.sample(n=int(self.num_train_samples * len(x) / len(data)), random_state=self.seed))
            data = data.reset_index(drop=True)
        if stage == 'val' and self.num_val_samples is not None:
            groups = data.groupby(self.label_str)
            data = groups.apply(lambda x: x.sample(n=int(self.num_val_samples * len(x) / len(data)), random_state=self.seed))
            data = data.reset_index(drop=True)

        return VLLMDataset(
            data=data,
            comments=comments,
            comment_format=self.comment_format,
            label_str=self.label_str,
            max_comment_words=self.max_comment_words,
        )

    def setup(self, stage=None):
        if (stage == 'fit' or stage is None) and not hasattr(self, 'train_dataset'):
            self.train_dataset = self._setup_stage('train')
            self.val_dataset = self._setup_stage('val')

        if stage in ['test', 'predict']:
            self.test_dataset = self._setup_stage('test')

    def train_dataloader(self, shuffle=True):
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            prefetch_factor=self.prefetch_factor,
            pin_memory=self.pin_memory,
            collate_fn=self.collator,
            shuffle=shuffle, # Debugging purposes
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            prefetch_factor=self.prefetch_factor,
            pin_memory=self.pin_memory,
            collate_fn=self.collator,
        )

    def test_dataloader(self, batch_size=None):
        return torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=batch_size if batch_size is not None else self.batch_size,
            num_workers=self.num_workers,
            prefetch_factor=self.prefetch_factor,
            pin_memory=self.pin_memory,
            collate_fn=self.collator,
        )
    
    @property
    def epoch_steps(self):
        self.setup('fit')
        return len(self.train_dataloader())