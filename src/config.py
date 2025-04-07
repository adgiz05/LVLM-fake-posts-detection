from typing import Any, List, Dict
from dataclasses import dataclass, asdict, field

from src.constants import CLS2SUBREDDIT

from torch.nn import Module

class BaseConfig:
    @property
    def dict(self):
        return asdict(self)


@dataclass
class FreezerConfig(BaseConfig):
    freeze              : str               = "all" # ["none", "all", "transformer_layers"]
    num_layers          : int               = 0


@dataclass
class CommentsConfig(BaseConfig):
    comment_format      : str               = "{author} ({score}): {text}"
    max_nodes_per_graph : int               = 16 # Prune graphs to this many nodes. None to disable pruning.
    max_comment_words   : int               = None # Truncate comments to this many words. None to disable.


@dataclass
class EncoderGNNModuleConfig(BaseConfig):
    __build__               : str           = "Transformer-based encoder + GNN"
    params_dict             : List[dict]    = None # If None, default_params will be used

    # Comments hyperparameters
    comments_config         : CommentsConfig = CommentsConfig()

    # Encoder hyperparameters
    encoder_id              : str           = "microsoft/deberta-v3-base"
    max_length              : int           = 128
    encoder_lr              : float         = 2.5e-5 # The one used for fine-tuning DeBERTaV3
    freezer_config          : FreezerConfig = FreezerConfig() # Freezer configuration

    # GNN hyperparameters
    gnn                     : Module        = 'gcn' 
    hidden_channels         : int           = 128
    num_layers              : int           = 3
    pooling                 : str           = 'global_mean_pool'
    gnn_lr                  : float         = 2.5e-4
    dropout                 : float         = 0.1

    @property
    def get_params_dict(self):
        if self.params_dict is not None:
            return self.params_dict
        else:
            return [
                {"params": "body.comments_encoder", "lr": self.encoder_lr},
                {"params": "body.gnn", "lr": self.gnn_lr},
                {"params": "head", "lr": self.gnn_lr}
            ]


@dataclass
class OptimizerConfig(BaseConfig):
    """
    "params" can be list of dicts containing "params" and "lr" keys
    for assigning different learning rates to different parts of the model:
    [{"params": "encoder", "lr": 1e-4}, {"params": "cls", "lr": 1e-3}]
    OptimizerFactory will then transfom "encoder" into eval("model.encoder.parameters()").
    """
    # Optimizer
    optimizer               : str           = "adamw"
    params                  : List[Dict]    = None # If None, all model parameters are optimized
    lr                      : float         = 2.5e-4
    weight_decay            : float         = 0.01
    decoupled_wd            : bool          = False
    fused                   : bool          = False

    # Scheduler
    scheduler               : str           = "none" # ["none", "plateau", "linear_schedule_with_warmup"...]
    warmup_steps            : int           = 100
    factor                  : float         = 0.1
    patience                : int           = 3


@dataclass
class DataConfig(BaseConfig):
    # Dataloader
    batch_size              : int           = 16
    num_workers             : int           = 4
    prefetch_factor         : int           = 2
    pin_memory              : bool          = True

    # Sampling
    num_train_samples       : int           = 15_000
    num_val_samples         : int           = 5_000
    # num_train_samples       : int           = None
    # num_val_samples         : int           = None
    
    # Label
    label_str               : str           = "2_way_label"

    @property
    def num_classes(self):
        if self.label_str.endswith("way_label"):
            return int(self.label_str.split("_")[0]) # n_way_label -> n
        elif self.label_str == 'subreddit':
            return len(CLS2SUBREDDIT)
        
        
    @property
    def task(self):
        if self.num_classes == 2:
            return "binary"
        elif self.num_classes > 2:
            return "multiclass"
        else:
            raise ValueError(f"Invalid number of classes: {self.num_classes}")
            
@dataclass
class LoggerConfig(BaseConfig):
    logger                  : str|List[str] = "wandb"  


@dataclass
class CallbackConfig(BaseConfig):
    # Callbacks
    callbacks: str | List[str] = field(default_factory=lambda: ["early_stopping", "model_checkpoint", "lr_monitor", "post_training_cleaner"])

    # Parameters
    monitor                 : str           = "val_loss"
    mode                    : str           = "min"
    save_top_k              : int           = 1
    save_last               : bool          = True
    patience                : int           = 10
    logging_interval        : int           = None


@dataclass
class TrainingConfig(BaseConfig):
    seed                    : int           = 42
    devices                 : int|List[int] = 2
    max_epochs              : int           = 50
    deterministic           : bool          = True
    accumulate_grad_batches : int           = 1
    precision               : str           = "bf16"
    val_check_interval      : float         = 1.0


@dataclass
class RunConfig(BaseConfig):
    # Experiment
    project                 : str       
    run                     : str       

    # Modules
    comments_module_config  : Any       = None 
    multimodal_module_config: Any       = None

    data_config             : Any       = DataConfig() # DataLoader 
    optimizer_config        : Any       = OptimizerConfig() # Optimizer
    callback_config         : Any       = CallbackConfig() # Callbacks
    logger_config           : Any       = LoggerConfig() # Logger
    training_config         : Any       = TrainingConfig() # Training

# </config>

# <vllm_config>
@dataclass
class LoraAdapterConfig(BaseConfig):
    target_modules      : List[str]         = field(default_factory=lambda: ["q_proj", "v_proj", "gate_proj", "up_proj", "down_proj"])
    r                   : int               = 8
    lora_alpha          : int               = 16
    lora_dropout        : float             = 0.1
    bias                : str               = "none"
    quantization        : int               = None

@dataclass
class VLLMModuleConfig(BaseConfig):
    __build__           : str               = "VLLM last token hidden state based classifier"
    model_id            : str               = "deepseek-ai/Janus-Pro-1B"
    dropout             : float             = 0.1
    comments_config     : CommentsConfig    = CommentsConfig()
    lora_config         : Any               = LoraAdapterConfig()