import pytorch_lightning as pl

class LoggerFactory:
    def __init__(self, config):
        self.config = config
        self.logger_config = config.logger_config

    def __call__(self):
        match self.logger_config.logger:
            case "wandb":
                return pl.loggers.WandbLogger(project=self.config.project, name=self.config.run)
            case _:
                return None