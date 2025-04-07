from src.utils import r_getattr

import inspect
import torch
import transformers

def decoupled_weight_decay(params, weight_decay):
    """
    Decoupled weight decay is necessary because weight decay acts as L2 regularization, 
    which penalizes large values of parameters by pushing them toward zero. 
    
    This is beneficial for weights as it prevents overfitting. However, applying weight 
    decay to biases and normalization parameters  is problematic because these parameters 
    control shifts and scaling in the network rather than defining transformations. 
    These parameters will tend to shrink toward zero, which can disrupt learning dynamics.
    """
    def _dwd(param_group, weight_decay):

        param_dict = {pn: p for pn, p in param_group['module'].named_parameters()} # Get named parameters
        lr = param_group.get('lr', None) # Get learning rate if present

        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad} # Filter out non-grad parameters

        # Create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = {
            'params' : [p for n, p in param_dict.items() if p.dim() >= 2],
            'weight_decay': weight_decay
        }

        no_decay_params = {
            'params' : [p for n, p in param_dict.items() if p.dim() < 2],
            'weight_decay': 0.0
        }

        # Add learning rate if present
        if lr is not None: 
            decay_params['lr'] = lr 
            no_decay_params['lr'] = lr
        
        return [decay_params, no_decay_params]
    
    decoupled_params = []
    for p in params:
        decoupled_params.extend(_dwd(p, weight_decay))
    
    return decoupled_params

def constant_schedule_with_warmup(optimizer, warmup_steps, last_epoch=-1):
    return {
        'optimizer': optimizer,
        'lr_scheduler': {
            'scheduler': transformers.get_constant_schedule_with_warmup(
                optimizer, 
                num_warmup_steps=warmup_steps,
                last_epoch=last_epoch
            ),
            'interval': 'step',
            'frequency': 1,
            'name': 'lr/constant_schedule_with_warmup',
        }
    }

def linear_schedule_with_warmup(optimizer, warmup_steps, training_steps, last_epoch=-1):
    return {
        'optimizer': optimizer,
        'lr_scheduler': {
            'scheduler': transformers.get_linear_schedule_with_warmup(
                optimizer, 
                num_warmup_steps=warmup_steps, 
                num_training_steps=training_steps,
                last_epoch=last_epoch
            ),
            'interval': 'step',
            'frequency': 1,
            'name': 'lr/linear_schedule_with_warmup',
        }
    }

def cosine_schedule_with_warmup(optimizer, warmup_steps, training_steps, num_cycles=0.5, last_epoch=-1):
    return {
        'optimizer': optimizer,
        'lr_scheduler': {
            'scheduler': transformers.get_cosine_schedule_with_warmup(
                optimizer, 
                num_warmup_steps=warmup_steps, 
                num_training_steps=training_steps,
                num_cycles=num_cycles,
                last_epoch=last_epoch
            ),
            'interval': 'step',
            'frequency': 1,
            'name': 'lr/cosine_schedule_with_warmup',
        }
    }

def plateau(optimizer, patience=3, factor=0.1, monitor='val_loss'):
    return {
        'optimizer': optimizer,
        'lr_scheduler': {
            'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=factor, patience=patience),
            'interval': 'epoch',
            'frequency': 1,
            'monitor': monitor,
            'strict': False,
            'name': 'lr/plateau',
        }
    }

class OptimizerFactory:
    def __init__(self, config, params_dict=None):
        self.config = config
        self.params_dict = params_dict

        match self.config.optimizer.lower():
            case 'adamw':
                self.optimizer = torch.optim.AdamW
            case _:
                raise ValueError(f"Optimizer {self.config.optimizer} not implemented")
    
    def _parse_params(self, model):
        if self.params_dict is None:
            return [{
                'module': model,
                'params': model.parameters(),
            }]

        params = []
        for param in self.params_dict:
            lr = param.get('lr', self.config.lr)
            module = r_getattr(model, param['params'])
            if module is None:
                raise AttributeError(f"El modelo no tiene el atributo '{param['params']}'")
            params.append({
                'module': module,
                'params': module.parameters(),
                'lr': lr
            })

        return params
    
    def _create_optimizer(self, params):
        if self.config.decoupled_wd:
            params = decoupled_weight_decay(params, self.config.weight_decay)
        else:
            for p in params:
                del p['module'] # Remove module key
     
        return self.optimizer(
            params, 
            lr=self.config.lr, 
            weight_decay=self.config.weight_decay, 
            fused=self.config.fused and 'fused' in inspect.signature(self.optimizer).parameters
        )

    def __call__(self, model, training_steps=None):
        params = self._parse_params(model)

        optimizer = self._create_optimizer(params)
        match self.config.scheduler:
            case 'none':
                return optimizer
            
            case 'constant_schedule_with_warmup':
                return constant_schedule_with_warmup(optimizer, self.config.warmup_steps)
            
            case 'linear_schedule_with_warmup':
                return linear_schedule_with_warmup(optimizer, self.config.warmup_steps, training_steps)
                
            case 'cosine_schedule_with_warmup':
                return cosine_schedule_with_warmup(optimizer, self.config.warmup_steps, training_steps)

            case 'plateau':
                return plateau(optimizer, self.config.patience, self.config.factor, self.config.monitor)

            case _:
                raise ValueError(f"Scheduler {self.config.scheduler} not implemented")