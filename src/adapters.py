from typing import List
from collections import deque

import torch
import networkx as nx
from janus.models import MultiModalityCausalLM
from peft import LoraConfig, get_peft_model
from transformers import BitsAndBytesConfig

class GraphPruner:
    """
    Prunes a graph to a maximum number of nodes performing a BFS from the root node.
    """
    def __init__(self, max_nodes_per_graph):
        self.max_nodes_per_graph = max_nodes_per_graph

    @staticmethod
    def bfs(graph, root, max_nodes):
        visited = {root}
        queue = deque([root])
        result = []
        while queue and len(result) < max_nodes:
            node = queue.popleft()
            result.append(node)
            for neighbor in graph[node]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
        return result
    
    def __call__(self, graph: nx.Graph) -> nx.Graph:
        if len(graph) <= self.max_nodes_per_graph:
            return graph

        root = graph.graph['id']
        pruned_nodes = self.bfs(graph, root, self.max_nodes_per_graph)
        pruned_graph = graph.subgraph(pruned_nodes).copy()
        return pruned_graph

class Freezer:
    def __init__(self, freeze, num_layers=0):
        self.freeze = freeze
        self.num_layers = num_layers

    def _freeze_all(self, model):
        for param in model.parameters():
            param.requires_grad = False
        return model
    
    def _freeze_transformer_layers(self, model):
        """
        Freeze the first n transformer layers of a transformer model.
        """
        if self.num_layers <= 0:
            return model
        
        for param in model.parameters(): #First freeze all parameters
            param.requires_grad = False

        for i in range(self.num_layers, len(model.encoder.layer)): #Unfreeze the last n layers
            for param in model.encoder.layer[i].parameters():
                param.requires_grad = True

        for param in model.encoder.LayerNorm.parameters(): # Keep the final layer norm unfrozen
            param.requires_grad = True

        return model

    def __call__(self, model: torch.nn.Module):
        if self.freeze == "all":
            return self._freeze_all(model)
        if self.freeze == "transformer_layers":
            return self._freeze_transformer_layers(model)
        else:
            return model

class LoraAdapter:
    def __init__(self,
                 target_modules: List[str] = ["q_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
                 r: int = 8,
                 lora_alpha: int = 16,
                 lora_dropout: float = 0.1,
                 bias: str = "none",
                 quantization: int = None,               
                 ):
        self.target_modules = target_modules
        self.r = r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.bias = bias
        self.quantization = quantization
    
    def __call__(self, model_id, verbose=True):
        if self.quantization is not None:
            bnb_config = BitsAndBytesConfig(**{
                f"load_in_{self.quantization}bit": True,
                f"bnb_{self.quantization}bit_use_double_quant": True,
                f"bnb_{self.quantization}bit_quant_type": f"nf{self.quantization}",
                f"bnb_{self.quantization}bit_compute_dtype": torch.bfloat16,
            })
            model = MultiModalityCausalLM.from_pretrained(
                model_id,
                quantization_config=bnb_config,
                torch_dtype=torch.bfloat16,
            )
        else:
            model = MultiModalityCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch.bfloat16,
            )

        lora_config = LoraConfig(
            target_modules=self.target_modules,
            r=self.r,
            lora_alpha=self.lora_alpha,
            lora_dropout=self.lora_dropout,
            bias=self.bias,
            task_type="SEQ_CLS",
        )
        model = get_peft_model(model, lora_config)
        if verbose:
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            total_params = sum(p.numel() for p in model.parameters())
            print(f"Trainable params: {trainable_params} / {total_params}")
        return model