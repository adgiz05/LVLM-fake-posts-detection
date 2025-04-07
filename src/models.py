from src.adapters import LoraAdapter

import torch

class VLLMBody(torch.nn.Module):
    def __init__(self,
                 model_id: str = "deepseek-ai/Janus-Pro-1B",
                 lora_config: dict = None,
                 ):
        super().__init__()
        if lora_config is not None:
            self.model = LoraAdapter(**lora_config)(model_id)
        else:
            self.model = MultiModalityCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16)

    def forward(self, **inputs):
        input_embeds = self.model.prepare_inputs_embeds(**inputs) # batch_size x seq_len x hidden_size
        last_hidden_state = self.model.language_model(
            input_ids=None,
            inputs_embeds=input_embeds,
            attention_mask=inputs["attention_mask"],
            output_hidden_states=True,
        ).hidden_states[-1] # batch_size x seq_len x hidden_size

        return last_hidden_state[:, -1, :] # batch_size x hidden_size
    
    @property
    def hidden_size(self):
        return self.model.config.language_config.hidden_size
    
class VLLMClassifier(torch.nn.Module):
    def __init__(self,
                 model_id: str = "deepseek-ai/Janus-Pro-1B",
                 num_classes: int = 2,
                 dropout: float = 0.1,
                 lora_config: dict = None,
                 ):
        super().__init__()
        self.body = VLLMBody(model_id, lora_config=lora_config)
        self.classifier = torch.nn.Linear(self.body.hidden_size, num_classes)

        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, **inputs):
        eos_embedding = self.body(**inputs) # batch_size x hidden_size
        eos_embedding = self.dropout(eos_embedding) 
        logits = self.classifier(eos_embedding) # batch_size x num_classes
        return logits