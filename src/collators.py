from src.utils import load_images

import torch
import transformers
from janus.models import VLChatProcessor

class VLLMCollator:
    def __init__(self, model_id):
        self.processor = VLChatProcessor.from_pretrained(model_id)
        self.tokenizer = self.processor.tokenizer

    def __call__(self, batch):
        batch_size = len(batch)

        texts = [item[0] for item in batch]
        image_paths = [item[1] for item in batch]
        labels = [item[2] for item in batch]

        image_tokens = (
            self.processor.image_start_token + 
            self.processor.image_token * self.processor.num_image_tokens + 
            self.processor.image_end_token
        ) # <begin_of_image> + <image_placeholder> * num_image_tokens + <end_of_image>

        # We will use a last eos token for classification
        prompts = [f"{image_tokens}\n{text}" + self.tokenizer.eos_token for text in texts]

        tokens = self.tokenizer(prompts, padding=True, return_tensors="pt")
        input_ids, attention_mask = tokens.input_ids, tokens.attention_mask

        images = load_images(image_paths)
        pixel_values = self.processor.image_processor(images, return_tensors="pt").pixel_values
        pixel_values = pixel_values.unsqueeze(1) # Number of images dimension (in our case, always 1)

        # images_seq_mask is a mask for the image tokens in the input_ids
        images_seq_mask = input_ids == self.processor.image_id

        # images_emb_mask is a mask for the pixel values across the batch
        # in our case is always True as we always have 1 image per sample
        images_emb_mask = torch.ones((batch_size, 1, self.processor.num_image_tokens), dtype=torch.bool)

        inputs = {
            "input_ids": input_ids, # batch_size x seq_len
            "attention_mask": attention_mask, # batch_size x seq_len
            "pixel_values": pixel_values, # batch_size x num_images x num_channels x height x width
            "images_seq_mask": images_seq_mask, # batch_size x seq_len
            "images_emb_mask": images_emb_mask, # batch_size x num_images x num_image_tokens
        }

        labels = torch.tensor(labels, dtype=torch.long) # batch_size x num_classes

        return inputs, labels