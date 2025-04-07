from src.utils import comments_tree_to_text
from src.constants import IMAGES_PATH

import torch
import pandas as pd

class VLLMDataset(torch.utils.data.Dataset):
    def __init__(self,
                 data               : pd.DataFrame, 
                 comments           : dict = None,
                 comment_format     : str = "{author} ({score}): {text}",
                 label_str          : str = "2_way_label",
                 max_comment_words  : int = None
                 ):
        super().__init__()
        self.data = data 
        self.comments = comments
        self.comment_format = comment_format
        self.label_str = label_str
        self.max_comment_words = max_comment_words

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        id_ = row['id']

        image_path = f"{IMAGES_PATH}/{id_}.jpg" # Image path

        if self.comments is not None: # If we add comments
            graph = self.comments[id_]
            text = comments_tree_to_text(graph, self.comment_format, max_comment_words=self.max_comment_words)
        else: # Only the image caption
            text = self.comment_format.format(
                author=row['author'],
                text=row['clean_title'],
                score=int(row['score'])
            )

        label = row[self.label_str]

        return text, image_path, label