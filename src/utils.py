from src.config import *
from src.constants import *

import random
import os
import yaml

import PIL
import tqdm
import pickle
import tarfile
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

### DATA PREPROCESSING ###
def sample(data, num_samples, seed=42):
    """
    Return a random sample from a dict.
    """
    if num_samples is None: # Return all data if num_samples is None
        return data

    random.seed(seed)
    sample_idx = random.sample(range(len(data)), num_samples)

    if isinstance(data, dict):
        idx2key = {i: k for i, k in enumerate(data)}
        return {idx2key[i]: data[idx2key[i]] for i in sample_idx}
    
    else:
        return [data[i] for i in sample_idx]
    
def r_getattr(obj, attr):
    for part in attr.split('.'):
        obj = getattr(obj, part)
    return obj

def ensure_node_attributes(graph):
    for id_, data in graph.nodes(data=True):
        data.setdefault('author', id_)
        data.setdefault('score', 0)
        data.setdefault('text', "")
    return graph

def save_comments_as_graphs(split):
    """
    Save all the comments in a split as a dictionary of nx.Graphs
    This function may take a while to execute as it has to load and group all comments by submission_id:
    - Train split (564k samples): ~6 mins
    - Val split (62k samples): ~1 min
    - Test split (62k samples): ~1 min
    """
    DATA_PATH, COMMENTS_PATH = f'dataset/{split}.csv', 'dataset/comments/all_comments.csv'
    data = pd.read_csv(DATA_PATH)
    comments = pd.read_csv(COMMENTS_PATH)

    comments_grouped = comments.groupby('submission_id') # Group by submission_id
    comments_trees = {}

    for post in tqdm.tqdm(data.itertuples(index=False)):
        post_id = post.id

        if post_id in comments_grouped.groups: # If there are comments for this post
            post_comments = comments_grouped.get_group(post_id).to_dict(orient='records')
        else: # If there are no comments for this post (empty list)
            post_comments = []

        G = nx.Graph()
        G.graph['id'] = post_id # The graph's id is the post's id
        G.graph['2_way_label'] = post._13
        G.graph['3_way_label'] = post._14
        G.graph['6_way_label'] = post._15

        G.add_node(post_id, author=post.author, text=post.clean_title, score=int(post.score)) # Add the post as a node

        for row in post_comments:
            parent_id = row['parent_id'].split('_')[-1]
            G.add_node(row['id'], author=row['author'], text=row['body'], score=row['ups'])
            G.add_edge(parent_id, row['id'])
        
        comments_trees[post_id] = G

    with open(f'dataset/comments/{split}_comment_trees.pkl', 'wb') as f:
        pickle.dump(comments_trees, f) # Save the graphs as a dictionary of nx.Graphs

### CONFIG PARSER ###
def generate_run_config(project, run, config=None, multimodal=False):
    project_path = os.path.join("runs", project)
    
    if not os.path.exists(project_path):
        os.makedirs(project_path)
    
    run_name = run
    counter = 1
    
    while os.path.exists(os.path.join(project_path, run_name)):
        run_name = f"{run}_{counter}"
        counter += 1
    
    run_path = os.path.join(project_path, run_name)
    os.makedirs(run_path)
    
    config_file = os.path.join(run_path, "config.yaml")
    if config is not None:
        config.run = run_name
        config_dict = config.dict
    else:
        if multimodal:
            multimodal_module_config = VLLMModuleConfig().dict
            config_dict = RunConfig(
                project=project,
                run=run_name,
                multimodal_module_config=multimodal_module_config
            ).dict
        else:
            comments_module_config = EncoderGNNModuleConfig().dict
            config_dict = RunConfig(
                project=project,
                run=run_name,
                comments_module_config=comments_module_config
            ).dict
    with open(config_file, "w") as f:
        yaml.dump(config_dict, f)
    
    return run_name

def load_config(yaml_path: str) -> RunConfig:
    with open(yaml_path, "r") as f:
        data = yaml.safe_load(f)

    comments_config = data.get("comments_module_config", None)
    if comments_config is not None:
        if "freezer_config" in comments_config and isinstance(comments_config["freezer_config"], dict):
            comments_config["freezer_config"] = FreezerConfig(**comments_config["freezer_config"])
        if "comments_config" in comments_config and isinstance(comments_config["comments_config"], dict):
            comments_config["comments_config"] = CommentsConfig(**comments_config["comments_config"])
        comments_module_config = EncoderGNNModuleConfig(**comments_config)
    else:
        comments_module_config = None

    multimodal_config = data.get("multimodal_module_config", {})
    if multimodal_config is not None:
        if "lora_config" in multimodal_config and isinstance(multimodal_config["lora_config"], dict):
            multimodal_config["lora_config"] = LoraAdapterConfig(**multimodal_config["lora_config"])
        if "comments_config" in multimodal_config and isinstance(multimodal_config["comments_config"], dict):
            multimodal_config["comments_config"] = CommentsConfig(**multimodal_config["comments_config"])
        multimodal_module_config = VLLMModuleConfig(**multimodal_config)
    else:
        multimodal_module_config = None

    data_config = DataConfig(**data.get("data_config", {}))
    optimizer_config = OptimizerConfig(**data.get("optimizer_config", {}))
    callback_config = CallbackConfig(**data.get("callback_config", {}))
    logger_config = LoggerConfig(**data.get("logger_config", {}))
    training_config = TrainingConfig(**data.get("training_config", {}))
    
    project = data["project"]
    run = data["run"]

    config = RunConfig(
        project=project,
        run=run,
        comments_module_config=comments_module_config,
        multimodal_module_config=multimodal_module_config,
        data_config=data_config,
        optimizer_config=optimizer_config,
        callback_config=callback_config,
        logger_config=logger_config,
        training_config=training_config
    )
    return config

### VISUALIZATION ###
def draw_post(id_=None, data=None, comment_trees=None, split='train'):
    """
    Draw a post with its image, caption, and comments tree.

    - id_: str, submission_id of the post, if None a random one will be chosen
    - data: pd.DataFrame, dataset with images and captions, loads CSV if None
    - comments_trees: dict, dictionary of nx.Graphs, loads PKL if None
    - split: str, split of the data, 'train' by default
    """
    # Load data if not provided
    if data is None:
        data = pd.read_csv(f'dataset/{split}.csv')
    if comment_trees is None:
        with open(f'dataset/comments/{split}_comment_trees.pkl', 'rb') as f:
            comment_trees = pickle.load(f)
    
    # Choose a random ID if not provided
    if id_ is None:
        id_ = random.choice(data['id'])
    
    # Load image and caption
    row = data[data['id'] == id_].iloc[0]
    image_path = f"dataset/images/{row['id']}.jpg"
    image = plt.imread(image_path)
    
    # Load comments tree
    G = comment_trees.get(id_)
    if G is None:
        print("No comments tree found for this post.")
        return
    
    pos = nx.kamada_kawai_layout(G)
    labels = {node: f"{G.nodes[node]['author']}\n{G.nodes[node]['text']}"[:50] for node in G.nodes}
    
    # Plot
    _, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Image and caption
    axes[0].imshow(image)
    axes[0].axis('off')
    axes[0].set_title(f'{row["author"]}\n{row["clean_title"]}')
    
    # Comments tree
    nx.draw(G, pos, ax=axes[1], with_labels=True, labels=labels, node_size=1000, 
            node_color='skyblue', font_size=7, font_color='black', edge_color='gray', 
            width=1, alpha=0.7)
    
    # Draw the id_ one in red
    nx.draw(G, pos, nodelist=[id_], ax=axes[1], node_size=1000,
            node_color='red', font_size=7, font_color='black', edge_color='gray', 
            width=1, alpha=0.7)
    
    label_2_way = INT2LABEL['2_way_label'][row['2_way_label']]
    label_3_way = INT2LABEL['3_way_label'][row['3_way_label']]
    label_6_way = INT2LABEL['6_way_label'][row['6_way_label']]

    axes[1].set_title(f"Labels: {label_2_way} | {label_3_way} | {label_6_way}")
    
    plt.tight_layout()
    plt.show()
    return id_

### MISC ###
def extract_tar(in_file, out_dir):
    with tarfile.open(in_file, "r:bz2") as tar:
        with tqdm.tqdm(desc="Extrayendo", unit="file") as pbar:
            for member in tar:
                tar.extract(member, path=out_dir)
                pbar.update(1)

def debug(fn):
    def wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except:
            import pdb; pdb.set_trace()
    return wrapper

def load_images(image_paths):
    if isinstance(image_paths, str):
        image_paths = [image_paths]

    images = []
    for image_path in image_paths:
        try:
            image = PIL.Image.open(image_path)
            image = image.convert("RGB")
        except:
            print(f"Error loading image {image_path}.")
            image = PIL.Image.new("RGB", (224, 224), (255, 255, 255)) # Placeholder full white image

        images.append(image)

    return images

def comments_tree_to_text(graph, comment_format, node=None, level=0, parent=None, max_comment_words=None):
    if node is None:
        node = graph.graph['id']
    node_data = graph.nodes[node]

    # Author
    author = node_data.get('author', node) # use node id as author if not available

    text = str(node_data.get('text', '')).replace("\n", ". ")
    if max_comment_words is not None:
        text = " ".join(text.split()[:max_comment_words])

    # Score
    score = int(node_data.get('score', 0))

    body = comment_format.format(
        author=author,
        text=text,
        score=score,
    )
    indent = "  " * level + ("|- " if level > 0 else "")
    comments = indent + body + "\n"
    
    for child in graph.neighbors(node):
        if child == parent:
            continue
        comments += comments_tree_to_text(graph, comment_format, node=child, level=level+1, parent=node, max_comment_words=max_comment_words)
    
    return comments