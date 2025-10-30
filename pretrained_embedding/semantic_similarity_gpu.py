"""
GPU-Accelerated Semantic Similarity for Reviewer Recommendation
Using Sentence Transformers (all-mpnet-base-v2) with caching
"""

import json
import torch
from sentence_transformers import SentenceTransformer, util
from collections import defaultdict
from tqdm import tqdm
import os

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")
if device == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}")

with open("cleaned_dataset.json") as f:
    data = json.load(f)

print(f"Loaded {len(data)} papers")

model = SentenceTransformer('all-mpnet-base-v2', device=device)
print("Model loaded successfully!")

cache_file = "embeddings_cache.pt"

if os.path.exists(cache_file):
    cache = torch.load(cache_file)
    embeddings = cache["embeddings"].to(device)
    authors = cache["authors"]
    print(f"Loaded {len(embeddings)} cached embeddings")
else:
    print("\nEncoding all papers (this will take 1-2 minutes)...")
    texts = [d.get("clean_text", d.get("text_content", "")) for d in data]
    authors = [d["author_name"] for d in data]
    
    embeddings = model.encode(
        texts,
        batch_size=32,
        convert_to_tensor=True,
        show_progress_bar=True,
        device=device
    )
    
    torch.save({
        "embeddings": embeddings,
        "authors": authors
    }, cache_file)
    print("Cache saved successfully!")

print(f"\nEmbedding shape: {embeddings.shape}")
print(f"Number of unique authors: {len(set(authors))}")
