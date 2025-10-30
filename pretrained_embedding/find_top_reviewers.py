"""
Find Top-10 Reviewers for a Research Paper
GPU-accelerated semantic similarity with sentence transformers
"""

import json
import torch
from sentence_transformers import SentenceTransformer, util
from collections import defaultdict
import re
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
import PyPDF2
import sys 
import os

nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def extract_text_from_pdf(pdf_path):
    try:
        text = ""
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
        return text
    except FileNotFoundError:
        print(f"ERROR: File not found: {pdf_path}")
        return None
    except Exception as e:
        print(f"ERROR: {e}")
        return None

def clean_text(text):
    if not text or text.strip() == "":
        return ""
    
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\d+', '', text)
    words = text.split()
    words = [word for word in words if word and word not in stop_words]
    words = [lemmatizer.lemmatize(word) for word in words]
    
    return ' '.join(words)

def find_top_reviewers(input_text, k=10, cache_path=None):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Default cache path
    if cache_path is None:
        cache_path = os.path.join(os.path.dirname(__file__), "embeddings_cache.pt")
    
    model = SentenceTransformer('all-mpnet-base-v2', device=device)
    cache = torch.load(cache_path)
    paper_embeddings = cache["embeddings"].to(device)
    authors = cache["authors"]
    
    cleaned_input = clean_text(input_text)
    query_embedding = model.encode(cleaned_input, convert_to_tensor=True, device=device)
    sim_scores = util.cos_sim(query_embedding, paper_embeddings)[0]
    
    author_scores = defaultdict(list)
    for score, author in zip(sim_scores, authors):
        author_scores[author].append(float(score))
    
    author_max_scores = {author: max(scores) for author, scores in author_scores.items()}
    top_k = sorted(author_max_scores.items(), key=lambda x: x[1], reverse=True)[:k]
    
    return top_k, author_scores

def display_results(top_k, detailed_scores):
    print(f"\n{'='*80}")
    print(f"TOP-{len(top_k)} RECOMMENDED REVIEWERS")
    print(f"{'='*80}\n")
    
    perfect_matches = [(author, score) for author, score in top_k if score >= 0.999]
    if perfect_matches:
        print(f"⚠️  {len(perfect_matches)} author(s) with perfect scores (paper in dataset):\n")
        for author, score in perfect_matches:
            print(f"   - {author} ({score:.6f})")
        print(f"{'='*80}\n")
    
    print(f"{'Rank':<6} {'Author':<40} {'Score':<12} {'Papers'}")
    print("-" * 80)
    
    for i, (author, score) in enumerate(top_k, 1):
        num_papers = len(detailed_scores[author])
        marker = " ★" if score >= 0.999 else ""
        print(f"{i:<6} {author:<40} {score:.6f}     {num_papers}{marker}")
    
    if perfect_matches:
        print("\n★ = Paper in dataset")
    
    print("\n" + "="*80)
    print("STATISTICS")
    print("="*80)
    scores = [score for _, score in top_k]
    print(f"Best:      {scores[0]:.6f}")
    print(f"10th:      {scores[-1]:.6f}")
    print(f"Range:     {scores[0] - scores[-1]:.6f}")
    print(f"Average:   {sum(scores)/len(scores):.6f}")
    
    non_perfect_scores = [score for _, score in top_k if score < 0.999]
    if non_perfect_scores and len(non_perfect_scores) < len(scores):
        print(f"\nExcluding perfect matches:")
        print(f"Best:      {max(non_perfect_scores):.6f}")
        print(f"Average:   {sum(non_perfect_scores)/len(non_perfect_scores):.6f}")
    
    print("="*80 + "\n")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        pdf_path = sys.argv[1]
    else:
        print("Enter PDF path:")
        pdf_path = input().strip().strip('"').strip("'")
    
    if not os.path.exists(pdf_path):
        print(f"ERROR: File not found: {pdf_path}")
        sys.exit(1)
    
    paper_text = extract_text_from_pdf(pdf_path)
    if not paper_text:
        sys.exit(1)
    
    if len(paper_text.strip()) < 100:
        print(f"WARNING: Short text. Might be scanned PDF.")
    
    top_10, all_scores = find_top_reviewers(paper_text, k=10)
    display_results(top_10, all_scores)
    
    print("TOP-3 DETAILS:")
    print("="*80)
    for i, (author, _) in enumerate(top_10[:3], 1):
        scores = all_scores[author]
        mean = sum(scores)/len(scores)
        stddev = (sum((s - mean)**2 for s in scores) / len(scores))**0.5
        print(f"{i}. {author}")
        print(f"   Papers: {len(scores)} | Max: {max(scores):.4f} | Mean: {mean:.4f} | StdDev: {stddev:.4f}")
    
    print("="*80 + "\n")
