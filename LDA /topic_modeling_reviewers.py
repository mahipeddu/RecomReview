"""
Topic Modeling Based Reviewer Recommendation System
Using LDA (Latent Dirichlet Allocation) for topic extraction and similarity
"""

import json
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation, NMF
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
import PyPDF2
import sys
import os
import pickle

def extract_text_from_pdf(pdf_path):
    try:
        text = ""
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            num_pages = len(pdf_reader.pages)
            print(f"Extracting {num_pages} pages from PDF...")
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
        return text
    except Exception as e:
        print(f"ERROR: {e}")
        return None

def load_dataset():
    """Load the cleaned dataset"""
    print("Loading dataset...")
    
    # Try different file locations
    possible_paths = [
        "cleaned_dataset.json",
        "intermediate_files/cleaned_dataset.json"
    ]
    
    data = None
    for path in possible_paths:
        try:
            with open(path) as f:
                data = json.load(f)
            break
        except FileNotFoundError:
            continue
    
    if data is None:
        print("ERROR: cleaned_dataset.json not found!")
        print("Please run the data preprocessing script first.")
        sys.exit(1)
    
    texts = [d.get("clean_text", d.get("text_content", "")) for d in data]
    authors = [d["author_name"] for d in data]
    
    print(f"Loaded {len(texts)} papers from {len(set(authors))} authors")
    return texts, authors

def train_lda_model(texts, n_topics=20, save_model=True):
    """Train LDA model on the corpus"""
    print(f"\nTraining LDA model with {n_topics} topics...")
    
    # Create document-term matrix
    print("Creating document-term matrix...")
    vectorizer = CountVectorizer(
        max_features=5000,
        max_df=0.95,
        min_df=2,
        stop_words='english'
    )
    doc_term_matrix = vectorizer.fit_transform(texts)
    
    print(f"Vocabulary size: {len(vectorizer.vocabulary_)}")
    print(f"Document-term matrix shape: {doc_term_matrix.shape}")
    
    # Train LDA model
    print(f"Training LDA (this may take a few minutes)...")
    lda_model = LatentDirichletAllocation(
        n_components=n_topics,
        max_iter=50,
        learning_method='online',
        random_state=42,
        n_jobs=-1,
        verbose=1
    )
    
    doc_topic_distributions = lda_model.fit_transform(doc_term_matrix)
    
    # Save model if requested
    if save_model:
        print("\nSaving LDA model and vectorizer...")
        with open('lda_model.pkl', 'wb') as f:
            pickle.dump(lda_model, f)
        with open('lda_vectorizer.pkl', 'wb') as f:
            pickle.dump(vectorizer, f)
        with open('lda_doc_topics.npy', 'wb') as f:
            np.save(f, doc_topic_distributions)
        print("Model saved!")
    
    return lda_model, vectorizer, doc_topic_distributions

def train_nmf_model(texts, n_topics=20, save_model=True):
    """Train NMF model on the corpus"""
    print(f"\nTraining NMF model with {n_topics} topics...")
    
    # Create TF-IDF matrix (NMF works better with TF-IDF)
    print("Creating TF-IDF matrix...")
    vectorizer = TfidfVectorizer(
        max_features=5000,
        max_df=0.95,
        min_df=2,
        stop_words='english'
    )
    tfidf_matrix = vectorizer.fit_transform(texts)
    
    print(f"Vocabulary size: {len(vectorizer.vocabulary_)}")
    print(f"TF-IDF matrix shape: {tfidf_matrix.shape}")
    
    # Train NMF model
    print(f"Training NMF...")
    nmf_model = NMF(
        n_components=n_topics,
        max_iter=500,
        random_state=42,
        init='nndsvda'
    )
    
    doc_topic_distributions = nmf_model.fit_transform(tfidf_matrix)
    
    # Save model if requested
    if save_model:
        print("\nSaving NMF model and vectorizer...")
        with open('nmf_model.pkl', 'wb') as f:
            pickle.dump(nmf_model, f)
        with open('nmf_vectorizer.pkl', 'wb') as f:
            pickle.dump(vectorizer, f)
        with open('nmf_doc_topics.npy', 'wb') as f:
            np.save(f, doc_topic_distributions)
        print("Model saved!")
    
    return nmf_model, vectorizer, doc_topic_distributions

def display_topics(model, vectorizer, n_words=10, model_name="LDA"):
    """Display top words for each topic"""
    print(f"\n{'='*80}")
    print(f"TOP {n_words} WORDS PER TOPIC ({model_name})")
    print(f"{'='*80}\n")
    
    feature_names = vectorizer.get_feature_names_out()
    
    for topic_idx, topic in enumerate(model.components_):
        top_indices = topic.argsort()[-n_words:][::-1]
        top_words = [feature_names[i] for i in top_indices]
        print(f"Topic {topic_idx + 1}: {', '.join(top_words)}")

def find_reviewers_topic_modeling(input_text, model_type='lda', k=10):
    """
    Find top-k reviewers using topic modeling
    
    Args:
        input_text: Raw text from the paper
        model_type: 'lda' or 'nmf'
        k: Number of reviewers to return
    """
    print(f"\n{'='*80}")
    print(f"Finding Top-{k} Reviewers (Method: {model_type.upper()} Topic Modeling)")
    print(f"{'='*80}\n")
    
    # Load pre-trained model or train new one
    try:
        if model_type == 'lda':
            with open('lda_model.pkl', 'rb') as f:
                model = pickle.load(f)
            with open('lda_vectorizer.pkl', 'rb') as f:
                vectorizer = pickle.load(f)
            doc_topic_distributions = np.load('lda_doc_topics.npy')
            print("Loaded pre-trained LDA model")
        else:  # nmf
            with open('nmf_model.pkl', 'rb') as f:
                model = pickle.load(f)
            with open('nmf_vectorizer.pkl', 'rb') as f:
                vectorizer = pickle.load(f)
            doc_topic_distributions = np.load('nmf_doc_topics.npy')
            print("Loaded pre-trained NMF model")
    except FileNotFoundError:
        print(f"No pre-trained {model_type.upper()} model found. Please train first!")
        return None, None
    
    # Load authors
    texts, authors = load_dataset()
    
    # Transform input text to topic distribution
    print("Computing topic distribution for input paper...")
    if model_type == 'lda':
        input_vector = vectorizer.transform([input_text])
        input_topic_dist = model.transform(input_vector)
    else:  # nmf
        input_vector = vectorizer.transform([input_text])
        input_topic_dist = model.transform(input_vector)
    
    # Display dominant topics in input
    dominant_topics = input_topic_dist[0].argsort()[-3:][::-1]
    print(f"\nDominant topics in input paper: {dominant_topics + 1}")
    for topic_idx in dominant_topics:
        print(f"  Topic {topic_idx + 1}: {input_topic_dist[0][topic_idx]:.4f}")
    
    # Compute cosine similarity between input and all papers
    print("\nComputing similarities...")
    similarities = cosine_similarity(input_topic_dist, doc_topic_distributions)[0]
    
    # Aggregate by author (using MAX)
    author_scores = defaultdict(list)
    for score, author in zip(similarities, authors):
        author_scores[author].append(float(score))
    
    author_max_scores = {author: max(scores) for author, scores in author_scores.items()}
    top_k = sorted(author_max_scores.items(), key=lambda x: x[1], reverse=True)[:k]
    
    return top_k, author_scores

def display_results(top_k, detailed_scores, model_type):
    """Display results"""
    print(f"\n{'='*80}")
    print(f"TOP-{len(top_k)} RECOMMENDED REVIEWERS ({model_type.upper()} Topic Modeling)")
    print(f"{'='*80}\n")
    
    perfect_matches = [(author, score) for author, score in top_k if score >= 0.999]
    if perfect_matches:
        print(f"⚠️  WARNING: {len(perfect_matches)} author(s) have perfect scores")
        print(f"   Paper likely in dataset (co-authors):\n")
        for author, score in perfect_matches:
            print(f"   - {author} ({score:.6f})")
        print(f"{'='*80}\n")
    
    print(f"{'Rank':<6} {'Author Name':<40} {'Score':<12} {'Papers'}")
    print("-" * 80)
    
    for i, (author, score) in enumerate(top_k, 1):
        num_papers = len(detailed_scores[author])
        marker = " ★" if score >= 0.999 else ""
        print(f"{i:<6} {author:<40} {score:.6f}     {num_papers}{marker}")
    
    if perfect_matches:
        print("\n★ = Perfect match (paper in dataset)")
    
    print("\n" + "="*80)
    print("STATISTICS")
    print("="*80)
    scores = [score for _, score in top_k]
    print(f"Best score:     {scores[0]:.6f} ({top_k[0][0]})")
    print(f"10th score:     {scores[-1]:.6f} ({top_k[-1][0]})")
    print(f"Difference:     {scores[0] - scores[-1]:.6f}")
    print(f"Average:        {sum(scores)/len(scores):.6f}")
    
    non_perfect_scores = [score for _, score in top_k if score < 0.999]
    if non_perfect_scores and len(non_perfect_scores) < len(scores):
        print(f"\nEXCLUDING PERFECT MATCHES:")
        print(f"Best score:     {max(non_perfect_scores):.6f}")
        print(f"Average:        {sum(non_perfect_scores)/len(non_perfect_scores):.6f}")
    
    print("="*80 + "\n")

if __name__ == "__main__":
    print("\n" + "="*80)
    print("TOPIC MODELING BASED REVIEWER RECOMMENDATION SYSTEM")
    print("="*80)
    
    # Check if models exist, otherwise train
    if not os.path.exists('lda_model.pkl'):
        print("\nNo trained models found. Training LDA and NMF models...")
        texts, authors = load_dataset()
        
        # Train LDA
        lda_model, lda_vectorizer, lda_doc_topics = train_lda_model(texts, n_topics=20)
        display_topics(lda_model, lda_vectorizer, n_words=8, model_name="LDA")
        
        # Train NMF
        nmf_model, nmf_vectorizer, nmf_doc_topics = train_nmf_model(texts, n_topics=20)
        display_topics(nmf_model, nmf_vectorizer, n_words=8, model_name="NMF")
        
        print("\n" + "="*80)
        print("Models trained and saved!")
        print("Run the script again with a PDF path to find reviewers.")
        print("="*80)
        sys.exit(0)
    
    # Get PDF path
    if len(sys.argv) > 1:
        pdf_path = sys.argv[1]
    else:
        print("\nEnter PDF path:")
        pdf_path = input().strip().strip('"').strip("'")
    
    if not os.path.exists(pdf_path):
        print(f"\nERROR: File not found: {pdf_path}")
        print("\nUsage: python3 topic_modeling_reviewers.py <path_to_pdf>")
        sys.exit(1)
    
    # Extract text
    paper_text = extract_text_from_pdf(pdf_path)
    if not paper_text:
        print("Failed to extract text. Exiting.")
        sys.exit(1)
    
    print(f"\nAnalyzing: {os.path.basename(pdf_path)}\n")
    
    # Find reviewers using both LDA and NMF
    print("\n" + "="*80)
    print("METHOD 1: LDA (Latent Dirichlet Allocation)")
    print("="*80)
    lda_reviewers, lda_scores = find_reviewers_topic_modeling(paper_text, model_type='lda', k=10)
    if lda_reviewers:
        display_results(lda_reviewers, lda_scores, 'lda')
    
    print("\n" + "="*80)
    print("METHOD 2: NMF (Non-negative Matrix Factorization)")
    print("="*80)
    nmf_reviewers, nmf_scores = find_reviewers_topic_modeling(paper_text, model_type='nmf', k=10)
    if nmf_reviewers:
        display_results(nmf_reviewers, nmf_scores, 'nmf')
    
    # Compare methods
    if lda_reviewers and nmf_reviewers:
        print("\n" + "="*80)
        print("COMPARISON: LDA vs NMF")
        print("="*80)
        
        lda_top5 = set([author for author, _ in lda_reviewers[:5]])
        nmf_top5 = set([author for author, _ in nmf_reviewers[:5]])
        
        overlap = lda_top5 & nmf_top5
        print(f"\nTop-5 overlap: {len(overlap)}/5 authors")
        if overlap:
            print(f"Common authors: {', '.join(overlap)}")
        
        print("\n" + "="*80)
        print(f"COMPLETE: {os.path.basename(pdf_path)}")
        print("="*80 + "\n")
