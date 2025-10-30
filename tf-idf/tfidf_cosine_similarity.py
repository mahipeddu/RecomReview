"""
TF-IDF with Cosine Similarity for Reviewer Recommendation
Implements a baseline approach using TF-IDF vectorization and cosine similarity
"""

import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
import PyPDF2
import re
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
import sys
import os
import pickle

nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def extract_text_from_pdf(pdf_path):
    """Extract text content from PDF file"""
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
    """Clean and preprocess text"""
    if not text or text.strip() == "":
        return ""
    
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\d+', '', text)
    words = text.split()
    words = [word for word in words if word and word not in stop_words and len(word) > 2]
    words = [lemmatizer.lemmatize(word) for word in words]
    
    return ' '.join(words)

def load_dataset(dataset_path='cleaned_dataset.json'):
    """Load the cleaned dataset"""
    try:
        with open(dataset_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    except Exception as e:
        return None

def build_tfidf_model(documents, max_features=5000, ngram_range=(1, 2)):
    """
    Build TF-IDF model from documents
    
    Args:
        documents: List of document texts
        max_features: Maximum number of features (vocabulary size)
        ngram_range: Range of n-grams to extract
    
    Returns:
        vectorizer, tfidf_matrix
    """
    print(f"Building TF-IDF model with max_features={max_features}, ngram_range={ngram_range}...")
    
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        max_df=0.95,  # Ignore terms appearing in > 95% of documents
        min_df=2,     # Ignore terms appearing in < 2 documents
        sublinear_tf=True,  # Use logarithmic term frequency
        use_idf=True,
        smooth_idf=True
    )
    
    tfidf_matrix = vectorizer.fit_transform(documents)
    
    print(f"TF-IDF matrix shape: {tfidf_matrix.shape}")
    print(f"Vocabulary size: {len(vectorizer.get_feature_names_out())}")
    
    return vectorizer, tfidf_matrix

def compute_cosine_similarities(query_vector, doc_vectors):
    """
    Compute cosine similarities between query and documents
    
    Args:
        query_vector: TF-IDF vector of query document
        doc_vectors: TF-IDF matrix of all documents
    
    Returns:
        Array of cosine similarity scores
    """
    similarities = cosine_similarity(query_vector, doc_vectors)[0]
    return similarities

def aggregate_author_scores(similarities, authors, method='max'):
    """
    Aggregate similarity scores by author
    
    Args:
        similarities: Array of document similarity scores
        authors: List of author names corresponding to documents
        method: Aggregation method ('max', 'mean', 'weighted_mean')
    
    Returns:
        Dictionary of author scores
    """
    author_scores = defaultdict(list)
    
    for idx, author in enumerate(authors):
        author_scores[author].append(similarities[idx])
    
    author_aggregated = {}
    for author, scores in author_scores.items():
        if method == 'max':
            author_aggregated[author] = max(scores)
        elif method == 'mean':
            author_aggregated[author] = np.mean(scores)
        elif method == 'weighted_mean':
            weights = np.array(scores)
            author_aggregated[author] = np.average(scores, weights=weights)
        else:
            raise ValueError(f"Unknown aggregation method: {method}")
    
    return author_aggregated

def get_top_terms(vectorizer, tfidf_matrix, author_indices, top_n=10):
    """
    Get top terms for an author based on their papers
    
    Args:
        vectorizer: Fitted TF-IDF vectorizer
        tfidf_matrix: TF-IDF matrix of all documents
        author_indices: List of document indices for the author
        top_n: Number of top terms to return
    
    Returns:
        List of (term, score) tuples
    """
    author_vector = tfidf_matrix[author_indices].mean(axis=0).A1
    
    feature_names = vectorizer.get_feature_names_out()
    
    top_indices = author_vector.argsort()[-top_n:][::-1]
    top_terms = [(feature_names[i], author_vector[i]) for i in top_indices]
    
    return top_terms

def find_top_reviewers_tfidf(input_text, vectorizer, tfidf_matrix, 
                             authors, k=10, aggregation='max'):
    """
    Find top reviewers using TF-IDF and cosine similarity
    
    Args:
        input_text: Query paper text
        vectorizer: Fitted TF-IDF vectorizer
        tfidf_matrix: TF-IDF matrix of all documents
        authors: List of author names
        k: Number of top reviewers to return
        aggregation: Method to aggregate multiple papers per author
    
    Returns:
        List of (author, score) tuples
    """
    cleaned_input = clean_text(input_text)
    
    query_vector = vectorizer.transform([cleaned_input])
    
    similarities = compute_cosine_similarities(query_vector, tfidf_matrix)
    
    author_scores = aggregate_author_scores(similarities, authors, method=aggregation)
    
    top_k = sorted(author_scores.items(), key=lambda x: x[1], reverse=True)[:k]
    
    detailed_scores = defaultdict(list)
    for idx, author in enumerate(authors):
        detailed_scores[author].append(similarities[idx])
    
    return top_k, detailed_scores

def save_tfidf_model(vectorizer, tfidf_matrix, authors, model_path='tfidf_model.pkl'):
    """Save TF-IDF model and related data"""
    model_data = {
        'vectorizer': vectorizer,
        'tfidf_matrix': tfidf_matrix,
        'authors': authors
    }
    
    with open(model_path, 'wb') as f:
        pickle.dump(model_data, f)
    
    print(f"TF-IDF model saved to {model_path}")

def load_tfidf_model(model_path='tfidf_model.pkl'):
    """Load TF-IDF model"""
    try:
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        return (model_data['vectorizer'], model_data['tfidf_matrix'], 
                model_data['authors'])
    except Exception as e:
        return None, None, None

def display_results(top_k, detailed_scores, query_terms=None):
    """Display the top-k reviewers and their statistics"""
    print(f"\n{'='*80}")
    print(f"TOP-{len(top_k)} RECOMMENDED REVIEWERS (TF-IDF + Cosine Similarity)")
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
    
    if query_terms:
        print("TOP QUERY TERMS:")
        print("="*80)
        for i, (term, score) in enumerate(query_terms[:15], 1):
            print(f"{i:2d}. {term:<25} {score:.6f}")
        print("="*80 + "\n")
    
    print("TOP-3 REVIEWER DETAILS:")
    print("="*80)
    for i, (author, _) in enumerate(top_k[:3], 1):
        scores = detailed_scores[author]
        mean = sum(scores)/len(scores)
        stddev = (sum((s - mean)**2 for s in scores) / len(scores))**0.5
        print(f"{i}. {author}")
        print(f"   Papers: {len(scores)} | Max: {max(scores):.4f} | Mean: {mean:.4f} | StdDev: {stddev:.4f}")
    
    print("="*80 + "\n")

def get_query_top_terms(vectorizer, query_vector, top_n=15):
    """Get top terms from query vector"""
    feature_names = vectorizer.get_feature_names_out()
    query_array = query_vector.toarray()[0]
    
    top_indices = query_array.argsort()[-top_n:][::-1]
    top_terms = [(feature_names[i], query_array[i]) for i in top_indices if query_array[i] > 0]
    
    return top_terms

def train_and_save_model(dataset_path='cleaned_dataset.json', 
                         max_features=5000,
                         ngram_range=(1, 2),
                         model_path='tfidf_model.pkl'):
    """Train TF-IDF model on dataset and save it"""
    print("Loading dataset...")
    data = load_dataset(dataset_path)
    
    if not data:
        return
    
    documents = []
    authors = []
    
    for item in data:
        documents.append(item['text_content'])
        authors.append(item['author_name'])
    
    print(f"Loaded {len(documents)} documents from {len(set(authors))} authors")
    
    vectorizer, tfidf_matrix = build_tfidf_model(
        documents, 
        max_features=max_features,
        ngram_range=ngram_range
    )
    
    save_tfidf_model(vectorizer, tfidf_matrix, authors, model_path)
    
    print("\n" + "="*80)
    print("MODEL STATISTICS")
    print("="*80)
    print(f"Total documents: {len(documents)}")
    print(f"Unique authors: {len(set(authors))}")
    print(f"Vocabulary size: {len(vectorizer.get_feature_names_out())}")
    sparsity = (1.0 - tfidf_matrix.nnz / (tfidf_matrix.shape[0] * tfidf_matrix.shape[1])) * 100
    print(f"Matrix sparsity: {sparsity:.2f}%")
    print("="*80 + "\n")
    
    print("Training complete!")
    
    return vectorizer, tfidf_matrix, authors

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='TF-IDF + Cosine Similarity Reviewer Recommendation')
    parser.add_argument('--train', action='store_true', help='Train new TF-IDF model')
    parser.add_argument('--pdf', type=str, help='Path to input PDF for recommendation')
    parser.add_argument('--max-features', type=int, default=5000, 
                       help='Maximum vocabulary size (default: 5000)')
    parser.add_argument('--ngram', type=int, nargs=2, default=[1, 2],
                       help='N-gram range (default: 1 2)')
    parser.add_argument('--k', type=int, default=10, 
                       help='Number of reviewers to recommend (default: 10)')
    parser.add_argument('--aggregation', type=str, default='max',
                       choices=['max', 'mean', 'weighted_mean'],
                       help='Score aggregation method (default: max)')
    parser.add_argument('--model', type=str, default='tfidf_model.pkl',
                       help='Path to model file')
    
    args = parser.parse_args()
    
    if args.train:
        print("Training new TF-IDF model...")
        train_and_save_model(
            max_features=args.max_features,
            ngram_range=tuple(args.ngram),
            model_path=args.model
        )
    
    elif args.pdf:
        print("Loading TF-IDF model...")
        vectorizer, tfidf_matrix, authors = load_tfidf_model(args.model)
        
        if vectorizer is None:
            print("Error: Could not load model. Train a model first with --train")
            sys.exit(1)
        
        print(f"Processing PDF: {args.pdf}")
        paper_text = extract_text_from_pdf(args.pdf)
        
        if not paper_text:
            sys.exit(1)
        
        if len(paper_text.strip()) < 100:
            print("WARNING: Short text extracted. Might be a scanned PDF.")
        
        top_k, detailed_scores = find_top_reviewers_tfidf(
            paper_text,
            vectorizer,
            tfidf_matrix,
            authors,
            k=args.k,
            aggregation=args.aggregation
        )
        
        cleaned_input = clean_text(paper_text)
        query_vector = vectorizer.transform([cleaned_input])
        query_terms = get_query_top_terms(vectorizer, query_vector)
        
        display_results(top_k, detailed_scores, query_terms)
    
    else:
        parser.print_help()
        print("\nExamples:")
        print("  # Train model:")
        print("  python3 tfidf_cosine_similarity.py --train --max-features 5000")
        print("\n  # Find reviewers:")
        print("  python3 tfidf_cosine_similarity.py --pdf 'path/to/paper.pdf' --k 10 --aggregation max")
