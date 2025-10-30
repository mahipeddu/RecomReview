"""
Topic Modeling using LDA (Latent Dirichlet Allocation)
Implements reviewer recommendation based on topic similarity
"""

import json
import numpy as np
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
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

def build_lda_model(documents, n_topics=20, n_top_words=15, random_state=42):
    """
    Build LDA model from documents
    
    Args:
        documents: List of document texts
        n_topics: Number of topics to extract
        n_top_words: Number of top words per topic
        random_state: Random seed for reproducibility
    
    Returns:
        lda_model, vectorizer, doc_topic_dist, feature_names
    """
    print(f"Building LDA model with {n_topics} topics...")
    
    vectorizer = CountVectorizer(
        max_df=0.95,  # Ignore terms appearing in > 95% of documents
        min_df=2,      # Ignore terms appearing in < 2 documents
        max_features=5000,
        ngram_range=(1, 2)  # Include unigrams and bigrams
    )
    
    doc_term_matrix = vectorizer.fit_transform(documents)
    
    lda_model = LatentDirichletAllocation(
        n_components=n_topics,
        max_iter=50,
        learning_method='online',
        random_state=random_state,
        batch_size=128,
        n_jobs=-1,
        verbose=1
    )
    
    doc_topic_dist = lda_model.fit_transform(doc_term_matrix)
    
    feature_names = vectorizer.get_feature_names_out()
    
    return lda_model, vectorizer, doc_topic_dist, feature_names

def display_topics(lda_model, feature_names, n_top_words=15):
    """Display top words for each topic"""
    print("\n" + "="*80)
    print("DISCOVERED TOPICS")
    print("="*80)
    
    for topic_idx, topic in enumerate(lda_model.components_):
        top_word_indices = topic.argsort()[-n_top_words:][::-1]
        top_words = [feature_names[i] for i in top_word_indices]
        top_weights = [topic[i] for i in top_word_indices]
        
        print(f"\nTopic #{topic_idx + 1}:")
        print(f"  Words: {', '.join(top_words[:10])}")
        print(f"  Top 3 weights: {top_weights[:3]}")

def compute_topic_similarity(query_topic_dist, doc_topic_distributions, method='cosine'):
    """
    Compute similarity between query and documents based on topic distributions
    
    Args:
        query_topic_dist: Topic distribution of query document
        doc_topic_distributions: Topic distributions of all documents
        method: Similarity metric ('cosine', 'kl_divergence', 'hellinger')
    
    Returns:
        Array of similarity scores
    """
    if method == 'cosine':
        from sklearn.metrics.pairwise import cosine_similarity
        similarities = cosine_similarity(query_topic_dist.reshape(1, -1), 
                                        doc_topic_distributions)[0]
    
    elif method == 'kl_divergence':
        epsilon = 1e-10
        query_dist = query_topic_dist + epsilon
        doc_dists = doc_topic_distributions + epsilon
        
        query_dist = query_dist / query_dist.sum()
        doc_dists = doc_dists / doc_dists.sum(axis=1, keepdims=True)
        
        kl_divs = np.sum(query_dist * np.log(query_dist / doc_dists), axis=1)
        similarities = -kl_divs  # Negative because lower KL is better
    
    elif method == 'hellinger':
        sqrt_query = np.sqrt(query_topic_dist)
        sqrt_docs = np.sqrt(doc_topic_distributions)
        
        similarities = 1 - (1/np.sqrt(2)) * np.sqrt(
            np.sum((sqrt_query - sqrt_docs)**2, axis=1)
        )
    
    else:
        raise ValueError(f"Unknown similarity method: {method}")
    
    return similarities

def find_top_reviewers_lda(input_text, lda_model, vectorizer, 
                           author_topic_dists, authors, 
                           k=10, similarity_method='cosine'):
    """
    Find top reviewers based on LDA topic modeling
    
    Args:
        input_text: Query paper text
        lda_model: Trained LDA model
        vectorizer: Fitted CountVectorizer
        author_topic_dists: Dictionary mapping authors to their topic distributions
        authors: List of all authors
        k: Number of top reviewers to return
        similarity_method: Method to compute similarity
    
    Returns:
        List of (author, score) tuples
    """
    cleaned_input = clean_text(input_text)
    
    query_dtm = vectorizer.transform([cleaned_input])
    
    query_topic_dist = lda_model.transform(query_dtm)[0]
    
    author_avg_topics = {}
    for author, topic_dists in author_topic_dists.items():
        author_avg_topics[author] = np.mean(topic_dists, axis=0)
    
    author_scores = {}
    for author in authors:
        if author in author_avg_topics:
            author_topic_dist = author_avg_topics[author]
            similarity = compute_topic_similarity(
                query_topic_dist, 
                author_topic_dist.reshape(1, -1),
                method=similarity_method
            )[0]
            author_scores[author] = similarity
    
    top_k = sorted(author_scores.items(), key=lambda x: x[1], reverse=True)[:k]
    
    return top_k, author_avg_topics

def save_lda_model(lda_model, vectorizer, doc_topic_dist, authors, 
                   model_path='lda_model.pkl'):
    """Save trained LDA model and related data"""
    model_data = {
        'lda_model': lda_model,
        'vectorizer': vectorizer,
        'doc_topic_dist': doc_topic_dist,
        'authors': authors
    }
    
    with open(model_path, 'wb') as f:
        pickle.dump(model_data, f)
    
    print(f"Model saved to {model_path}")

def load_lda_model(model_path='lda_model.pkl'):
    """Load trained LDA model"""
    try:
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        return (model_data['lda_model'], model_data['vectorizer'],
                model_data['doc_topic_dist'], model_data['authors'])
    except Exception as e:
        return None, None, None, None

def display_results(top_k, author_avg_topics, query_topic_dist=None):
    """Display the top-k reviewers and their statistics"""
    print(f"\n{'='*80}")
    print(f"TOP-{len(top_k)} RECOMMENDED REVIEWERS (LDA-based)")
    print(f"{'='*80}\n")
    
    print(f"{'Rank':<6} {'Author':<40} {'Score':<12}")
    print("-" * 80)
    
    for i, (author, score) in enumerate(top_k, 1):
        print(f"{i:<6} {author:<40} {score:.6f}")
    
    print("\n" + "="*80)
    print("STATISTICS")
    print("="*80)
    
    scores = [score for _, score in top_k]
    print(f"Best Score:       {scores[0]:.6f}")
    print(f"10th Score:       {scores[-1]:.6f}")
    print(f"Score Range:      {scores[0] - scores[-1]:.6f}")
    print(f"Average Score:    {sum(scores)/len(scores):.6f}")
    print("="*80 + "\n")
    
    if query_topic_dist is not None:
        print("DOMINANT TOPICS:")
        print("="*80)
        print(f"\nQuery Paper Top Topics:")
        top_topics = np.argsort(query_topic_dist)[-3:][::-1]
        for topic_idx in top_topics:
            print(f"  Topic #{topic_idx + 1}: {query_topic_dist[topic_idx]:.4f}")
        
        print(f"\nTop-3 Reviewers' Dominant Topics:")
        for i, (author, _) in enumerate(top_k[:3], 1):
            print(f"\n{i}. {author}")
            author_dist = author_avg_topics[author]
            top_topics = np.argsort(author_dist)[-3:][::-1]
            for topic_idx in top_topics:
                print(f"   Topic #{topic_idx + 1}: {author_dist[topic_idx]:.4f}")
        print("="*80 + "\n")

def train_and_save_model(dataset_path='cleaned_dataset.json', 
                         n_topics=20, 
                         model_path='lda_model.pkl'):
    """Train LDA model on dataset and save it"""
    print("Loading dataset...")
    data = load_dataset(dataset_path)
    
    if not data:
        return
    
    documents = []
    authors = []
    doc_to_author = {}
    
    for idx, item in enumerate(data):
        documents.append(item['text_content'])
        authors.append(item['author_name'])
        doc_to_author[idx] = item['author_name']
    
    print(f"Loaded {len(documents)} documents from {len(set(authors))} authors")
    
    lda_model, vectorizer, doc_topic_dist, feature_names = build_lda_model(
        documents, n_topics=n_topics
    )
    
    display_topics(lda_model, feature_names)
    
    author_topic_dists = defaultdict(list)
    for idx, author in enumerate(authors):
        author_topic_dists[author].append(doc_topic_dist[idx])
    
    save_lda_model(lda_model, vectorizer, doc_topic_dist, authors, model_path)
    
    author_topics_path = model_path.replace('.pkl', '_author_topics.pkl')
    with open(author_topics_path, 'wb') as f:
        pickle.dump(dict(author_topic_dists), f)
    
    print(f"\nAuthor topic distributions saved to {author_topics_path}")
    print("\nTraining complete!")
    
    return lda_model, vectorizer, author_topic_dists, list(set(authors))

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='LDA-based Reviewer Recommendation')
    parser.add_argument('--train', action='store_true', help='Train new LDA model')
    parser.add_argument('--pdf', type=str, help='Path to input PDF for recommendation')
    parser.add_argument('--topics', type=int, default=20, help='Number of topics (default: 20)')
    parser.add_argument('--k', type=int, default=10, help='Number of reviewers to recommend (default: 10)')
    parser.add_argument('--method', type=str, default='cosine', 
                       choices=['cosine', 'kl_divergence', 'hellinger'],
                       help='Similarity method (default: cosine)')
    parser.add_argument('--model', type=str, default='lda_model.pkl', 
                       help='Path to model file')
    
    args = parser.parse_args()
    
    if args.train:
        print("Training new LDA model...")
        train_and_save_model(n_topics=args.topics, model_path=args.model)
    
    elif args.pdf:
        print("Loading LDA model...")
        lda_model, vectorizer, doc_topic_dist, authors = load_lda_model(args.model)
        
        if lda_model is None:
            print("Error: Could not load model. Train a model first with --train")
            sys.exit(1)
        
        author_topics_path = args.model.replace('.pkl', '_author_topics.pkl')
        with open(author_topics_path, 'rb') as f:
            author_topic_dists = pickle.load(f)
        
        print(f"Processing PDF: {args.pdf}")
        paper_text = extract_text_from_pdf(args.pdf)
        
        if not paper_text:
            sys.exit(1)
        
        if len(paper_text.strip()) < 100:
            print("WARNING: Short text extracted. Might be a scanned PDF.")
        
        top_k, author_avg_topics = find_top_reviewers_lda(
            paper_text, 
            lda_model, 
            vectorizer,
            author_topic_dists,
            list(set(authors)),
            k=args.k,
            similarity_method=args.method
        )
        
        cleaned_input = clean_text(paper_text)
        query_dtm = vectorizer.transform([cleaned_input])
        query_topic_dist = lda_model.transform(query_dtm)[0]
        
        display_results(top_k, author_avg_topics, query_topic_dist)
    
    else:
        parser.print_help()
        print("\nExamples:")
        print("  # Train model:")
        print("  python topic_modeling_lda.py --train --topics 20")
        print("\n  # Find reviewers:")
        print("  python topic_modeling_lda.py --pdf 'path/to/paper.pdf' --k 10 --method cosine")
