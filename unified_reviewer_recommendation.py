"""
Unified Reviewer Recommendation System
Combines LDA, Pretrained Embeddings, and TF-IDF to recommend top 10 reviewers for a given PDF
"""

import json
import os
import sys
import pickle
import numpy as np
import torch
import PyPDF2
import re
import string
from collections import defaultdict
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
from sentence_transformers import SentenceTransformer, util
from sklearn.metrics.pairwise import cosine_similarity

# Download NLTK data
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


def get_lda_recommendations(paper_text, lda_model_path='LDA /lda_model.pkl', 
                           author_topics_path='LDA /lda_model_author_topics.pkl', k=10):
    """Get top-k recommendations using LDA"""
    print("\n" + "="*80)
    print("LDA METHOD")
    print("="*80)
    
    try:
        # Load LDA model
        with open(lda_model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        lda_model = model_data['lda_model']
        vectorizer = model_data['vectorizer']
        
        # Load author topic distributions
        with open(author_topics_path, 'rb') as f:
            author_topic_dists = pickle.load(f)
        
        # Process query
        cleaned_input = clean_text(paper_text)
        query_dtm = vectorizer.transform([cleaned_input])
        query_topic_dist = lda_model.transform(query_dtm)[0]
        
        # Compute author average topic distributions
        author_avg_topics = {}
        for author, topic_dists in author_topic_dists.items():
            author_avg_topics[author] = np.mean(topic_dists, axis=0)
        
        # Compute similarities
        author_scores = {}
        for author, author_topic_dist in author_avg_topics.items():
            similarity = cosine_similarity(
                query_topic_dist.reshape(1, -1), 
                author_topic_dist.reshape(1, -1)
            )[0][0]
            author_scores[author] = similarity
        
        # Get top-k
        top_k = sorted(author_scores.items(), key=lambda x: x[1], reverse=True)[:k]
        
        print("\nTop-10 Reviewers (LDA):")
        print(f"{'Rank':<6} {'Author':<40} {'Score':<12}")
        print("-" * 80)
        for i, (author, score) in enumerate(top_k, 1):
            print(f"{i:<6} {author:<40} {score:.6f}")
        
        return top_k
    
    except Exception as e:
        print(f"ERROR in LDA: {e}")
        return []


def get_pretrained_embedding_recommendations(paper_text, 
                                            cache_path='pretrained_embedding/embeddings_cache.pt', 
                                            k=10):
    """Get top-k recommendations using pretrained embeddings"""
    print("\n" + "="*80)
    print("PRETRAINED EMBEDDING METHOD (Sentence Transformers)")
    print("="*80)
    
    try:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {device}")
        
        # Load model and cache
        model = SentenceTransformer('all-mpnet-base-v2', device=device)
        cache = torch.load(cache_path)
        paper_embeddings = cache["embeddings"].to(device)
        authors = cache["authors"]
        
        # Process query
        cleaned_input = clean_text(paper_text)
        query_embedding = model.encode(cleaned_input, convert_to_tensor=True, device=device)
        
        # Compute similarities
        sim_scores = util.cos_sim(query_embedding, paper_embeddings)[0]
        
        # Aggregate by author (max score)
        author_scores = defaultdict(list)
        for score, author in zip(sim_scores, authors):
            author_scores[author].append(float(score))
        
        author_max_scores = {author: max(scores) for author, scores in author_scores.items()}
        
        # Get top-k
        top_k = sorted(author_max_scores.items(), key=lambda x: x[1], reverse=True)[:k]
        
        print("\nTop-10 Reviewers (Pretrained Embedding):")
        print(f"{'Rank':<6} {'Author':<40} {'Score':<12} {'Papers'}")
        print("-" * 80)
        for i, (author, score) in enumerate(top_k, 1):
            num_papers = len(author_scores[author])
            marker = " ★" if score >= 0.999 else ""
            print(f"{i:<6} {author:<40} {score:.6f}     {num_papers}{marker}")
        
        if any(score >= 0.999 for _, score in top_k):
            print("\n★ = Perfect match (paper likely in dataset)")
        
        return top_k
    
    except Exception as e:
        print(f"ERROR in Pretrained Embedding: {e}")
        return []


def get_tfidf_recommendations(paper_text, tfidf_model_path='tf-idf/tfidf_model.pkl', k=10):
    """Get top-k recommendations using TF-IDF"""
    print("\n" + "="*80)
    print("TF-IDF METHOD (Cosine Similarity)")
    print("="*80)
    
    try:
        # Load TF-IDF model
        with open(tfidf_model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        vectorizer = model_data['vectorizer']
        tfidf_matrix = model_data['tfidf_matrix']
        authors = model_data['authors']
        
        # Process query
        cleaned_input = clean_text(paper_text)
        query_vector = vectorizer.transform([cleaned_input])
        
        # Compute similarities
        similarities = cosine_similarity(query_vector, tfidf_matrix)[0]
        
        # Aggregate by author (max score)
        author_scores = defaultdict(list)
        for idx, author in enumerate(authors):
            author_scores[author].append(similarities[idx])
        
        author_max_scores = {author: max(scores) for author, scores in author_scores.items()}
        
        # Get top-k
        top_k = sorted(author_max_scores.items(), key=lambda x: x[1], reverse=True)[:k]
        
        print("\nTop-10 Reviewers (TF-IDF):")
        print(f"{'Rank':<6} {'Author':<40} {'Score':<12} {'Papers'}")
        print("-" * 80)
        for i, (author, score) in enumerate(top_k, 1):
            num_papers = len(author_scores[author])
            marker = " ★" if score >= 0.999 else ""
            print(f"{i:<6} {author:<40} {score:.6f}     {num_papers}{marker}")
        
        if any(score >= 0.999 for _, score in top_k):
            print("\n★ = Perfect match (paper likely in dataset)")
        
        return top_k
    
    except Exception as e:
        print(f"ERROR in TF-IDF: {e}")
        return []


def create_comparison_table(lda_results, embedding_results, tfidf_results):
    """Create a comparison table of all three methods"""
    print("\n" + "="*80)
    print("COMPARISON TABLE - TOP 10 REVIEWERS FROM EACH METHOD")
    print("="*80)
    
    # Create header
    print(f"\n{'Rank':<6} {'LDA':<35} {'Pretrained Embedding':<35} {'TF-IDF':<35}")
    print("-" * 115)
    
    # Print results side by side
    max_len = max(len(lda_results), len(embedding_results), len(tfidf_results))
    for i in range(max_len):
        rank = i + 1
        
        lda_name = lda_results[i][0] if i < len(lda_results) else ""
        lda_score = f"({lda_results[i][1]:.4f})" if i < len(lda_results) else ""
        lda_str = f"{lda_name[:25]} {lda_score}" if lda_name else ""
        
        emb_name = embedding_results[i][0] if i < len(embedding_results) else ""
        emb_score = f"({embedding_results[i][1]:.4f})" if i < len(embedding_results) else ""
        emb_str = f"{emb_name[:25]} {emb_score}" if emb_name else ""
        
        tfidf_name = tfidf_results[i][0] if i < len(tfidf_results) else ""
        tfidf_score = f"({tfidf_results[i][1]:.4f})" if i < len(tfidf_results) else ""
        tfidf_str = f"{tfidf_name[:25]} {tfidf_score}" if tfidf_name else ""
        
        print(f"{rank:<6} {lda_str:<35} {emb_str:<35} {tfidf_str:<35}")
    
    print("="*80)


def analyze_consensus(lda_results, embedding_results, tfidf_results):
    """Analyze consensus among the three methods"""
    print("\n" + "="*80)
    print("CONSENSUS ANALYSIS")
    print("="*80)
    
    # Get author sets
    lda_authors = set([author for author, _ in lda_results])
    emb_authors = set([author for author, _ in embedding_results])
    tfidf_authors = set([author for author, _ in tfidf_results])
    
    # Find overlaps
    all_three = lda_authors & emb_authors & tfidf_authors
    lda_emb = (lda_authors & emb_authors) - all_three
    lda_tfidf = (lda_authors & tfidf_authors) - all_three
    emb_tfidf = (emb_authors & tfidf_authors) - all_three
    
    print(f"\nAuthors appearing in ALL THREE methods: {len(all_three)}")
    if all_three:
        for author in sorted(all_three):
            lda_rank = next((i+1 for i, (a, _) in enumerate(lda_results) if a == author), None)
            emb_rank = next((i+1 for i, (a, _) in enumerate(embedding_results) if a == author), None)
            tfidf_rank = next((i+1 for i, (a, _) in enumerate(tfidf_results) if a == author), None)
            print(f"  • {author}")
            print(f"    Ranks: LDA={lda_rank}, Embedding={emb_rank}, TF-IDF={tfidf_rank}")
    
    print(f"\nAuthors in TWO methods:")
    print(f"  LDA & Embedding: {len(lda_emb)}")
    if lda_emb:
        for author in sorted(list(lda_emb)[:3]):
            print(f"    • {author}")
    
    print(f"  LDA & TF-IDF: {len(lda_tfidf)}")
    if lda_tfidf:
        for author in sorted(list(lda_tfidf)[:3]):
            print(f"    • {author}")
    
    print(f"  Embedding & TF-IDF: {len(emb_tfidf)}")
    if emb_tfidf:
        for author in sorted(list(emb_tfidf)[:3]):
            print(f"    • {author}")
    
    # Calculate consensus score
    total_unique = len(lda_authors | emb_authors | tfidf_authors)
    consensus_score = (len(all_three) * 3 + (len(lda_emb) + len(lda_tfidf) + len(emb_tfidf)) * 2) / (total_unique * 3)
    print(f"\nConsensus Score: {consensus_score:.2%}")
    print("(Higher score = more agreement between methods)")
    
    print("="*80)


def main(pdf_path):
    """Main function to run all three methods and compare results"""
    print("\n" + "="*100)
    print("UNIFIED REVIEWER RECOMMENDATION SYSTEM")
    print("="*100)
    print(f"PDF: {pdf_path}")
    print("="*100)
    
    # Extract text from PDF
    print("\nExtracting text from PDF...")
    paper_text = extract_text_from_pdf(pdf_path)
    
    if not paper_text:
        print("Failed to extract text from PDF.")
        return
    
    if len(paper_text.strip()) < 100:
        print("WARNING: Short text extracted. Might be a scanned PDF or corrupted file.")
    
    print(f"Extracted {len(paper_text)} characters from PDF")
    
    # Run all three methods
    lda_results = get_lda_recommendations(paper_text, k=10)
    embedding_results = get_pretrained_embedding_recommendations(paper_text, k=10)
    tfidf_results = get_tfidf_recommendations(paper_text, k=10)
    
    # Compare results
    if lda_results and embedding_results and tfidf_results:
        create_comparison_table(lda_results, embedding_results, tfidf_results)
        analyze_consensus(lda_results, embedding_results, tfidf_results)
    
    print("\n" + "="*100)
    print("ANALYSIS COMPLETE")
    print("="*100 + "\n")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Unified Reviewer Recommendation - Get top 10 reviewers using LDA, Pretrained Embeddings, and TF-IDF',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python unified_reviewer_recommendation.py paper.pdf
  python unified_reviewer_recommendation.py /path/to/paper.pdf
  python unified_reviewer_recommendation.py --pdf paper.pdf
        """
    )
    
    parser.add_argument('pdf', nargs='?', type=str, help='Path to input PDF file')
    parser.add_argument('--pdf', dest='pdf_alt', type=str, help='Alternative way to specify PDF path')
    
    args = parser.parse_args()
    
    # Get PDF path from either argument
    pdf_path = args.pdf or args.pdf_alt
    
    if not pdf_path:
        parser.print_help()
        print("\nERROR: Please provide a PDF file path")
        sys.exit(1)
    
    if not os.path.exists(pdf_path):
        print(f"ERROR: File not found: {pdf_path}")
        sys.exit(1)
    
    main(pdf_path)
