"""
Streamlit Web App for Reviewer Recommendation System
Upload a PDF and get top 10 reviewer recommendations using Pretrained Embeddings
"""

import streamlit as st
import os
import tempfile
import torch
import PyPDF2
import re
import string
from collections import defaultdict
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
from sentence_transformers import SentenceTransformer, util
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Download NLTK data
@st.cache_resource
def download_nltk_data():
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)

download_nltk_data()

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Page configuration
st.set_page_config(
    page_title="Reviewer Recommendation System",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stAlert {
        margin-top: 1rem;
    }
    h1 {
        color: #1f77b4;
        padding-bottom: 1rem;
    }
    h2 {
        color: #2ca02c;
        padding-top: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    </style>
    """, unsafe_allow_html=True)


@st.cache_resource
def load_model_and_cache():
    """Load the sentence transformer model and embeddings cache"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    cache_path = 'pretrained_embedding/embeddings_cache.pt'
    
    if not os.path.exists(cache_path):
        st.error(f"❌ Embeddings cache not found at: {cache_path}")
        st.error("Please generate embeddings first using semantic_similarity_gpu.py")
        return None, None, None, None
    
    try:
        model = SentenceTransformer('all-mpnet-base-v2', device=device)
        cache = torch.load(cache_path, map_location=device)
        paper_embeddings = cache["embeddings"].to(device)
        authors = cache["authors"]
        
        return model, paper_embeddings, authors, device
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return None, None, None, None


def extract_text_from_pdf(pdf_file):
    """Extract text content from uploaded PDF file"""
    try:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
        return text
    except Exception as e:
        st.error(f"❌ Error extracting text from PDF: {e}")
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


def find_top_reviewers(input_text, model, paper_embeddings, authors, device, k=10):
    """Find top-k reviewers using pretrained embeddings"""
    cleaned_input = clean_text(input_text)
    
    if not cleaned_input:
        return [], {}
    
    # Encode query
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
    
    return top_k, author_scores


def create_results_dataframe(top_k, author_scores):
    """Create a pandas DataFrame from results"""
    data = []
    for rank, (author, score) in enumerate(top_k, 1):
        num_papers = len(author_scores[author])
        avg_score = sum(author_scores[author]) / len(author_scores[author])
        data.append({
            'Rank': rank,
            'Author': author,
            'Max Score': f"{score:.6f}",
            'Papers': num_papers,
            'Avg Score': f"{avg_score:.6f}",
            'Score Value': score  # For plotting
        })
    
    return pd.DataFrame(data)


def create_score_chart(df):
    """Create a bar chart of reviewer scores"""
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=df['Author'],
        y=df['Score Value'],
        text=df['Max Score'],
        textposition='auto',
        marker=dict(
            color=df['Score Value'],
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title="Score")
        ),
        hovertemplate='<b>%{x}</b><br>Score: %{y:.6f}<br><extra></extra>'
    ))
    
    fig.update_layout(
        title="Top 10 Reviewers by Similarity Score",
        xaxis_title="Reviewer",
        yaxis_title="Similarity Score",
        xaxis_tickangle=-45,
        height=500,
        showlegend=False,
        hovermode='x'
    )
    
    return fig


def main():
    # Header
    st.title("📄 Reviewer Recommendation System")
    st.markdown("### Upload a research paper PDF to get AI-powered reviewer recommendations")
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.header("ℹ️ About")
        st.markdown("""
        This system uses **Pretrained Embeddings** (Sentence Transformers) to find the most suitable reviewers for your research paper.
        
        **How it works:**
        1. Upload your research paper (PDF)
        2. The system extracts and analyzes the content
        3. It compares your paper with a database of researchers
        4. Returns the top 10 most relevant reviewers
        
        **Model:** `all-mpnet-base-v2`
        """)
        
        st.markdown("---")
        
        # Load model and show status
        with st.spinner("Loading AI model..."):
            model, paper_embeddings, authors, device = load_model_and_cache()
        
        if model is not None:
            st.success(f"✅ Model loaded successfully!")
            st.info(f"🖥️ Using: **{device.upper()}**")
            st.metric("Database Size", f"{len(set(authors))} authors")
            st.metric("Total Papers", len(authors))
        else:
            st.error("❌ Failed to load model")
            st.stop()
        
        st.markdown("---")
        
        # Settings
        st.header("⚙️ Settings")
        num_recommendations = st.slider(
            "Number of recommendations",
            min_value=5,
            max_value=20,
            value=10,
            step=1
        )
        
        show_details = st.checkbox("Show detailed statistics", value=True)
        show_chart = st.checkbox("Show visualization", value=True)
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("📤 Upload Your Research Paper")
        uploaded_file = st.file_uploader(
            "Choose a PDF file",
            type=['pdf'],
            help="Upload a research paper in PDF format"
        )
    
    with col2:
        if uploaded_file is not None:
            st.success("✅ File uploaded successfully!")
            st.info(f"**Filename:** {uploaded_file.name}")
            st.info(f"**Size:** {uploaded_file.size / 1024:.2f} KB")
    
    # Process uploaded file
    if uploaded_file is not None:
        st.markdown("---")
        
        # Extract text
        with st.spinner("📖 Extracting text from PDF..."):
            paper_text = extract_text_from_pdf(uploaded_file)
        
        if not paper_text:
            st.error("❌ Failed to extract text from PDF. Please check the file.")
            st.stop()
        
        # Show text statistics
        with st.expander("📊 Text Extraction Summary", expanded=False):
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Characters", len(paper_text))
            with col2:
                st.metric("Words", len(paper_text.split()))
            with col3:
                st.metric("Lines", paper_text.count('\n'))
            
            if len(paper_text.strip()) < 100:
                st.warning("⚠️ Short text extracted. The PDF might be scanned or corrupted.")
        
        # Find reviewers
        with st.spinner("🔍 Finding best reviewers..."):
            top_k, author_scores = find_top_reviewers(
                paper_text, 
                model, 
                paper_embeddings, 
                authors, 
                device, 
                k=num_recommendations
            )
        
        if not top_k:
            st.error("❌ No reviewers found. Please check your PDF content.")
            st.stop()
        
        # Display results
        st.markdown("---")
        st.header("🏆 Top Recommended Reviewers")
        
        # Create DataFrame
        df = create_results_dataframe(top_k, author_scores)
        
        # Check for perfect matches
        perfect_matches = [(author, score) for author, score in top_k if score >= 0.999]
        if perfect_matches:
            st.warning(f"⚠️ {len(perfect_matches)} reviewer(s) have perfect scores (paper might be in the database)")
        
        # Display table
        st.dataframe(
            df[['Rank', 'Author', 'Max Score', 'Papers', 'Avg Score']],
            use_container_width=True,
            hide_index=True
        )
        
        # Download button
        csv = df[['Rank', 'Author', 'Max Score', 'Papers', 'Avg Score']].to_csv(index=False)
        st.download_button(
            label="📥 Download Results (CSV)",
            data=csv,
            file_name=f"reviewers_{uploaded_file.name.replace('.pdf', '')}.csv",
            mime="text/csv"
        )
        
        # Visualization
        if show_chart:
            st.markdown("---")
            st.header("📊 Visualization")
            fig = create_score_chart(df)
            st.plotly_chart(fig, use_container_width=True)
        
        # Detailed statistics
        if show_details:
            st.markdown("---")
            st.header("📈 Detailed Statistics")
            
            col1, col2, col3, col4 = st.columns(4)
            
            scores = [score for _, score in top_k]
            
            with col1:
                st.metric("Best Score", f"{scores[0]:.6f}")
            with col2:
                st.metric(f"{num_recommendations}th Score", f"{scores[-1]:.6f}")
            with col3:
                st.metric("Score Range", f"{scores[0] - scores[-1]:.6f}")
            with col4:
                st.metric("Average Score", f"{sum(scores)/len(scores):.6f}")
            
            # Non-perfect scores statistics
            non_perfect_scores = [score for score in scores if score < 0.999]
            if non_perfect_scores and len(non_perfect_scores) < len(scores):
                st.markdown("##### Excluding Perfect Matches:")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Best Score", f"{max(non_perfect_scores):.6f}")
                with col2:
                    st.metric("Average Score", f"{sum(non_perfect_scores)/len(non_perfect_scores):.6f}")
            
            # Top-3 details
            st.markdown("---")
            st.subheader("🥇 Top 3 Reviewer Details")
            
            for i, (author, score) in enumerate(top_k[:3], 1):
                with st.expander(f"#{i} - {author} (Score: {score:.6f})"):
                    scores_list = author_scores[author]
                    mean = sum(scores_list) / len(scores_list)
                    stddev = (sum((s - mean)**2 for s in scores_list) / len(scores_list))**0.5
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Papers", len(scores_list))
                    with col2:
                        st.metric("Max Score", f"{max(scores_list):.6f}")
                    with col3:
                        st.metric("Mean Score", f"{mean:.6f}")
                    with col4:
                        st.metric("Std Dev", f"{stddev:.6f}")
    
    else:
        # Show instructions when no file is uploaded
        st.info("👆 Please upload a PDF file to get started")
        
        st.markdown("---")
        st.header("🚀 Quick Start Guide")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **Step 1:** Upload your research paper
            - Click the upload button above
            - Select a PDF file from your computer
            - Wait for the upload to complete
            """)
        
        with col2:
            st.markdown("""
            **Step 2:** Get recommendations
            - The system will automatically analyze your paper
            - View the top recommended reviewers
            - Download results as CSV if needed
            """)


if __name__ == "__main__":
    main()
