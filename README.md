# 📄 RecomReview - AI-Powered Reviewer Recommendation System

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red.svg)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-FF4B4B.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An advanced machine learning system for recommending suitable reviewers for research papers using multiple state-of-the-art NLP techniques. Upload a research paper PDF and get intelligent reviewer recommendations based on semantic similarity, topic modeling, and text analysis.

## 🌟 Features

### Three Complementary Recommendation Methods

1. **🧠 Pretrained Embeddings (Sentence Transformers)**
   - Uses `all-mpnet-base-v2` model for semantic understanding
   - GPU-accelerated inference for fast recommendations
   - Captures deep semantic relationships between papers
   - Best for finding reviewers with similar research interests

2. **📊 LDA Topic Modeling**
   - Latent Dirichlet Allocation for topic discovery
   - Identifies hidden thematic structures in papers
   - Configurable number of topics (default: 20)
   - Excellent for finding reviewers within specific research domains

3. **📝 TF-IDF Cosine Similarity**
   - Traditional keyword-based matching
   - Fast and interpretable baseline method
   - Unigram and bigram support
   - Effective for explicit keyword overlap

### Interactive Web Interface

- **🎨 Streamlit Web App** - User-friendly interface for uploading PDFs
- **📊 Interactive Visualizations** - Charts and graphs using Plotly
- **📥 Export Results** - Download recommendations as CSV
- **⚙️ Customizable Settings** - Adjust number of recommendations and display options
- **📈 Detailed Statistics** - Score distributions, consensus analysis, and more

### Advanced Analysis Features

- **🤝 Consensus Analysis** - Identifies reviewers recommended by multiple methods
- **📊 Score Comparison** - Side-by-side comparison of all three methods
- **🔍 Perfect Match Detection** - Flags papers already in the database
- **📈 Statistical Summaries** - Mean, standard deviation, and score ranges

---

## 📋 Table of Contents

- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Project Structure](#-project-structure)
- [Dataset Setup](#-dataset-setup)
- [Training Models](#-training-models)
- [Usage](#-usage)
  - [Web Interface](#web-interface-streamlit-app)
  - [Command Line](#command-line-unified-system)
  - [Individual Methods](#individual-methods)
- [System Architecture](#-system-architecture)
- [Performance](#-performance)
- [Configuration](#-configuration)
- [Troubleshooting](#-troubleshooting)
- [Contributing](#-contributing)
- [License](#-license)

---

## 🚀 Installation

### Prerequisites

- **Python 3.8 or higher**
- **CUDA-capable GPU** (optional, but recommended for faster processing)
- **At least 8GB RAM** (16GB recommended)
- **5GB free disk space** (for models and embeddings)

### Step 1: Clone the Repository

```bash
git clone https://github.com/mahipeddu/RecomReview.git
cd RecomReview
```

### Step 2: Create Virtual Environment (Recommended)

```bash
# Using venv
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Or using conda
conda create -n recomreview python=3.10
conda activate recomreview
```

### Step 3: Install Dependencies

```bash
# Install core dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118  # For CUDA 11.8
# Or for CPU only:
# pip install torch torchvision torchaudio

# Install other required packages
pip install sentence-transformers==2.2.2
pip install scikit-learn==1.3.0
pip install nltk==3.8.1
pip install PyPDF2==3.0.1
pip install streamlit==1.28.0
pip install plotly==5.17.0
pip install pandas==2.0.3
pip install numpy==1.24.3
pip install tqdm==4.66.1
```

### Step 4: Download NLTK Data

```bash
python3 -c "import nltk; nltk.download('stopwords'); nltk.download('wordnet')"
```

### Verify Installation

```bash
# Check Python version
python3 --version

# Check if GPU is available (optional)
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# Test imports
python3 -c "import streamlit, sentence_transformers, sklearn, nltk, PyPDF2; print('All packages imported successfully!')"
```

---

## ⚡ Quick Start

### Option 1: Web Interface (Easiest)

```bash
# Start the Streamlit web app
bash run_streamlit.sh

# Or directly:
streamlit run streamlit_app.py
```

Then open your browser to `http://localhost:8501` and upload a PDF!

### Option 2: Command Line

```bash
# Get recommendations from all three methods
python3 unified_reviewer_recommendation.py path/to/paper.pdf
```

**That's it!** You'll get top 10 reviewer recommendations with consensus analysis.

---

## 📁 Project Structure

```
RecomReview/
├── README.md                          # This file
├── streamlit_app.py                   # Web interface application
├── unified_reviewer_recommendation.py # CLI tool for all three methods
├── run_streamlit.sh                   # Quick start script for web app
│
├── Dataset/                           # Raw PDF papers organized by author
│   ├── parse_dataset.py              # Script to parse PDFs into JSON
│   ├── Amit Saxena/                  # Author folder (example)
│   │   ├── paper1.pdf
│   │   └── paper2.pdf
│   └── [Other author folders]/
│
├── LDA/                               # LDA Topic Modeling
│   ├── topic_modeling_lda.py         # LDA training and inference
│   ├── topic_modeling_reviewers.py   # Author-specific topic analysis
│   ├── lda_model.pkl                 # Trained LDA model (generated)
│   └── lda_model_author_topics.pkl   # Author topic distributions (generated)
│
├── pretrained_embedding/              # Sentence Transformer embeddings
│   ├── semantic_similarity_gpu.py    # GPU-accelerated embedding generation
│   ├── find_top_reviewers.py         # Inference script
│   ├── embeddings_cache.pt           # Cached embeddings (generated)
│   └── cleaned_dataset.json          # Preprocessed dataset
│
├── tf-idf/                            # TF-IDF method
│   ├── tfidf_cosine_similarity.py    # TF-IDF training and inference
│   └── tfidf_model.pkl               # Trained TF-IDF model (generated)
│
├── intermediate_files/                # Data preprocessing
│   └── process_dataset.py            # Dataset cleaning and preparation
│
├── evaluations/                       # Evaluation scripts
│   ├── evaluate_embeddings.py        # Embedding quality evaluation
│   └── evaluate_precision_recall.py  # Recommendation accuracy metrics
│
├── visualizations/                    # Output visualizations and plots
│
└── __pycache__/                       # Python cache files
```

### Key Files Explained

| File | Purpose |
|------|---------|
| `streamlit_app.py` | Interactive web application with upload, visualization, and export |
| `unified_reviewer_recommendation.py` | Command-line tool combining all three methods |
| `cleaned_dataset.json` | Preprocessed dataset ready for model training |
| `run_streamlit.sh` | One-command launcher for the web interface |

---

## 📚 Dataset Setup

### Preparing Your Dataset

1. **Organize PDFs by Author**

   Place each author's papers in their own folder:

   ```
   Dataset/
   ├── John Doe/
   │   ├── paper1.pdf
   │   ├── paper2.pdf
   │   └── paper3.pdf
   ├── Jane Smith/
   │   └── paper1.pdf
   └── Bob Johnson/
       ├── paper1.pdf
       └── paper2.pdf
   ```

2. **Parse PDFs to JSON**

   ```bash
   cd Dataset
   python3 parse_dataset.py
   ```

   This creates `parsed_dataset.json` with extracted text.

3. **Clean and Preprocess**

   ```bash
   cd ../intermediate_files
   python3 process_dataset.py
   ```

   This creates `cleaned_dataset.json` with:
   - Lowercased text
   - Removed punctuation and numbers
   - Stopword filtering
   - Lemmatization

### Dataset Statistics

The current dataset includes:
- **~80 authors** (reviewers)
- **~500+ research papers**
- Papers from conferences and journals in Computer Science domains

---

## 🎓 Training Models

Before using the system, you need to train/generate the models. **This is a one-time process.**

### 1️⃣ Generate Pretrained Embeddings (Required)

```bash
cd pretrained_embedding
python3 semantic_similarity_gpu.py
```

**What it does:**
- Loads `cleaned_dataset.json`
- Encodes all papers using `all-mpnet-base-v2`
- Saves embeddings to `embeddings_cache.pt` (~500MB)
- **Time:** 2-5 minutes with GPU, 10-20 minutes with CPU

### 2️⃣ Train LDA Model (Required)

```bash
cd "LDA "
python3 topic_modeling_lda.py --train --topics 20
```

**What it does:**
- Builds topic model from papers
- Discovers 20 hidden topics
- Saves `lda_model.pkl` and `lda_model_author_topics.pkl`
- **Time:** 3-10 minutes

**Options:**
```bash
--topics N       # Number of topics (default: 20)
--model PATH     # Custom model save path
```

### 3️⃣ Train TF-IDF Model (Required)

```bash
cd tf-idf
python3 tfidf_cosine_similarity.py --train
```

**What it does:**
- Builds TF-IDF matrix from papers
- Saves `tfidf_model.pkl`
- **Time:** 1-2 minutes

**Options:**
```bash
--max-features N    # Maximum vocabulary size (default: 5000)
--ngram-range X Y   # N-gram range (default: 1 2)
```

### ✅ Verify Models Are Ready

```bash
# Check if all model files exist
ls "LDA "/lda_model.pkl
ls pretrained_embedding/embeddings_cache.pt
ls tf-idf/tfidf_model.pkl
```

All three files should be present before using the system.

---

## 💻 Usage

### Web Interface (Streamlit App)

**Recommended for most users!**

#### Start the App

```bash
bash run_streamlit.sh
```

Or:

```bash
streamlit run streamlit_app.py
```

#### Using the Web Interface

1. **Upload PDF** - Click "Browse files" and select a research paper
2. **View Results** - See top 10 recommended reviewers instantly
3. **Analyze** - Check similarity scores, visualizations, and statistics
4. **Export** - Download results as CSV for record-keeping
5. **Customize** - Adjust settings in the sidebar:
   - Number of recommendations (5-20)
   - Show/hide detailed statistics
   - Toggle visualizations

#### Features in Web App

- 📊 **Interactive Charts** - Bar plots of similarity scores
- 📈 **Statistics Dashboard** - Best score, averages, standard deviations
- 🏆 **Top-3 Deep Dive** - Detailed analysis of top reviewers
- ⚠️ **Perfect Match Detection** - Flags if paper exists in database
- 📥 **CSV Export** - Download results for external use

---

### Command Line (Unified System)

**Get recommendations from all three methods simultaneously:**

```bash
python3 unified_reviewer_recommendation.py paper.pdf
```

#### Output Includes:

1. **LDA Results** - Topic-based recommendations
2. **Pretrained Embedding Results** - Semantic similarity recommendations
3. **TF-IDF Results** - Keyword-based recommendations
4. **Comparison Table** - Side-by-side view of all methods
5. **Consensus Analysis** - Reviewers appearing in multiple methods

#### Example Output:

```
================================================================================
UNIFIED REVIEWER RECOMMENDATION SYSTEM
================================================================================
PDF: example_paper.pdf
================================================================================

================================================================================
LDA METHOD
================================================================================
Top-10 Reviewers (LDA):
Rank   Author                                   Score       
--------------------------------------------------------------------------------
1      Dr. John Smith                           0.856234
2      Prof. Jane Doe                           0.823451
...

================================================================================
PRETRAINED EMBEDDING METHOD (Sentence Transformers)
================================================================================
Using device: cuda
Top-10 Reviewers (Pretrained Embedding):
Rank   Author                                   Score        Papers
--------------------------------------------------------------------------------
1      Dr. Alice Brown                          0.923451     5
2      Dr. Bob Johnson                          0.891234     3
...

================================================================================
TF-IDF METHOD (Cosine Similarity)
================================================================================
Top-10 Reviewers (TF-IDF):
Rank   Author                                   Score        Papers
--------------------------------------------------------------------------------
1      Prof. Carol White                        0.867890     4
...

================================================================================
COMPARISON TABLE
================================================================================
Rank   LDA                    Pretrained Embedding       TF-IDF
--------------------------------------------------------------------------------
1      Dr. John Smith (0.86)  Dr. Alice Brown (0.92)     Prof. Carol White (0.87)
...

================================================================================
CONSENSUS ANALYSIS
================================================================================
Authors appearing in ALL THREE methods: 3
  • Dr. John Smith
    Ranks: LDA=1, Embedding=3, TF-IDF=2
...

Consensus Score: 52.34%
(Higher score = more agreement between methods)
================================================================================
```

---

### Individual Methods

#### Using Only Pretrained Embeddings

```bash
cd pretrained_embedding
python3 find_top_reviewers.py --pdf ../path/to/paper.pdf --k 10
```

**Options:**
- `--pdf PATH` - Path to input PDF
- `--k N` - Number of reviewers (default: 10)
- `--device cuda/cpu` - Force specific device

#### Using Only LDA

```bash
cd "LDA "
python3 topic_modeling_lda.py --pdf ../path/to/paper.pdf --k 10 --method cosine
```

**Options:**
- `--pdf PATH` - Path to input PDF
- `--k N` - Number of reviewers (default: 10)
- `--method METHOD` - Similarity metric: `cosine`, `kl_divergence`, `hellinger`
- `--model PATH` - Path to trained model

#### Using Only TF-IDF

```bash
cd tf-idf
python3 tfidf_cosine_similarity.py --pdf ../path/to/paper.pdf --k 10
```

**Options:**
- `--pdf PATH` - Path to input PDF
- `--k N` - Number of reviewers (default: 10)
- `--model PATH` - Path to trained model

---

## 🏗️ System Architecture

### Data Flow Pipeline

```
┌─────────────────┐
│   Raw PDFs      │
│  (by Author)    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  parse_dataset  │ ──► parsed_dataset.json
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ process_dataset │ ──► cleaned_dataset.json
└────────┬────────┘
         │
         ├────────────────────┬────────────────────┐
         ▼                    ▼                    ▼
    ┌─────────┐         ┌──────────┐        ┌──────────┐
    │   LDA   │         │Embeddings│        │  TF-IDF  │
    │ Training│         │Generation│        │ Training │
    └────┬────┘         └─────┬────┘        └────┬─────┘
         │                    │                   │
         ▼                    ▼                   ▼
    lda_model.pkl    embeddings_cache.pt   tfidf_model.pkl
         │                    │                   │
         └────────────────────┼───────────────────┘
                              │
                              ▼
                    ┌─────────────────┐
                    │  Unified System │
                    │   or Web App    │
                    └────────┬────────┘
                             │
                             ▼
                    Top-K Reviewers + Analysis
```

### Method Comparison

| Feature | Pretrained Embedding | LDA | TF-IDF |
|---------|---------------------|-----|---------|
| **Semantic Understanding** | ⭐⭐⭐⭐⭐ Excellent | ⭐⭐⭐⭐ Good | ⭐⭐ Basic |
| **Speed** | ⭐⭐⭐⭐ Fast (GPU) | ⭐⭐⭐ Medium | ⭐⭐⭐⭐⭐ Very Fast |
| **Interpretability** | ⭐⭐ Low | ⭐⭐⭐⭐ High | ⭐⭐⭐⭐⭐ Very High |
| **Training Time** | ⭐⭐⭐ Medium | ⭐⭐ Slow | ⭐⭐⭐⭐⭐ Very Fast |
| **Memory Usage** | ⭐⭐⭐ High | ⭐⭐⭐⭐ Medium | ⭐⭐⭐⭐⭐ Low |
| **Works Best For** | Semantic similarity | Topic discovery | Keyword matching |

### Technology Stack

- **NLP Models:**
  - Sentence Transformers (`all-mpnet-base-v2`) - 768-dim embeddings
  - Scikit-learn LDA - Topic modeling
  - Scikit-learn TF-IDF - Text vectorization

- **Deep Learning:**
  - PyTorch - Model inference and GPU acceleration
  - CUDA - GPU computing (optional)

- **Web Framework:**
  - Streamlit - Interactive web interface
  - Plotly - Data visualizations

- **Text Processing:**
  - NLTK - Tokenization, stopwords, lemmatization
  - PyPDF2 - PDF text extraction

---

## ⚡ Performance

### Benchmarks (on NVIDIA RTX 3080)

| Operation | Time | Notes |
|-----------|------|-------|
| **Generate Embeddings** | 2-3 min | One-time, 500 papers |
| **Train LDA** | 5-7 min | One-time, 20 topics |
| **Train TF-IDF** | 30-60 sec | One-time |
| **Single Query (Web App)** | 1-2 sec | Using cached embeddings |
| **Unified CLI (All Methods)** | 3-5 sec | All three methods |
| **PDF Text Extraction** | 0.5-2 sec | Depends on PDF size |

### CPU vs GPU Performance

| Method | CPU Time | GPU Time | Speedup |
|--------|----------|----------|---------|
| Embedding Generation | ~15 min | ~3 min | 5x |
| Single Query | ~3 sec | ~0.5 sec | 6x |

### Memory Requirements

- **Training Phase:**
  - LDA: ~2GB RAM
  - Embeddings: ~4GB RAM, ~2GB VRAM (GPU)
  - TF-IDF: ~1GB RAM

- **Inference Phase:**
  - Web App: ~3GB RAM, ~1GB VRAM
  - CLI: ~2GB RAM, ~1GB VRAM

---

## ⚙️ Configuration

### Customizing Number of Topics (LDA)

```bash
cd "LDA "
python3 topic_modeling_lda.py --train --topics 30  # Use 30 topics instead of 20
```

More topics = finer granularity, but may overfit on small datasets.

### Changing the Embedding Model

Edit `pretrained_embedding/semantic_similarity_gpu.py`:

```python
# Change this line (line ~23):
model = SentenceTransformer('all-mpnet-base-v2', device=device)

# To another model, e.g.:
model = SentenceTransformer('all-MiniLM-L6-v2', device=device)  # Faster, smaller
# or
model = SentenceTransformer('all-mpnet-base-v3', device=device)  # Latest version
```

Popular alternatives:
- `all-MiniLM-L6-v2` - Faster, 384-dim (vs 768-dim)
- `multi-qa-mpnet-base-dot-v1` - Optimized for question-answering
- `all-distilroberta-v1` - Balanced speed/quality

### Adjusting TF-IDF Parameters

```bash
cd tf-idf
python3 tfidf_cosine_similarity.py --train --max-features 10000 --ngram-range 1 3
```

Options:
- `--max-features N` - Vocabulary size (default: 5000)
- `--ngram-range X Y` - N-gram range (default: 1 2)

### Web App Port Configuration

```bash
streamlit run streamlit_app.py --server.port 8080  # Use port 8080 instead of 8501
```

---

## 🔧 Troubleshooting

### Issue: "Embeddings cache not found"

**Solution:**
```bash
cd pretrained_embedding
python3 semantic_similarity_gpu.py
```

Wait for it to complete, then restart the web app.

---

### Issue: "CUDA out of memory"

**Solutions:**

1. **Reduce batch size** - Edit the embedding generation script:
   ```python
   # In semantic_similarity_gpu.py, line ~35
   embeddings = model.encode(..., batch_size=16, ...)  # Reduce from 32 to 16
   ```

2. **Use CPU instead:**
   ```bash
   CUDA_VISIBLE_DEVICES="" python3 semantic_similarity_gpu.py
   ```

3. **Clear GPU memory:**
   ```python
   import torch
   torch.cuda.empty_cache()
   ```

---

### Issue: "Short text extracted" warning

**Causes:**
- Scanned PDF (image-based, not text-based)
- Encrypted or protected PDF
- Corrupted file

**Solutions:**
1. Use OCR tools like `pdf2image` + `pytesseract` to extract text from scanned PDFs
2. Try a different PDF reader library (e.g., `pdfplumber`)
3. Convert PDF to text using Adobe Acrobat or online tools first

---

### Issue: "ERROR in LDA/TF-IDF"

**Cause:** Model not trained yet.

**Solution:**
```bash
# Train LDA
cd "LDA "
python3 topic_modeling_lda.py --train

# Train TF-IDF
cd tf-idf
python3 tfidf_cosine_similarity.py --train
```

---

### Issue: Perfect matches (score ≥ 0.999)

**Explanation:** The query paper is likely in your training dataset.

**Solutions:**
- This is expected behavior if testing with papers from your dataset
- For real-world use, ensure your dataset doesn't include the query paper
- Filter out authors from the query paper's institution

---

### Issue: "Module not found" errors

**Solution:**
```bash
# Reinstall dependencies
pip install --upgrade -r requirements.txt

# Or install manually:
pip install torch sentence-transformers scikit-learn nltk PyPDF2 streamlit plotly
```

---

### Issue: Streamlit won't start

**Solutions:**

1. **Check port availability:**
   ```bash
   lsof -i :8501  # Check if port is in use
   streamlit run streamlit_app.py --server.port 8080  # Try different port
   ```

2. **Clear Streamlit cache:**
   ```bash
   streamlit cache clear
   ```

3. **Run with verbose logging:**
   ```bash
   streamlit run streamlit_app.py --logger.level=debug
   ```

---

### Issue: Slow inference time

**Solutions:**

1. **Use GPU:**
   - Install CUDA-enabled PyTorch
   - Verify GPU is detected: `torch.cuda.is_available()`

2. **Reduce dataset size:**
   - Use subset of papers for faster queries
   - Cache embeddings properly

3. **Optimize embeddings:**
   - Use smaller model (e.g., `all-MiniLM-L6-v2`)
   - Reduce embedding dimensions (model-dependent)

---

## 📊 Evaluation

### Running Evaluations

```bash
cd evaluations

# Evaluate embedding quality
python3 evaluate_embeddings.py

# Evaluate precision and recall
python3 evaluate_precision_recall.py
```

### Metrics Computed

- **Precision@K** - Percentage of relevant reviewers in top-K
- **Recall@K** - Percentage of all relevant reviewers found in top-K
- **MAP (Mean Average Precision)** - Average precision across queries
- **NDCG (Normalized Discounted Cumulative Gain)** - Ranking quality metric

---

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

### Reporting Bugs

Open an issue with:
- Description of the bug
- Steps to reproduce
- Expected vs actual behavior
- System info (OS, Python version, GPU)

### Suggesting Features

Open an issue with:
- Feature description
- Use case / motivation
- Proposed implementation (optional)

### Submitting Pull Requests

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes
4. Add tests if applicable
5. Commit: `git commit -m "Add feature X"`
6. Push: `git push origin feature-name`
7. Open a Pull Request

### Code Style

- Follow PEP 8 guidelines
- Add docstrings to functions
- Comment complex logic
- Keep functions focused and modular

---

## 📄 License

This project is licensed under the **MIT License**.

```
MIT License

Copyright (c) 2025 Mahi Peddu

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

## 🙏 Acknowledgments

- **Sentence Transformers** - For the excellent pretrained models
- **Scikit-learn** - For LDA and TF-IDF implementations
- **Streamlit** - For the amazing web framework
- **PyTorch** - For GPU acceleration support
- **All contributors** - For their valuable feedback and contributions

---

## 📧 Contact

- **Author:** Mahi Peddu
- **GitHub:** [@mahipeddu](https://github.com/mahipeddu)
- **Repository:** [RecomReview](https://github.com/mahipeddu/RecomReview)

For questions, feedback, or collaboration opportunities, feel free to open an issue or reach out!

---

## 🎯 Future Enhancements

### Planned Features

- [ ] **Author Metadata Integration** - Include h-index, citations, affiliations
- [ ] **Multi-language Support** - Handle papers in languages other than English
- [ ] **Real-time Collaboration** - Multiple users, shared sessions
- [ ] **Historical Tracking** - Save past queries and recommendations
- [ ] **API Endpoint** - REST API for programmatic access
- [ ] **Docker Support** - Containerized deployment
- [ ] **Batch Processing** - Process multiple papers at once
- [ ] **Explainability** - Show why each reviewer was recommended
- [ ] **Author Disambiguation** - Handle authors with similar names
- [ ] **Conference/Journal Filtering** - Filter reviewers by publication venue

### Research Directions

- [ ] **Deep Learning Ranking** - Learn-to-rank models for better recommendations
- [ ] **Graph Neural Networks** - Model citation networks and co-authorship
- [ ] **Hybrid Methods** - Combine multiple signals intelligently
- [ ] **Active Learning** - Improve with user feedback
- [ ] **Transfer Learning** - Adapt to new domains with few examples

---

## 📚 Citation

If you use this system in your research, please cite:

```bibtex
@software{recomreview2025,
  author = {Peddu, Mahi},
  title = {RecomReview: AI-Powered Reviewer Recommendation System},
  year = {2025},
  url = {https://github.com/mahipeddu/RecomReview},
  version = {1.0.0}
}
```

---

## 📖 Related Work

### Papers

- Devlin et al. (2018) - BERT: Pre-training of Deep Bidirectional Transformers
- Blei et al. (2003) - Latent Dirichlet Allocation
- Reimers & Gurevych (2019) - Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks

### Similar Projects

- **PeerRead** - Dataset and tools for peer review analysis
- **OpenReview** - Open peer review platform
- **Microsoft Academic** - Academic graph and researcher search

---

## ⭐ Star History

If you find this project useful, please consider giving it a star! ⭐

---

<div align="center">

**Made with ❤️ by [Mahi Peddu](https://github.com/mahipeddu)**

[⬆ Back to Top](#-recomreview---ai-powered-reviewer-recommendation-system)

</div>
