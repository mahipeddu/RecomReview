# Unified Reviewer Recommendation System

This script combines **three different methods** (LDA, Pretrained Embeddings, and TF-IDF) to recommend the top 10 reviewers for a given research paper PDF.

## Features

- **LDA (Latent Dirichlet Allocation)**: Topic modeling-based recommendations
- **Pretrained Embeddings**: Semantic similarity using Sentence Transformers (all-mpnet-base-v2)
- **TF-IDF**: Traditional TF-IDF with cosine similarity
- **Comparison Table**: Side-by-side comparison of all three methods
- **Consensus Analysis**: Shows which reviewers appear in multiple methods

## Requirements

All models must be pre-trained before using this script:

1. **LDA Model**: `LDA /lda_model.pkl` and `LDA /lda_model_author_topics.pkl`
2. **Embeddings Cache**: `pretrained_embedding/embeddings_cache.pt`
3. **TF-IDF Model**: `tf-idf/tfidf_model.pkl`

## Usage

### Basic Usage
```bash
python unified_reviewer_recommendation.py paper.pdf
```

### With full path
```bash
python unified_reviewer_recommendation.py /path/to/your/paper.pdf
```

### Alternative syntax
```bash
python unified_reviewer_recommendation.py --pdf paper.pdf
```

## Output

The script will output:

1. **LDA Results**: Top 10 reviewers based on topic similarity
2. **Pretrained Embedding Results**: Top 10 reviewers based on semantic similarity
3. **TF-IDF Results**: Top 10 reviewers based on keyword similarity
4. **Comparison Table**: All three methods side-by-side
5. **Consensus Analysis**: 
   - Authors appearing in all three methods
   - Authors appearing in two methods
   - Consensus score (agreement level)

## Example Output Structure

```
================================================================================
UNIFIED REVIEWER RECOMMENDATION SYSTEM
================================================================================
PDF: paper.pdf
================================================================================

Extracting text from PDF...
Extracted 25430 characters from PDF

================================================================================
LDA METHOD
================================================================================
Top-10 Reviewers (LDA):
Rank   Author                                   Score       
--------------------------------------------------------------------------------
1      John Doe                                 0.856234
...

================================================================================
PRETRAINED EMBEDDING METHOD (Sentence Transformers)
================================================================================
Using device: cuda
Top-10 Reviewers (Pretrained Embedding):
Rank   Author                                   Score        Papers
--------------------------------------------------------------------------------
1      Jane Smith                               0.923451     5
...

================================================================================
TF-IDF METHOD (Cosine Similarity)
================================================================================
Top-10 Reviewers (TF-IDF):
Rank   Author                                   Score        Papers
--------------------------------------------------------------------------------
1      Bob Johnson                              0.891234     3
...

================================================================================
COMPARISON TABLE - TOP 10 REVIEWERS FROM EACH METHOD
================================================================================

Rank   LDA                                Pretrained Embedding               TF-IDF
-----------------------------------------------------------------------------------------------------------
1      John Doe (0.8562)                  Jane Smith (0.9235)                Bob Johnson (0.8912)
...

================================================================================
CONSENSUS ANALYSIS
================================================================================
Authors appearing in ALL THREE methods: 4
  • Alice Brown
    Ranks: LDA=2, Embedding=3, TF-IDF=1
  ...

Authors in TWO methods:
  LDA & Embedding: 3
  LDA & TF-IDF: 2
  Embedding & TF-IDF: 4

Consensus Score: 45.67%
(Higher score = more agreement between methods)
================================================================================
```

## Notes

- **Perfect Matches (★)**: If a reviewer has a score ≥ 0.999, it means the paper is likely already in the dataset
- **GPU Support**: The pretrained embedding method will automatically use GPU if available
- **Text Extraction**: Works best with text-based PDFs. Scanned PDFs may produce poor results
- **Model Paths**: The script assumes default paths. Modify the paths in the code if your models are elsewhere

## Troubleshooting

### "File not found" error
Make sure the PDF path is correct and the file exists.

### "ERROR in LDA/Embedding/TF-IDF"
Ensure all models are trained and saved in the correct locations:
- Train LDA: `cd "LDA " && python topic_modeling_lda.py --train`
- Generate embeddings: `cd pretrained_embedding && python semantic_similarity_gpu.py`
- Train TF-IDF: `cd tf-idf && python tfidf_cosine_similarity.py --train`

### Short text warning
If you see "Short text extracted", the PDF might be:
- Scanned (image-based) rather than text-based
- Corrupted
- Empty

Try using OCR tools or a different PDF.

## Performance

- **Speed**: ~5-10 seconds per paper (with GPU)
- **Memory**: ~2-4 GB RAM, ~1-2 GB VRAM (GPU)
- **Accuracy**: Consensus across methods provides robust recommendations
