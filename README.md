# RecomReview

Reviewer recommendation for research papers. Given a paper as a PDF, it ranks the authors in a
collection of 630 papers by 71 researchers by how close their past work is to the new paper. The
ranking is done three ways, so the methods can be compared on the same data:

- **TF-IDF**: unigram and bigram TF-IDF (5,000 terms) with cosine similarity.
- **Sentence-BERT**: `all-mpnet-base-v2` document embeddings with cosine similarity.
- **LDA**: 20-topic LDA, comparing the paper's topic mix with each author's average topic mix.

For TF-IDF and SBERT, an author's score is the similarity of their single closest paper. There is
a CLI that runs all three methods and reports where they agree, and a Streamlit app for uploading
a PDF.

## Results

The evaluation is leave-one-out. 200 random papers are drawn from authors with at least two
papers, each paper is removed from the candidate pool, and I check where its real author lands in
the ranking. Hit@K is the fraction of queries where the real author is in the top K.

| Method | Hit@1 | Hit@3 | Hit@5 | Hit@10 | MRR@10 |
|---|---:|---:|---:|---:|---:|
| TF-IDF | **0.650** | **0.735** | **0.760** | 0.815 | **0.700** |
| Sentence-BERT | 0.550 | 0.700 | 0.735 | 0.805 | 0.632 |
| LDA | 0.345 | 0.590 | 0.670 | **0.825** | 0.483 |

With 71 candidate authors, a random guess would get Hit@1 of about 0.014.

TF-IDF does best on this dataset. My read is that papers by the same author share very specific
vocabulary: method names, datasets, and recurring co-authors and affiliations in the extracted
text. Exact term matching rewards that more than a general-purpose sentence embedding does. SBERT
is also at a disadvantage in this setup for two reasons. It is given the cleaned text (stopwords
removed, words lemmatized), which strips the sentence structure the model was trained on. And it
truncates input at 384 tokens, so it only sees the start of each paper. Embedding the raw text in
chunks and pooling the results is the obvious next thing to try.

I also compared how well each representation separates authors, without running retrieval
(`evaluations/evaluate_embeddings.py`):

| Representation | Silhouette (71 clusters) | Same-author similarity | Different-author similarity |
|---|---:|---:|---:|
| LDA topics | 0.370 | 0.546 | 0.126 |
| Sentence-BERT | 0.108 | 0.435 | 0.239 |
| TF-IDF | 0.059 | 0.298 | 0.167 |

LDA forms the tightest author clusters, but that doesn't carry over to ranking: it gets the
real author first in only about a third of queries. Twenty topics are too coarse to tell apart
researchers who work in the same area.

t-SNE and UMAP plots of each representation are in [visualizations/](visualizations/).

### Caveats

- **Small sample.** 200 queries gives roughly ±3 points of sampling noise on each number. The
  sample isn't seeded, so a rerun will differ slightly.
- **Fitted on all papers.** The TF-IDF vocabulary, the IDF weights and the LDA model were fitted
  on all 630 papers, including the held-out ones.
- **LDA leak.** The LDA author profiles are averaged before the query paper is removed, so the
  held-out paper still counts towards its own author. The LDA numbers are therefore somewhat
  optimistic, and they are still the weakest.
- **Easy target.** An author's own papers are an easier target than real reviewing, where the
  right reviewer is someone *other* than the paper's authors.

## How it works

```
Dataset/<author>/*.pdf
  -> Dataset/parse_dataset.py                 extract text from each PDF -> parsed_dataset.json
  -> intermediate_files/process_dataset.py   lowercase, strip numbers and punctuation,
                                             remove stopwords, lemmatize
  -> cleaned_dataset.json                     630 papers, 71 authors
  -> tf-idf/            TF-IDF matrix          -> tfidf_model.pkl
     pretrained_embedding/  SBERT embeddings   -> embeddings_cache.pt
     LDA /              20-topic LDA           -> lda_model.pkl
  -> unified_reviewer_recommendation.py       all three rankings, plus where they agree
     streamlit_app.py                         upload a PDF, view and download the ranking
```

A query PDF goes through the same cleaning, is scored against every paper, and the paper scores
are aggregated per author.

## Running it

```bash
pip install -r requirements.txt
python -c "import nltk; nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('punkt')"
```

The trained models are committed, so recommendations work straight away:

```bash
python unified_reviewer_recommendation.py path/to/paper.pdf   # all three methods
streamlit run streamlit_app.py                                 # web app on localhost:8501
```

To retrain on a different set of papers, put them in `Dataset/<author name>/` and run the
following from the repo root. Each training script reads `cleaned_dataset.json` from its own
folder and saves its model there.

```bash
python Dataset/parse_dataset.py                 # -> parsed_dataset.json
python intermediate_files/process_dataset.py    # -> cleaned_dataset.json
for d in tf-idf pretrained_embedding "LDA "; do cp cleaned_dataset.json "$d/"; done
rm -f pretrained_embedding/embeddings_cache.pt  # otherwise the old embeddings are reused

(cd tf-idf && python tfidf_cosine_similarity.py --train)
(cd pretrained_embedding && python semantic_similarity_gpu.py)
(cd "LDA " && python topic_modeling_lda.py --train --topics 20)
```

Note that the LDA folder name ends with a space, so it needs quotes. To reproduce the
evaluation, run these from the repo root:

```bash
python evaluations/evaluate_precision_recall.py
python evaluations/evaluate_embeddings.py
```

A GPU speeds up the SBERT step but isn't required.

## Repository layout

```
Dataset/                 source PDFs, one folder per author, and the PDF parser
intermediate_files/      text extraction and cleaning
tf-idf/                  TF-IDF model and retrieval
pretrained_embedding/    Sentence-BERT embeddings and retrieval
LDA /                    LDA topic model and retrieval
evaluations/             leave-one-out retrieval eval, clustering / separation eval
visualizations/          result JSONs and plots
unified_reviewer_recommendation.py   CLI comparing all three methods
streamlit_app.py         web app
```
