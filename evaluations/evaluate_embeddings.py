"""
Comprehensive Evaluation of Embedding Quality
Compares TF-IDF, LDA, and Sentence-BERT embeddings using:
1. Clustering Metrics (Silhouette, Davies-Bouldin, Calinski-Harabasz)
2. Pairwise Semantic Consistency (Intra vs Inter-author similarity)
3. Dimensional Visualization (t-SNE and UMAP)
"""

import json
import pickle
import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.manifold import TSNE
import umap
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import os
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

class EmbeddingEvaluator:
    def __init__(self, dataset_path='cleaned_dataset.json'):
        """Initialize evaluator with dataset"""
        print("Loading dataset...")
        with open(dataset_path, 'r', encoding='utf-8') as f:
            self.dataset = json.load(f)
        
        self.authors = [paper['author_name'] for paper in self.dataset]
        self.unique_authors = list(set(self.authors))
        self.n_papers = len(self.dataset)
        
        print(f"Loaded {self.n_papers} papers from {len(self.unique_authors)} authors")
        
        self.embeddings = {}
        self.author_to_indices = defaultdict(list)
        
        for idx, author in enumerate(self.authors):
            self.author_to_indices[author].append(idx)
    
    def load_sentence_bert_embeddings(self, cache_path='pretrained_embedding/embeddings_cache.pt'):
        """Load Sentence-BERT embeddings"""
        cache = torch.load(cache_path)
        embeddings = cache['embeddings'].cpu().numpy()
        
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / (norms + 1e-8)
        
        self.embeddings['sentence_bert'] = embeddings
        print(f"  Shape: {embeddings.shape}")
        return embeddings
    
    def load_lda_embeddings(self, model_dir='LDA '):
        """Load LDA topic distributions as embeddings"""
        
        model_path = os.path.join(model_dir, 'lda_model.pkl')
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        lda_model = model_data['lda_model']
        vectorizer = model_data['vectorizer']
        
        texts = [paper['text_content'] for paper in self.dataset]
        doc_term_matrix = vectorizer.transform(texts)
        topic_distributions = lda_model.transform(doc_term_matrix)
        
        norms = np.linalg.norm(topic_distributions, axis=1, keepdims=True)
        topic_distributions = topic_distributions / (norms + 1e-8)
        
        self.embeddings['lda'] = topic_distributions
        print(f"  Shape: {topic_distributions.shape}")
        return topic_distributions
    
    def load_tfidf_embeddings(self, model_dir='tf-idf'):
        """Load TF-IDF vectors as embeddings"""
        
        model_path = os.path.join(model_dir, 'tfidf_model.pkl')
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        vectorizer = model_data['vectorizer']
        
        texts = [paper['text_content'] for paper in self.dataset]
        tfidf_matrix = vectorizer.transform(texts).toarray()
        
        norms = np.linalg.norm(tfidf_matrix, axis=1, keepdims=True)
        tfidf_matrix = tfidf_matrix / (norms + 1e-8)
        
        self.embeddings['tfidf'] = tfidf_matrix
        print(f"  Shape: {tfidf_matrix.shape}")
        return tfidf_matrix
    
    def evaluate_clustering(self, embeddings, method_name, n_clusters=None):
        """
        Evaluate clustering quality using multiple metrics
        
        Args:
            embeddings: numpy array of embeddings
            method_name: name of the method
            n_clusters: number of clusters (defaults to number of authors)
        
        Returns:
            dict of clustering metrics
        """
        if n_clusters is None:
            n_clusters = min(len(self.unique_authors), 100)  # Cap at 100 for efficiency
        
        print(f"\nEvaluating clustering for {method_name} (k={n_clusters})...")
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(embeddings)
        
        silhouette = silhouette_score(embeddings, cluster_labels, sample_size=min(10000, len(embeddings)))
        davies_bouldin = davies_bouldin_score(embeddings, cluster_labels)
        calinski_harabasz = calinski_harabasz_score(embeddings, cluster_labels)
        
        metrics = {
            'silhouette_score': silhouette,
            'davies_bouldin_index': davies_bouldin,
            'calinski_harabasz_index': calinski_harabasz,
            'n_clusters': n_clusters
        }
        
        print(f"  Silhouette Score:        {silhouette:.4f}  (higher is better, range: [-1, 1])")
        print(f"  Davies-Bouldin Index:    {davies_bouldin:.4f}  (lower is better)")
        print(f"  Calinski-Harabasz Index: {calinski_harabasz:.2f}  (higher is better)")
        
        return metrics, cluster_labels
    
    def compute_pairwise_consistency(self, embeddings, method_name, n_samples=5000):
        """
        Compute intra-author vs inter-author similarity
        
        Args:
            embeddings: numpy array of embeddings
            method_name: name of the method
            n_samples: number of random pairs to sample
        
        Returns:
            dict of consistency metrics
        """
        print(f"\nComputing pairwise semantic consistency for {method_name}...")
        
        valid_authors = [author for author, indices in self.author_to_indices.items() 
                        if len(indices) >= 2]
        
        print(f"  Authors with 2+ papers: {len(valid_authors)}")
        
        intra_similarities = []
        intra_count = 0
        
        for author in tqdm(valid_authors[:100], desc="  Intra-author pairs"):  # Limit to 100 authors
            indices = self.author_to_indices[author]
            if len(indices) >= 2:
                n_pairs = min(10, len(indices) * (len(indices) - 1) // 2)
                for _ in range(n_pairs):
                    i = np.random.choice(indices)
                    j = np.random.choice([idx for idx in indices if idx != i])
                    
                    sim = np.dot(embeddings[i], embeddings[j])
                    intra_similarities.append(sim)
                    intra_count += 1
                    
                    if intra_count >= n_samples // 2:
                        break
            
            if intra_count >= n_samples // 2:
                break
        
        inter_similarities = []
        inter_count = 0
        
        print(f"  Sampling inter-author pairs...")
        for _ in tqdm(range(n_samples // 2), desc="  Inter-author pairs"):
            author1, author2 = np.random.choice(valid_authors, size=2, replace=False)
            
            i = np.random.choice(self.author_to_indices[author1])
            j = np.random.choice(self.author_to_indices[author2])
            
            sim = np.dot(embeddings[i], embeddings[j])
            inter_similarities.append(sim)
            inter_count += 1
        
        intra_mean = np.mean(intra_similarities)
        intra_std = np.std(intra_similarities)
        inter_mean = np.mean(inter_similarities)
        inter_std = np.std(inter_similarities)
        separation_margin = intra_mean - inter_mean
        
        separation_ratio = separation_margin / (intra_std + inter_std + 1e-8)
        
        metrics = {
            'intra_author_mean': intra_mean,
            'intra_author_std': intra_std,
            'inter_author_mean': inter_mean,
            'inter_author_std': inter_std,
            'separation_margin': separation_margin,
            'separation_ratio': separation_ratio,
            'intra_similarities': intra_similarities,
            'inter_similarities': inter_similarities
        }
        
        print(f"  Intra-author similarity: {intra_mean:.4f} ± {intra_std:.4f}")
        print(f"  Inter-author similarity: {inter_mean:.4f} ± {inter_std:.4f}")
        print(f"  Separation margin:       {separation_margin:.4f} (higher is better)")
        print(f"  Separation ratio:        {separation_ratio:.4f} (higher is better)")
        
        return metrics
    
    def visualize_embeddings(self, embeddings, method_name, technique='tsne', 
                           n_samples=1000, output_dir='visualizations'):
        """
        Visualize embeddings using t-SNE or UMAP
        
        Args:
            embeddings: numpy array of embeddings
            method_name: name of the method
            technique: 'tsne' or 'umap'
            n_samples: number of samples to visualize
            output_dir: directory to save plots
        """
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"\nVisualizing {method_name} using {technique.upper()}...")
        
        if len(embeddings) > n_samples:
            indices = np.random.choice(len(embeddings), n_samples, replace=False)
            embeddings_sample = embeddings[indices]
            authors_sample = [self.authors[i] for i in indices]
        else:
            embeddings_sample = embeddings
            authors_sample = self.authors
            indices = np.arange(len(embeddings))
        
        if technique == 'tsne':
            print(f"  Running t-SNE...")
            reducer = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
        else:  # umap
            print(f"  Running UMAP...")
            reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=15)
        
        embedding_2d = reducer.fit_transform(embeddings_sample)
        
        print(f"  Creating visualization...")
        
        author_counts = defaultdict(int)
        for author in authors_sample:
            author_counts[author] += 1
        
        top_authors = sorted(author_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        top_author_names = [author for author, _ in top_authors]
        
        colors = []
        for author in authors_sample:
            if author in top_author_names:
                colors.append(top_author_names.index(author))
            else:
                colors.append(-1)  # Gray for others
        
        fig, ax = plt.subplots(figsize=(14, 10))
        
        mask = np.array(colors) == -1
        if np.any(mask):
            ax.scatter(embedding_2d[mask, 0], embedding_2d[mask, 1], 
                      c='lightgray', alpha=0.3, s=20, label='Other authors')
        
        cmap = plt.cm.get_cmap('tab10')
        for i, author in enumerate(top_author_names):
            mask = np.array(colors) == i
            if np.any(mask):
                ax.scatter(embedding_2d[mask, 0], embedding_2d[mask, 1], 
                          c=[cmap(i)], alpha=0.7, s=50, label=author)
        
        ax.set_xlabel(f'{technique.upper()} Dimension 1', fontsize=12)
        ax.set_ylabel(f'{technique.upper()} Dimension 2', fontsize=12)
        ax.set_title(f'{method_name} - {technique.upper()} Visualization\n(Top 10 Authors by Paper Count)', 
                    fontsize=14, fontweight='bold')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        filename = f"{method_name.lower().replace(' ', '_')}_{technique}.png"
        filepath = os.path.join(output_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"  Saved to: {filepath}")
        
        plt.close()
        
        return embedding_2d
    
    def plot_similarity_distributions(self, metrics_dict, output_dir='visualizations'):
        """Plot intra vs inter-author similarity distributions"""
        os.makedirs(output_dir, exist_ok=True)
        
        print("\nCreating similarity distribution plots...")
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        for idx, (method_name, metrics) in enumerate(metrics_dict.items()):
            ax = axes[idx]
            
            ax.hist(metrics['intra_similarities'], bins=50, alpha=0.6, 
                   label='Intra-author', color='blue', density=True)
            ax.hist(metrics['inter_similarities'], bins=50, alpha=0.6, 
                   label='Inter-author', color='red', density=True)
            
            ax.axvline(metrics['intra_author_mean'], color='blue', 
                      linestyle='--', linewidth=2, label=f"Intra mean: {metrics['intra_author_mean']:.3f}")
            ax.axvline(metrics['inter_author_mean'], color='red', 
                      linestyle='--', linewidth=2, label=f"Inter mean: {metrics['inter_author_mean']:.3f}")
            
            ax.set_xlabel('Cosine Similarity', fontsize=11)
            ax.set_ylabel('Density', fontsize=11)
            ax.set_title(f'{method_name}\nSeparation: {metrics["separation_margin"]:.4f}', 
                        fontsize=12, fontweight='bold')
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        filepath = os.path.join(output_dir, 'similarity_distributions.png')
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"  Saved to: {filepath}")
        
        plt.close()
    
    def create_comparison_table(self, all_results, output_dir='visualizations'):
        """Create comparison table of all metrics"""
        os.makedirs(output_dir, exist_ok=True)
        
        print("\nCreating comparison table...")
        
        table_data = []
        for method_name, results in all_results.items():
            row = {
                'Method': method_name,
                'Silhouette↑': f"{results['clustering']['silhouette_score']:.4f}",
                'Davies-Bouldin↓': f"{results['clustering']['davies_bouldin_index']:.4f}",
                'Calinski-Harabasz↑': f"{results['clustering']['calinski_harabasz_index']:.1f}",
                'Intra-Author Sim': f"{results['consistency']['intra_author_mean']:.4f}",
                'Inter-Author Sim': f"{results['consistency']['inter_author_mean']:.4f}",
                'Separation Margin↑': f"{results['consistency']['separation_margin']:.4f}",
                'Separation Ratio↑': f"{results['consistency']['separation_ratio']:.4f}"
            }
            table_data.append(row)
        
        fig, ax = plt.subplots(figsize=(16, 4))
        ax.axis('tight')
        ax.axis('off')
        
        table = ax.table(cellText=[list(row.values()) for row in table_data],
                        colLabels=list(table_data[0].keys()),
                        cellLoc='center',
                        loc='center',
                        colWidths=[0.15, 0.11, 0.14, 0.16, 0.13, 0.13, 0.14, 0.14])
        
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        
        for i in range(len(table_data[0])):
            table[(0, i)].set_facecolor('#4472C4')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        colors = ['#D9E1F2', '#FFFFFF', '#D9E1F2']
        for i in range(1, len(table_data) + 1):
            for j in range(len(table_data[0])):
                table[(i, j)].set_facecolor(colors[i-1])
        
        plt.title('Embedding Quality Comparison\n(↑ higher is better, ↓ lower is better)', 
                 fontsize=14, fontweight='bold', pad=20)
        
        filepath = os.path.join(output_dir, 'comparison_table.png')
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"  Saved to: {filepath}")
        
        plt.close()
        
        print("\n" + "="*120)
        print("EMBEDDING QUALITY COMPARISON")
        print("="*120)
        
        header = list(table_data[0].keys())
        print(f"{header[0]:<20}", end="")
        for h in header[1:]:
            print(f"{h:>15}", end="")
        print()
        print("-"*120)
        
        for row in table_data:
            print(f"{row['Method']:<20}", end="")
            for key in header[1:]:
                print(f"{row[key]:>15}", end="")
            print()
        
        print("="*120)
        print("\nInterpretation:")
        print("  • Silhouette Score: Measures cluster cohesion and separation [-1, 1]")
        print("  • Davies-Bouldin Index: Average similarity ratio of each cluster with most similar one")
        print("  • Calinski-Harabasz: Ratio of between-cluster to within-cluster dispersion")
        print("  • Separation Margin: Difference between intra and inter-author similarities")
        print("  • Separation Ratio: Normalized separation accounting for variance")
        print("="*120)

def main():
    """Main evaluation pipeline"""
    print("="*80)
    print("COMPREHENSIVE EMBEDDING EVALUATION")
    print("="*80)
    
    evaluator = EmbeddingEvaluator('cleaned_dataset.json')
    
    evaluator.load_sentence_bert_embeddings()
    evaluator.load_lda_embeddings()
    evaluator.load_tfidf_embeddings()
    
    all_results = {}
    consistency_metrics = {}
    
    for method_name, embeddings in evaluator.embeddings.items():
        print("\n" + "="*80)
        print(f"EVALUATING: {method_name.upper().replace('_', '-')}")
        print("="*80)
        
        clustering_metrics, cluster_labels = evaluator.evaluate_clustering(
            embeddings, method_name.replace('_', '-').title()
        )
        
        consistency = evaluator.compute_pairwise_consistency(
            embeddings, method_name.replace('_', '-').title()
        )
        consistency_metrics[method_name.replace('_', '-').title()] = consistency
        
        for technique in ['tsne', 'umap']:
            evaluator.visualize_embeddings(
                embeddings, 
                method_name.replace('_', '-').title(),
                technique=technique,
                n_samples=1000
            )
        
        all_results[method_name.replace('_', '-').title()] = {
            'clustering': clustering_metrics,
            'consistency': consistency
        }
    
    print("\n" + "="*80)
    print("CREATING COMPARISON VISUALIZATIONS")
    print("="*80)
    
    evaluator.plot_similarity_distributions(consistency_metrics)
    evaluator.create_comparison_table(all_results)
    
    output_file = 'visualizations/evaluation_results.json'
    with open(output_file, 'w') as f:
        results_serializable = {}
        for method, results in all_results.items():
            results_serializable[method] = {
                'clustering': {k: float(v) if isinstance(v, (np.floating, np.integer)) else v 
                             for k, v in results['clustering'].items()},
                'consistency': {k: float(v) if isinstance(v, (np.floating, np.integer)) else v 
                              for k, v in results['consistency'].items() 
                              if k not in ['intra_similarities', 'inter_similarities']}
            }
        
        json.dump(results_serializable, f, indent=2)
    
    print(f"\nDetailed results saved to: {output_file}")
    
    print("\n" + "="*80)
    print("EVALUATION COMPLETE!")
    print("="*80)
    print("\nAll visualizations saved to: visualizations/")
    print("  • *_tsne.png: t-SNE projections")
    print("  • *_umap.png: UMAP projections")
    print("  • similarity_distributions.png: Intra vs Inter-author similarities")
    print("  • comparison_table.png: Comprehensive metrics comparison")
    print("  • evaluation_results.json: Detailed numerical results")

if __name__ == "__main__":
    main()
