#!/usr/bin/env python3
"""
Simple script to load saved task embeddings and create similarity heatmap
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import AgglomerativeClustering

def create_clustered_summary(similarity_matrix, task_names, n_clusters=10):
    """Create a clustered summary of the similarity matrix for easier visualization"""
    
    # Convert similarity to distance for clustering
    distance_matrix = 1 - similarity_matrix
    
    # Perform hierarchical clustering
    clustering = AgglomerativeClustering(
        n_clusters=n_clusters, 
        metric='precomputed', 
        linkage='average'
    )
    cluster_labels = clustering.fit_predict(distance_matrix)
    
    # Create cluster summary
    cluster_similarities = np.zeros((n_clusters, n_clusters))
    cluster_names = []
    
    for i in range(n_clusters):
        cluster_i_tasks = np.where(cluster_labels == i)[0]
        cluster_names.append(f"Cluster {i+1} ({len(cluster_i_tasks)} tasks)")
        
        for j in range(n_clusters):
            cluster_j_tasks = np.where(cluster_labels == j)[0]
            
            if i == j:
                # Within-cluster similarity
                if len(cluster_i_tasks) > 1:
                    cluster_similarities[i, j] = np.mean(
                        similarity_matrix[np.ix_(cluster_i_tasks, cluster_i_tasks)]
                    )
                else:
                    cluster_similarities[i, j] = 1.0
            else:
                # Between-cluster similarity
                cluster_similarities[i, j] = np.mean(
                    similarity_matrix[np.ix_(cluster_i_tasks, cluster_j_tasks)]
                )
    
    return cluster_similarities, cluster_names, cluster_labels

def load_and_plot_similarity(saved_tokens_path, save_plot_path=None, force_annotations=None, max_labels=None):
    """Load saved task tokens and create similarity heatmap
    
    Args:
        saved_tokens_path: Path to saved task tokens file
        save_plot_path: Optional path to save the plot
        force_annotations: Force showing/hiding annotations (True/False/None for auto)
        max_labels: Maximum number of labels to show on axes (None for auto)
    """
    
    print(f"Loading task tokens from: {saved_tokens_path}")
    data = torch.load(saved_tokens_path, map_location='cpu')
    
    # Get embeddings
    if data['decouple_embeddings']:
        embeddings = data['input_embeddings'].float().numpy()
    else:
        embeddings = data['embeddings'].float().numpy()
    
    task_names = data['task_names']
    num_tasks = len(task_names)
    print(f"Loaded {num_tasks} task embeddings")
    
    # Compute similarity matrix
    similarity_matrix = cosine_similarity(embeddings)
    
    # For very large matrices (>150 tasks), offer clustered summary
    if num_tasks > 150:
        print(f"\nLarge number of tasks ({num_tasks}). Creating clustered summary view...")
        n_clusters = min(20, max(10, num_tasks // 10))  # Adaptive number of clusters
        cluster_sim, cluster_names, cluster_labels = create_clustered_summary(
            similarity_matrix, task_names, n_clusters
        )
        
        # Plot clustered summary
        plt.figure(figsize=(14, 12))
        sns.heatmap(cluster_sim,
                   xticklabels=cluster_names,
                   yticklabels=cluster_names,
                   annot=True,
                   fmt='.3f',
                   cmap='viridis',
                   cbar_kws={'label': 'Average Cosine Similarity'})
        
        plt.title(f'Task Clusters Similarity Summary ({num_tasks} tasks → {n_clusters} clusters)')
        plt.xlabel('Task Clusters')
        plt.ylabel('Task Clusters')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        if save_plot_path:
            cluster_path = save_plot_path.replace('.png', '_clustered.png')
            plt.savefig(cluster_path, dpi=300, bbox_inches='tight')
            print(f"Clustered summary saved to: {cluster_path}")
        
        plt.show()
        
        # Print cluster information
        print(f"\nCluster assignments:")
        for i in range(n_clusters):
            cluster_tasks = [task_names[j] for j in range(num_tasks) if cluster_labels[j] == i]
            print(f"Cluster {i+1}: {cluster_tasks[:5]}{'...' if len(cluster_tasks) > 5 else ''}")
        
        print(f"\nContinuing with full similarity matrix visualization...")
    
    # Adaptive figure sizing and settings based on number of tasks
    if num_tasks <= 20:
        # Small number of tasks: show everything clearly
        figsize = (12, 10)
        show_annot = True
        annot_fontsize = 8
        tick_fontsize = 10
        show_all_labels = True
        fmt = '.3f'
    elif num_tasks <= 50:
        # Medium number of tasks: reduce annotations
        figsize = (16, 14)
        show_annot = True
        annot_fontsize = 6
        tick_fontsize = 8
        show_all_labels = True
        fmt = '.2f'
    elif num_tasks <= 100:
        # Large number of tasks: minimal annotations
        figsize = (20, 18)
        show_annot = False
        annot_fontsize = 4
        tick_fontsize = 6
        show_all_labels = True
        fmt = '.1f'
    else:
        # Very large number of tasks: no annotations, selective labels
        base_size = max(24, num_tasks * 0.2)
        figsize = (base_size, base_size)
        show_annot = False
        annot_fontsize = 3
        tick_fontsize = 4
        show_all_labels = False
        fmt = '.1f'
    
    # Override with user preferences if provided
    if force_annotations is not None:
        show_annot = force_annotations
    
    # Determine label display strategy
    if max_labels is not None:
        show_all_labels = num_tasks <= max_labels
    
    # Create heatmap
    plt.figure(figsize=figsize)
    
    # Prepare labels
    if show_all_labels:
        x_labels = task_names
        y_labels = task_names
    else:
        # Show every nth label to reduce crowding
        target_labels = max_labels if max_labels is not None else 20
        step = max(1, num_tasks // target_labels)
        x_labels = [name if i % step == 0 else '' for i, name in enumerate(task_names)]
        y_labels = [name if i % step == 0 else '' for i, name in enumerate(task_names)]
    
    sns.heatmap(similarity_matrix, 
               xticklabels=x_labels,
               yticklabels=y_labels,
               annot=show_annot,
               fmt=fmt,
               cmap='viridis',
               cbar_kws={'label': 'Cosine Similarity'},
               annot_kws={'size': annot_fontsize} if show_annot else {})
    
    plt.title(f'Task Token Similarity Matrix ({num_tasks} tasks)', fontsize=14)
    plt.xlabel('Tasks', fontsize=12)
    plt.ylabel('Tasks', fontsize=12)
    
    # Adjust tick parameters
    plt.xticks(rotation=45, ha='right', fontsize=tick_fontsize)
    plt.yticks(rotation=0, fontsize=tick_fontsize)
    
    # Additional spacing for large matrices
    if num_tasks > 50:
        plt.subplots_adjust(bottom=0.15, left=0.15)
    else:
        plt.tight_layout()
    
    if save_plot_path:
        plt.savefig(save_plot_path, dpi=300, bbox_inches='tight')
        print(f"Similarity heatmap saved to: {save_plot_path}")
    
    plt.show()
    
    # Print basic stats
    mask = np.eye(len(task_names), dtype=bool)
    off_diagonal = similarity_matrix[~mask]
    print(f"\nSimilarity stats:")
    print(f"Mean: {off_diagonal.mean():.3f}")
    print(f"Std: {off_diagonal.std():.3f}")
    print(f"Min: {off_diagonal.min():.3f}")
    print(f"Max: {off_diagonal.max():.3f}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python analyze_task_similarity.py <saved_tokens_path> [output_plot_path]")
        print("Example: python analyze_task_similarity.py saved_models/task_tokens_20240101.pt similarity.png")
        sys.exit(1)
    
    saved_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else None
    
    load_and_plot_similarity(saved_path, output_path)