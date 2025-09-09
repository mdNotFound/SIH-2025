#!/usr/bin/env python3
"""
Novel Taxa Discovery Module for Deep-Sea eDNA Analysis
=====================================================

This module implements unsupervised learning algorithms to discover
novel eukaryotic taxa from eDNA sequences without relying on reference
databases. It uses clustering and dimensionality reduction techniques.

Author: SIH 2025 Team
Date: 2025-09-09
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Any
import logging
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

# Dimensionality reduction
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap

# Clustering algorithms
from sklearn.cluster import DBSCAN, KMeans
from hdbscan import HDBSCAN

# Network analysis
import networkx as nx
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances

# Utilities
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.preprocessing import StandardScaler
import joblib

from .sequence_encoder import SequenceEncoder


class NovelTaxaDiscoverer:
    """
    Discovers novel taxa using unsupervised machine learning approaches.
    
    This class implements a pipeline for:
    1. Dimensionality reduction of sequence feature vectors
    2. Clustering to identify potential taxa groups
    3. Network analysis to understand relationships
    4. Novel taxa identification and validation
    """
    
    def __init__(self, config: Dict):
        """
        Initialize the NovelTaxaDiscoverer.
        
        Args:
            config: Configuration dictionary with clustering parameters
        """
        self.config = config
        self.logger = self._setup_logging()
        
        # Algorithm parameters
        self.dim_reduction_method = config['dim_reduction']['method']
        self.clustering_method = config['clustering']['method']
        self.novelty_threshold = config.get('novelty_threshold', 0.7)
        
        # Fitted components
        self.dim_reducer = None
        self.clusterer = None
        self.sequence_encoder = None
        
        # Results
        self.cluster_labels = None
        self.reduced_features = None
        self.cluster_representatives = None
        
    def _setup_logging(self) -> logging.Logger:
        """Set up logging for the discoverer."""
        logger = logging.getLogger('NovelTaxaDiscoverer')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            
        return logger
    
    def discover_taxa(self, sequences: List[str], sequence_ids: List[str]) -> Dict[str, Any]:
        """
        Main pipeline for discovering novel taxa from sequences.
        
        Args:
            sequences: List of DNA sequence strings
            sequence_ids: List of sequence identifiers
            
        Returns:
            Dictionary containing discovery results
        """
        self.logger.info(f"Starting taxa discovery on {len(sequences)} sequences")
        
        # Step 1: Encode sequences to numerical features
        self.logger.info("Step 1: Encoding sequences")
        encoder_config = {
            'encoding_method': self.config.get('encoding_method', 'kmer'),
            'kmer_size': self.config.get('kmer_size', 6),
            'max_features': self.config.get('max_features', 10000)
        }
        
        self.sequence_encoder = SequenceEncoder(encoder_config)
        feature_matrix = self.sequence_encoder.fit_transform(sequences)
        
        # Step 2: Dimensionality reduction
        self.logger.info("Step 2: Dimensionality reduction")
        self.reduced_features = self._perform_dimensionality_reduction(feature_matrix)
        
        # Step 3: Clustering
        self.logger.info("Step 3: Clustering sequences")
        self.cluster_labels = self._perform_clustering(self.reduced_features)
        
        # Step 4: Identify cluster representatives
        self.logger.info("Step 4: Identifying cluster representatives")
        self.cluster_representatives = self._identify_representatives(
            sequences, sequence_ids, self.cluster_labels, self.reduced_features
        )
        
        # Step 5: Build sequence similarity network
        self.logger.info("Step 5: Building similarity network")
        similarity_network = self._build_similarity_network(
            self.reduced_features, sequence_ids, self.cluster_labels
        )
        
        # Step 6: Evaluate clustering quality
        clustering_metrics = self._evaluate_clustering(self.reduced_features, self.cluster_labels)
        
        # Compile results
        results = {
            'cluster_labels': self.cluster_labels,
            'reduced_features': self.reduced_features,
            'cluster_representatives': self.cluster_representatives,
            'similarity_network': similarity_network,
            'clustering_metrics': clustering_metrics,
            'n_clusters': len(set(self.cluster_labels)) - (1 if -1 in self.cluster_labels else 0),
            'n_noise_points': list(self.cluster_labels).count(-1),
            'sequences': sequences,
            'sequence_ids': sequence_ids
        }
        
        self.logger.info(f"Discovery completed: {results['n_clusters']} clusters found")
        return results
