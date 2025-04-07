#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
DBSCAN-based deduplication using density-based clustering.
"""

import numpy as np
from typing import Dict, List, Any, Set, Optional, Tuple
from collections import defaultdict

try:
    from sklearn.cluster import DBSCAN
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    import logging
    logging.getLogger(__name__).warning(
        "scikit-learn not installed. Install with: pip install scikit-learn"
    )

from deduplication.dedup_base import DeduplicationMethod


class DBSCANDedup(DeduplicationMethod):
    """
    Deduplication using DBSCAN (Density-Based Spatial Clustering of Applications with Noise).
    Uses TF-IDF vectorization and density-based clustering to identify similar items.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize DBSCAN deduplication with configuration.
        
        Args:
            config: Configuration dictionary
        """
        super().__init__(config)
        
        if not SKLEARN_AVAILABLE:
            raise ImportError(
                "scikit-learn is required for DBSCAN deduplication. "
                "Install with: pip install scikit-learn"
            )
        
        # DBSCAN parameters
        self.eps = config.get("eps", 0.2)  # Maximum distance between two samples to be in the same cluster
        self.min_samples = config.get("min_samples", 2)  # Minimum number of samples in a cluster
        self.ngram_range = config.get("ngram_range", (1, 2))  # n-gram range for TF-IDF
        self.max_features = config.get("max_features", 10000)  # Maximum number of features for TF-IDF
        
        self.logger.info(f"Initialized DBSCAN deduplication with eps={self.eps}, "
                         f"min_samples={self.min_samples}")
    
    def process(self, data: List[Dict[str, Any]], key: str = "conversation") -> List[Dict[str, Any]]:
        """
        Apply DBSCAN deduplication to the input data.
        
        Args:
            data: Input data to deduplicate
            key: The key to use for text extraction from items
            
        Returns:
            Deduplicated data
        """
        if not data:
            return []
            
        self.logger.info(f"Running DBSCAN deduplication on {len(data)} items with key: {key}")
        
        # Step 1: Extract text from each item
        texts = []
        valid_indices = []
        
        for idx, item in enumerate(data):
            text = self._extract_text(item, key)
            if text:
                texts.append(text)
                valid_indices.append(idx)
        
        if not texts:
            return data
            
        # Step 2: Convert texts to TF-IDF vectors
        self.logger.info("Vectorizing texts with TF-IDF")
        vectorizer = TfidfVectorizer(
            ngram_range=self.ngram_range,
            max_features=self.max_features,
            stop_words='english'
        )
        tfidf_matrix = vectorizer.fit_transform(texts)
        
        # Step 3: Apply DBSCAN clustering
        self.logger.info("Running DBSCAN clustering")
        # Calculate pairwise cosine similarities (1 - distance)
        similarity_matrix = cosine_similarity(tfidf_matrix)
        # Convert to distance matrix (1 - similarity)
        distance_matrix = 1 - similarity_matrix
        
        dbscan = DBSCAN(
            eps=self.eps,
            min_samples=self.min_samples,
            metric='precomputed'
        )
        cluster_labels = dbscan.fit_predict(distance_matrix)
        
        # Step 4: Group items by cluster
        clusters = defaultdict(list)
        for i, label in enumerate(cluster_labels):
            if label != -1:  # Skip noise points
                clusters[label].append(valid_indices[i])
                
        # Convert dictionary to list of clusters
        cluster_list = list(clusters.values())
        
        # Step 5: Select representatives from each cluster
        deduplicated = self._select_representatives(cluster_list, data)
        
        n_duplicates = sum(len(cluster) - 1 for cluster in cluster_list)
        self.logger.info(f"Deduplication reduced {len(data)} to {len(deduplicated)} items "
                         f"(removed {n_duplicates} duplicates in {len(cluster_list)} clusters)")
        
        return deduplicated
    
    def _select_representatives(self, clusters: List[List[int]], 
                               data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Select representative items from clusters.
        
        Args:
            clusters: List of clusters (each cluster is a list of item indexes)
            data: Original data items
            
        Returns:
            List of deduplicated items (one representative per cluster)
        """
        result = []
        
        # Add all singletons (items not in any cluster)
        all_items_in_clusters = set()
        for cluster in clusters:
            all_items_in_clusters.update(cluster)
            
        for idx in range(len(data)):
            if idx not in all_items_in_clusters:
                result.append(data[idx])
        
        # Add one representative from each cluster
        for cluster in clusters:
            cluster_items = [data[idx] for idx in cluster]
            representative = self._select_representative(cluster_items)
            result.append(representative)
        
        return result
    
    def _is_similar(self, item1: Dict[str, Any], item2: Dict[str, Any], threshold: Optional[float] = None, key: str = "conversation") -> bool:
        """
        Check if two items are similar based on cosine similarity of TF-IDF vectors.
        
        Args:
            item1: First item
            item2: Second item
            threshold: Optional threshold override
            key: The key to use for text extraction from items
            
        Returns:
            True if items are similar, False otherwise
        """
        text1 = self._extract_text(item1, key)
        text2 = self._extract_text(item2, key)
        
        if not text1 or not text2:
            return False
            
        # Create TF-IDF vectors
        vectorizer = TfidfVectorizer(
            ngram_range=self.ngram_range,
            max_features=self.max_features,
            stop_words='english'
        )
        tfidf_matrix = vectorizer.fit_transform([text1, text2])
        
        # Calculate cosine similarity
        similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        
        threshold = threshold if threshold is not None else (1 - self.eps)
        return similarity >= threshold 