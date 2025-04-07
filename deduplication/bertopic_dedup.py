#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
BERTopic-based deduplication using topic modeling.
"""

import numpy as np
from typing import Dict, List, Any, Set, Optional, Tuple
from collections import defaultdict

try:
    from bertopic import BERTopic
    import umap
    import hdbscan
    from sentence_transformers import SentenceTransformer
    BERTOPIC_AVAILABLE = True
except ImportError:
    BERTOPIC_AVAILABLE = False
    import logging
    logging.getLogger(__name__).warning(
        "BERTopic and its dependencies not installed. Install with: "
        "pip install bertopic sentence-transformers umap-learn hdbscan"
    )

from deduplication.dedup_base import DeduplicationMethod


class BERTopicDedup(DeduplicationMethod):
    """
    Deduplication using BERTopic for topic-based similarity detection.
    Groups conversations into topics and identifies similar items within topics.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize BERTopic deduplication with configuration.
        
        Args:
            config: Configuration dictionary
        """
        super().__init__(config)
        
        if not BERTOPIC_AVAILABLE:
            raise ImportError(
                "BERTopic and its dependencies are required. "
                "Install with: pip install bertopic sentence-transformers umap-learn hdbscan"
            )
        
        # BERTopic parameters
        self.embedding_model_name = config.get("model_name", "all-MiniLM-L6-v2")
        self.min_topic_size = config.get("min_topic_size", 2)
        self.n_neighbors = config.get("n_neighbors", 15)
        self.min_dist = config.get("min_dist", 0.1)
        self.min_cluster_size = config.get("min_cluster_size", 2)
        
        # Initialize embedding model
        self.embedding_model = SentenceTransformer(self.embedding_model_name)
        
        # Initialize UMAP for dimensionality reduction
        self.umap_model = umap.UMAP(
            n_neighbors=self.n_neighbors,
            n_components=5,
            min_dist=self.min_dist,
            metric='cosine',
            random_state=42
        )
        
        # Initialize HDBSCAN for clustering
        self.hdbscan_model = hdbscan.HDBSCAN(
            min_cluster_size=self.min_cluster_size,
            metric='euclidean',
            cluster_selection_method='eom',
            prediction_data=True
        )
        
        # Initialize BERTopic
        self.topic_model = BERTopic(
            embedding_model=self.embedding_model,
            umap_model=self.umap_model,
            hdbscan_model=self.hdbscan_model,
            min_topic_size=self.min_topic_size,
            verbose=True
        )
        
        self.logger.info(f"Initialized BERTopic deduplication with min_topic_size={self.min_topic_size}")
    
    def process(self, data: List[Dict[str, Any]], key: str = "conversation") -> List[Dict[str, Any]]:
        """
        Apply BERTopic deduplication to the input data.
        
        Args:
            data: Input data to deduplicate
            key: The key to use for text extraction from items
            
        Returns:
            Deduplicated data
        """
        if not data:
            return []
            
        self.logger.info(f"Running BERTopic deduplication on {len(data)} items with key: {key}")
        
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
            
        # Skip if too few documents
        if len(texts) < self.min_topic_size:
            self.logger.info(f"Too few documents ({len(texts)}) for topic modeling")
            return data
            
        # Step 2: Apply BERTopic to find topics
        self.logger.info("Training BERTopic model")
        topics, probs = self.topic_model.fit_transform(texts)
        
        # Step 3: Group items by topic
        clusters = defaultdict(list)
        for i, topic_idx in enumerate(topics):
            if topic_idx != -1:  # Skip outlier topic (-1)
                clusters[topic_idx].append(valid_indices[i])
        
        # Convert dictionary to list of clusters
        cluster_list = []
        for topic_idx, item_indices in clusters.items():
            # Only consider topics with multiple items as clusters
            if len(item_indices) > 1:
                cluster_list.append(item_indices)
        
        # Step 4: Select representatives from each cluster
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
    
    def _get_item_embedding(self, item: Dict[str, Any], key: str = "conversation") -> np.ndarray:
        """
        Get embedding for an item.
        
        Args:
            item: Input item
            key: The key to use for text extraction from items
            
        Returns:
            Embedding vector
        """
        text = self._extract_text(item, key)
        if not text:
            return np.zeros(self.embedding_model.get_sentence_embedding_dimension())
            
        return self.embedding_model.encode(text)
    
    def _is_similar(self, item1: Dict[str, Any], item2: Dict[str, Any], threshold: Optional[float] = None, key: str = "conversation") -> bool:
        """
        Check if two items are similar based on their embeddings.
        
        Args:
            item1: First item
            item2: Second item
            threshold: Optional threshold override
            key: The key to use for text extraction from items
            
        Returns:
            True if items are similar, False otherwise
        """
        # Calculate embeddings
        emb1 = self._get_item_embedding(item1, key)
        emb2 = self._get_item_embedding(item2, key)
        
        # Calculate cosine similarity
        similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
        
        # Use provided threshold or default
        thresh = threshold if threshold is not None else self.threshold
        
        return similarity >= thresh 