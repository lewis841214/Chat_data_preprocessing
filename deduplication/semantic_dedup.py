#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Semantic deduplication using text embeddings and cosine similarity.
"""

import numpy as np
from typing import Dict, List, Any, Set, Optional, Tuple
from collections import defaultdict
import logging

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    logging.getLogger(__name__).warning(
        "sentence-transformers not installed. Install with: pip install sentence-transformers"
    )

from deduplication.dedup_base import DeduplicationMethod


class SemanticDeduplicate(DeduplicationMethod):
    """
    Semantic deduplication using text embeddings and cosine similarity.
    Uses sentence transformers to generate text embeddings.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize semantic deduplication with configuration.
        
        Args:
            config: Configuration dictionary
        """
        super().__init__(config)
        
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "sentence-transformers is required for semantic deduplication. "
                "Install with: pip install sentence-transformers"
            )
        
        # Semantic parameters
        self.model_name = config.get("model_name", "all-MiniLM-L6-v2")
        self.batch_size = config.get("batch_size", 32)
        
        # Load the embedding model
        self.logger.info(f"Loading sentence transformer model: {self.model_name}")
        self.model = SentenceTransformer(self.model_name)
        
        self.logger.info(f"Initialized Semantic deduplication with model {self.model_name}")
    
    def process(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Apply semantic deduplication to the input data.
        
        Args:
            data: Input data to deduplicate
            
        Returns:
            Deduplicated data
        """
        if not data:
            return []
            
        self.logger.info(f"Running semantic deduplication on {len(data)} items")
        
        # Step 1: Extract text from each item
        texts = []
        valid_indices = []
        
        for idx, item in enumerate(data):
            text = self._extract_text(item)
            if text:
                texts.append(text)
                valid_indices.append(idx)
        
        if not texts:
            return data
        breakpoint()
        # Step 2: Generate embeddings for all texts
        self.logger.info(f"Generating embeddings for {len(texts)} texts")
        embeddings = self.model.encode(texts, batch_size=self.batch_size)
        
        # Create a mapping from original indices to embeddings
        index_to_embedding = {idx: embeddings[i] for i, idx in enumerate(valid_indices)}
        
        # Step 3: Find similar items based on cosine similarity
        clusters = self._find_clusters(index_to_embedding)
        
        # Step 4: Select representatives from each cluster
        deduplicated = self._select_representatives(clusters, data)
        
        self.logger.info(f"Deduplication reduced {len(data)} to {len(deduplicated)} items "
                         f"({len(clusters)} clusters)")
        
        return deduplicated
    
    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        Compute cosine similarity between two vectors.
        
        Args:
            vec1: First vector
            vec2: Second vector
            
        Returns:
            Cosine similarity in [0, 1] range
        """
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
            
        return dot_product / (norm1 * norm2)
    
    def _find_clusters(self, index_to_embedding: Dict[int, np.ndarray]) -> List[List[int]]:
        """
        Find clusters of similar items using cosine similarity.
        
        Args:
            index_to_embedding: Dictionary mapping item indexes to their embeddings
            
        Returns:
            List of clusters (each cluster is a list of item indexes)
        """
        # Build graph from similar pairs
        graph = defaultdict(set)
        
        # Compare all pairs of items
        items = list(index_to_embedding.keys())
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                idx1, idx2 = items[i], items[j]
                
                # Calculate cosine similarity
                similarity = self._cosine_similarity(
                    index_to_embedding[idx1], 
                    index_to_embedding[idx2]
                )
                
                if similarity >= self.threshold:
                    graph[idx1].add(idx2)
                    graph[idx2].add(idx1)
        
        # Find connected components (clusters)
        visited = set()
        clusters = []
        
        for idx in index_to_embedding.keys():
            if idx in visited:
                continue
                
            # BFS to find connected component
            cluster = []
            queue = [idx]
            visited.add(idx)
            
            while queue:
                node = queue.pop(0)
                cluster.append(node)
                
                for neighbor in graph[node]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append(neighbor)
            
            if len(cluster) > 1:  # Only add clusters with more than one item
                clusters.append(cluster)
        
        return clusters
    
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
    
    def _is_similar(self, item1: Dict[str, Any], item2: Dict[str, Any], threshold: Optional[float] = None) -> bool:
        """
        Check if two items are similar based on semantic similarity.
        
        Args:
            item1: First item
            item2: Second item
            threshold: Optional threshold override
            
        Returns:
            True if items are similar, False otherwise
        """
        text1 = self._extract_text(item1)
        text2 = self._extract_text(item2)
        
        if not text1 or not text2:
            return False
            
        # Generate embeddings
        emb1 = self.model.encode(text1)
        emb2 = self.model.encode(text2)
        
        # Calculate similarity
        similarity = self._cosine_similarity(emb1, emb2)
        
        threshold = threshold if threshold is not None else self.threshold
        return similarity >= threshold 