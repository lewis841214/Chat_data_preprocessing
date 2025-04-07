#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Suffix Array implementation for efficient substring-based deduplication.
"""

import re
from typing import Dict, List, Any, Set, Optional, Tuple
from collections import defaultdict
import numpy as np

try:
    import suffixtree
    SUFFIXTREE_AVAILABLE = True
except ImportError:
    SUFFIXTREE_AVAILABLE = False
    import logging
    logging.getLogger(__name__).warning(
        "python-suffix-tree not installed. Install with: pip install python-suffix-tree"
    )

from deduplication.dedup_base import DeduplicationMethod


class SuffixArrayDedup(DeduplicationMethod):
    """
    Deduplication using suffix arrays to find common substrings.
    Effective for detecting near-duplicates with shared text segments.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize suffix array deduplication with configuration.
        
        Args:
            config: Configuration dictionary
        """
        super().__init__(config)
        
        if not SUFFIXTREE_AVAILABLE:
            raise ImportError(
                "python-suffix-tree is required for suffix array deduplication. "
                "Install with: pip install python-suffix-tree"
            )
        
        # Parameters
        self.min_substring_length = config.get("min_substring_length", 20)
        self.similarity_threshold = config.get("similarity_threshold", 0.5)
        
        self.logger.info(f"Initialized Suffix Array deduplication with min substring length "
                         f"{self.min_substring_length} and similarity threshold {self.similarity_threshold}")
    
    def process(self, data: List[Dict[str, Any]], key: str = "conversation") -> List[Dict[str, Any]]:
        """
        Apply suffix array deduplication to the input data.
        
        Args:
            data: Input data to deduplicate
            key: The key to use for text extraction from items
            
        Returns:
            Deduplicated data
        """
        if not data:
            return []
            
        self.logger.info(f"Running suffix array deduplication on {len(data)} items with key: {key}")
        
        # Step 1: Preprocess texts and build item mapping
        texts = []
        valid_indices = []
        item_texts = {}
        
        for idx, item in enumerate(data):
            text = self._extract_text(item, key)
            if text:
                # Clean text
                text = self._preprocess_text(text)
                if len(text) < self.min_substring_length:
                    continue
                    
                texts.append(text)
                valid_indices.append(idx)
                item_texts[idx] = text
        
        if not texts:
            return data
            
        # Step 2: Find similar item pairs based on common substrings
        similarity_pairs = self._find_similar_pairs(item_texts)
        
        # Step 3: Group similar items into clusters
        clusters = self._build_clusters(similarity_pairs)
        
        # Step 4: Select representatives from each cluster
        deduplicated = self._select_representatives(clusters, data)
        
        self.logger.info(f"Deduplication reduced {len(data)} to {len(deduplicated)} items "
                         f"({len(clusters)} clusters)")
        
        return deduplicated
    
    def _preprocess_text(self, text: str) -> str:
        """
        Preprocess text for suffix array analysis.
        
        Args:
            text: Input text
            
        Returns:
            Preprocessed text
        """
        # Convert to lowercase
        text = text.lower()
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Remove special characters if needed
        # text = re.sub(r'[^\w\s]', '', text)
        
        return text
    
    def _find_longest_common_substring(self, text1: str, text2: str) -> Tuple[str, int]:
        """
        Find the longest common substring between two texts.
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Tuple of (common substring, length)
        """
        # Create suffix tree
        tree = suffixtree.SuffixTree({0: text1, 1: text2})
        lcs = tree.longest_common_substrings()
        
        if not lcs:
            return "", 0
            
        # Return the first (longest) common substring and its length
        return lcs[0][0], len(lcs[0][0])
    
    def _string_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate string similarity based on longest common substring.
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Similarity score between 0 and 1
        """
        _, lcs_length = self._find_longest_common_substring(text1, text2)
        
        # No common substring found or too short
        if lcs_length < self.min_substring_length:
            return 0.0
            
        # Normalize by the length of the shorter text
        shorter_length = min(len(text1), len(text2))
        similarity = lcs_length / shorter_length
        
        return similarity
    
    def _find_similar_pairs(self, item_texts: Dict[int, str]) -> List[Tuple[int, int, float]]:
        """
        Find pairs of similar items based on common substrings.
        
        Args:
            item_texts: Dictionary mapping item indexes to their text
            
        Returns:
            List of (idx1, idx2, similarity) tuples
        """
        similar_pairs = []
        
        # Compare all pairs of items
        items = list(item_texts.keys())
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                idx1, idx2 = items[i], items[j]
                
                # Calculate string similarity
                similarity = self._string_similarity(item_texts[idx1], item_texts[idx2])
                
                if similarity >= self.similarity_threshold:
                    similar_pairs.append((idx1, idx2, similarity))
        
        return similar_pairs
    
    def _build_clusters(self, similar_pairs: List[Tuple[int, int, float]]) -> List[List[int]]:
        """
        Build clusters from similar pairs using connected components.
        
        Args:
            similar_pairs: List of (idx1, idx2, similarity) tuples
            
        Returns:
            List of clusters (each cluster is a list of item indexes)
        """
        # Build graph from similar pairs
        graph = defaultdict(set)
        
        for idx1, idx2, _ in similar_pairs:
            graph[idx1].add(idx2)
            graph[idx2].add(idx1)
        
        # Find connected components (clusters)
        visited = set()
        clusters = []
        
        for idx in graph.keys():
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
    
    def _is_similar(self, item1: Dict[str, Any], item2: Dict[str, Any], threshold: Optional[float] = None, key: str = "conversation") -> bool:
        """
        Check if two items are similar based on common substrings.
        
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
            
        # Preprocess texts
        text1 = self._preprocess_text(text1)
        text2 = self._preprocess_text(text2)
        
        # Calculate similarity
        similarity = self._string_similarity(text1, text2)
        
        threshold = threshold if threshold is not None else self.similarity_threshold
        return similarity >= threshold 