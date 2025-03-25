#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
SimHash implementation for efficient near-duplicate detection.
"""

import re
import hashlib
import numpy as np
from typing import Dict, List, Any, Set, Optional, Tuple
from collections import defaultdict

from deduplication.dedup_base import DeduplicationMethod


class SimHash(DeduplicationMethod):
    """
    SimHash algorithm for efficient near-duplicate detection.
    Uses feature hashing for dimension reduction and Hamming distance for similarity.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize SimHash with configuration.
        
        Args:
            config: Configuration dictionary
        """
        super().__init__(config)
        
        # SimHash parameters
        self.hash_bits = config.get("hash_bits", 64)  # Number of bits in the hash
        self.ngram_size = config.get("ngram_size", 3)  # Size of n-grams
        self.max_hamming_distance = config.get("max_hamming_distance", 3)  # Max Hamming distance for similarity
        
        self.logger.info(f"Initialized SimHash with {self.hash_bits} bits, "
                         f"{self.ngram_size} n-gram size, "
                         f"and {self.max_hamming_distance} max Hamming distance")
    
    def process(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Apply SimHash deduplication to the input data.
        
        Args:
            data: Input data to deduplicate
            
        Returns:
            Deduplicated data
        """
        if not data:
            return []
            
        self.logger.info(f"Running SimHash on {len(data)} items")
        
        # Step 1: Generate SimHash for all items
        hash_values = {}
        for idx, item in enumerate(data):
            text = self._extract_text(item)
            if not text:
                continue
                
            hash_values[idx] = self._compute_simhash(text)
        
        # Step 2: Find similar items (clusters) based on Hamming distance
        clusters = self._find_clusters(hash_values)
        
        # Step 3: Select representatives from each cluster
        deduplicated = self._select_representatives(clusters, data)
        
        self.logger.info(f"Deduplication reduced {len(data)} to {len(deduplicated)} items "
                         f"({len(clusters)} clusters)")
        
        return deduplicated
    
    def _get_ngrams(self, text: str) -> List[str]:
        """
        Extract character n-grams from text.
        
        Args:
            text: Input text
            
        Returns:
            List of n-grams
        """
        # Clean text and convert to lowercase
        text = re.sub(r'\s+', ' ', text.lower()).strip()
        
        # Generate n-grams
        ngrams = []
        for i in range(len(text) - self.ngram_size + 1):
            ngrams.append(text[i:i + self.ngram_size])
            
        return ngrams
    
    def _hash_feature(self, feature: str) -> int:
        """
        Hash a feature to an integer.
        
        Args:
            feature: Input feature (n-gram)
            
        Returns:
            Hash value
        """
        return int(hashlib.md5(feature.encode('utf-8')).hexdigest(), 16)
    
    def _compute_simhash(self, text: str) -> int:
        """
        Compute SimHash value for text.
        
        Args:
            text: Input text
            
        Returns:
            SimHash value as an integer
        """
        # Initialize weight vector for each bit
        v = [0] * self.hash_bits
        
        # Get features (n-grams) from text
        features = self._get_ngrams(text)
        
        # For each feature, update the weight vector
        for feature in features:
            feature_hash = self._hash_feature(feature)
            
            # Update each bit based on the feature hash
            for i in range(self.hash_bits):
                bit_mask = 1 << i
                if feature_hash & bit_mask:
                    v[i] += 1
                else:
                    v[i] -= 1
        
        # Generate final hash based on the weight vector
        simhash = 0
        for i in range(self.hash_bits):
            if v[i] > 0:
                simhash |= (1 << i)
                
        return simhash
    
    def _hamming_distance(self, hash1: int, hash2: int) -> int:
        """
        Compute Hamming distance between two hash values.
        
        Args:
            hash1: First hash value
            hash2: Second hash value
            
        Returns:
            Hamming distance (number of different bits)
        """
        xor = hash1 ^ hash2
        distance = 0
        
        # Count the number of set bits in the XOR result
        while xor:
            distance += 1
            xor &= xor - 1  # Clear the least significant bit
            
        return distance
    
    def _find_clusters(self, hash_values: Dict[int, int]) -> List[List[int]]:
        """
        Find clusters of similar items using SimHash and Hamming distance.
        
        Args:
            hash_values: Dictionary mapping item indexes to their SimHash values
            
        Returns:
            List of clusters (each cluster is a list of item indexes)
        """
        # Build graph from similar pairs
        graph = defaultdict(set)
        
        # Compare all pairs of items
        items = list(hash_values.keys())
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                idx1, idx2 = items[i], items[j]
                
                # Calculate Hamming distance
                distance = self._hamming_distance(hash_values[idx1], hash_values[idx2])
                
                if distance <= self.max_hamming_distance:
                    graph[idx1].add(idx2)
                    graph[idx2].add(idx1)
        
        # Find connected components (clusters)
        visited = set()
        clusters = []
        
        for idx in hash_values.keys():
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
    
    def _is_similar(self, item1: Dict[str, Any], item2: Dict[str, Any], threshold: Optional[float] = None) -> bool:
        """
        Check if two items are similar based on SimHash and Hamming distance.
        
        Args:
            item1: First item
            item2: Second item
            threshold: Optional threshold override (not used in SimHash)
            
        Returns:
            True if items are similar, False otherwise
        """
        text1 = self._extract_text(item1)
        text2 = self._extract_text(item2)
        
        if not text1 or not text2:
            return False
            
        hash1 = self._compute_simhash(text1)
        hash2 = self._compute_simhash(text2)
        
        distance = self._hamming_distance(hash1, hash2)
        
        return distance <= self.max_hamming_distance 