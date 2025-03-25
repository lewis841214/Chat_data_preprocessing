#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
MinHash LSH implementation for efficient near-duplicate detection.
"""

import re
import hashlib
import random
from typing import Dict, List, Any, Set, Tuple, Optional, Union, Iterator
from collections import defaultdict
from tqdm import tqdm

from deduplication.dedup_base import DeduplicationMethod


class MinHashLSH(DeduplicationMethod):
    """
    MinHash with Locality Sensitive Hashing for efficient near-duplicate detection.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize MinHash LSH with configuration.
        
        Args:
            config: Configuration dictionary
        """
        super().__init__(config)
        
        # MinHash parameters
        self.ngram_size = config.get("ngram_size", 5)
        self.num_permutations = config.get("num_permutations", 128)
        self.band_size = config.get("band_size", 8)
        
        # Number of bands from permutations and band size
        self.num_bands = self.num_permutations // self.band_size
        if self.num_permutations % self.band_size != 0:
            self.logger.warning(f"num_permutations {self.num_permutations} is not divisible by band_size {self.band_size}")
            self.num_bands = self.num_permutations // self.band_size + 1
            
        # Generate permutation functions
        self.permutations = self._generate_permutations(self.num_permutations)
        
        self.logger.info(f"Initialized MinHash LSH with {self.num_permutations} permutations, "
                         f"{self.band_size} band size, and {self.num_bands} bands")
    
    def process(self, data: List[Dict[str, Any]], key: str = "conversation") -> List[Dict[str, Any]]:
        """
        Apply MinHash LSH deduplication to the input data.
        
        Args:
            data: Input data to deduplicate
            key: The key to use for text extraction from items
            
        Returns:
            Deduplicated data
        """
        if not data:
            return []
            
        self.logger.info(f"Running MinHash LSH on {len(data)} items with key: {key}")
        
        # Step 1: Generate MinHash signatures for all items
        signatures = {}
        for idx, item in tqdm(enumerate(data)):
            text = self._extract_text(item, key)
            if not text:
                continue
                
            signatures[idx] = self._minhash_signature(text)
        
        # Step 2: Apply LSH to find candidate pairs
        candidates = self._lsh_candidates(signatures)
        
        # Step 3: Compute exact Jaccard similarity for candidates
        clusters = self._find_clusters(candidates, signatures, data)
        
        # Step 4: Select representatives from each cluster
        deduplicated = self._select_representatives(clusters, data)
        
        self.logger.info(f"Deduplication reduced {len(data)} to {len(deduplicated)} items "
                         f"({len(clusters)} clusters)")
        
        return deduplicated
    
    def _generate_permutations(self, num_permutations: int) -> List[List[int]]:
        """
        Generate hash permutation functions.
        
        Args:
            num_permutations: Number of permutations to generate
            
        Returns:
            List of permutation parameters (a, b) for each permutation
        """
        # Use a large prime number for modulo
        self.max_hash = (1 << 32) - 1  # Using a 32-bit max hash
        
        # Seed for reproducibility
        random.seed(42)
        
        # Generate random parameters for permutation functions (ax + b) % p
        permutations = []
        for _ in range(num_permutations):
            a = random.randint(1, self.max_hash)
            b = random.randint(0, self.max_hash)
            permutations.append((a, b))
            
        return permutations
    
    def _get_ngrams(self, text: str) -> Set[str]:
        """
        Extract character n-grams from text.
        
        Args:
            text: Input text
            
        Returns:
            Set of n-grams
        """
        # Clean text and convert to lowercase
        text = re.sub(r'\s+', ' ', text.lower()).strip()
        
        # Generate n-grams
        ngrams = set()
        for i in range(len(text) - self.ngram_size + 1):
            ngrams.add(text[i:i + self.ngram_size])
            
        return ngrams
    
    def _hash_ngram(self, ngram: str) -> int:
        """
        Hash an n-gram to an integer.
        
        Args:
            ngram: Input n-gram
            
        Returns:
            Hash value
        """
        return int(hashlib.md5(ngram.encode('utf-8')).hexdigest(), 16) % self.max_hash
    
    def _minhash_signature(self, text: str) -> List[int]:
        """
        Generate MinHash signature for text.
        
        Args:
            text: Input text
            
        Returns:
            MinHash signature (list of hash values)
        """
        ngrams = self._get_ngrams(text)
        
        # Hash all n-grams
        hashes = [self._hash_ngram(ngram) for ngram in ngrams]
        
        # Apply permutations to find minimum hash value for each
        signature = []
        for a, b in self.permutations:
            min_hash = float('inf')
            for h in hashes:
                # Apply permutation function: (a*h + b) % max_hash
                permuted_hash = (a * h + b) % self.max_hash
                min_hash = min(min_hash, permuted_hash)
            signature.append(min_hash)
            
        return signature
    
    def _lsh_candidates(self, signatures: Dict[int, List[int]]) -> List[Tuple[int, int]]:
        """
        Use LSH to find candidate pairs of similar items.
        
        Args:
            signatures: Dictionary mapping item indexes to their signatures
            
        Returns:
            List of candidate pairs (index1, index2)
        """
        # Initialize band buckets
        buckets = [defaultdict(list) for _ in range(self.num_bands)]
        
        # Hash each signature into band buckets
        for idx, signature in signatures.items():
            for band_idx in range(self.num_bands):
                # Get the band segment of the signature
                start = band_idx * self.band_size
                end = min(start + self.band_size, len(signature))
                band = tuple(signature[start:end])
                
                # Add to the bucket
                buckets[band_idx][band].append(idx)
        
        # Find candidate pairs from buckets
        candidates = set()
        for band_bucket in tqdm(buckets):
            for items in band_bucket.values():
                if len(items) > 1:
                    # Add all pairs in the same bucket
                    for i in range(len(items)):
                        for j in range(i + 1, len(items)):
                            candidates.add((min(items[i], items[j]), max(items[i], items[j])))
        
        return list(candidates)
    
    def _jaccard_similarity(self, sig1: List[int], sig2: List[int]) -> float:
        """
        Compute Jaccard similarity from MinHash signatures.
        
        Args:
            sig1: First signature
            sig2: Second signature
            
        Returns:
            Estimated Jaccard similarity
        """
        # Estimate Jaccard similarity as the proportion of matching hash values
        matches = sum(1 for x, y in zip(sig1, sig2) if x == y)
        return matches / len(sig1)
    
    def _find_clusters(self, candidates: List[Tuple[int, int]], 
                       signatures: Dict[int, List[int]], 
                       data: List[Dict[str, Any]]) -> List[List[int]]:
        """
        Find clusters of similar items using transitive closure.
        
        Args:
            candidates: List of candidate pairs
            signatures: Dictionary of MinHash signatures
            data: Original data items
            
        Returns:
            List of clusters (each cluster is a list of item indexes)
        """
        # Build graph from similar pairs
        graph = defaultdict(set)
        
        for idx1, idx2 in tqdm(candidates):
            # Compute exact Jaccard similarity
            sim = self._jaccard_similarity(signatures[idx1], signatures[idx2])
            
            if sim >= self.threshold:
                graph[idx1].add(idx2)
                graph[idx2].add(idx1)
        
        # Find connected components (clusters)
        visited = set()
        clusters = []
        
        for idx in signatures.keys():
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
        
        # Add singletons for items not in any cluster
        for idx in range(len(data)):
            if idx in signatures and idx not in visited:
                clusters.append([idx])
                
        return clusters
    
    def _select_representatives(self, clusters: List[List[int]], 
                               data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Select representative items from each cluster.
        
        Args:
            clusters: List of clusters
            data: Original data items
            
        Returns:
            List of representatives
        """
        representatives = []
        
        for cluster in tqdm(clusters):
            if len(cluster) == 1:
                # Singleton cluster
                representatives.append(data[cluster[0]])
            else:
                # Select representative from the cluster
                cluster_items = [data[idx] for idx in cluster]
                representative = self._select_representative(cluster_items)
                representatives.append(representative)
                
        return representatives 