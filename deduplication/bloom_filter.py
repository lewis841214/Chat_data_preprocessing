#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Bloom Filter implementation for fast exact duplicate detection.
"""

import hashlib
import re
from typing import Dict, List, Any, Set, Optional, Tuple, Callable
from collections import defaultdict

try:
    import mmh3  # MurmurHash3 for better hash functions
    MMHASH_AVAILABLE = True
except ImportError:
    MMHASH_AVAILABLE = False
    import random
    import logging
    logging.getLogger(__name__).warning(
        "mmh3 not installed. Using fallback hashing. Install with: pip install mmh3"
    )

from deduplication.dedup_base import DeduplicationMethod


class BloomFilter:
    """
    Bloom filter implementation for fast membership testing.
    """
    
    def __init__(self, capacity: int, error_rate: float = 0.01):
        """
        Initialize a Bloom filter.
        
        Args:
            capacity: Expected number of elements
            error_rate: False positive probability
        """
        # Calculate optimal size and number of hash functions
        self.size = self._optimal_size(capacity, error_rate)
        self.hash_count = self._optimal_hash_count(self.size, capacity)
        
        # Initialize bit array
        self.bit_array = [0] * self.size
        
        # Use MurmurHash3 if available, otherwise fallback to MD5
        if MMHASH_AVAILABLE:
            self.hash_func = self._mmh3_hash
        else:
            self.hash_func = self._md5_hash
    
    def _optimal_size(self, capacity: int, error_rate: float) -> int:
        """
        Calculate optimal bit array size.
        
        Args:
            capacity: Expected number of elements
            error_rate: False positive probability
            
        Returns:
            Optimal bit array size
        """
        import math
        size = -capacity * math.log(error_rate) / (math.log(2) ** 2)
        return int(size) + 1
    
    def _optimal_hash_count(self, size: int, capacity: int) -> int:
        """
        Calculate optimal number of hash functions.
        
        Args:
            size: Bit array size
            capacity: Expected number of elements
            
        Returns:
            Optimal number of hash functions
        """
        import math
        hash_count = size / capacity * math.log(2)
        return int(hash_count) + 1
    
    def _mmh3_hash(self, item: str, seed: int) -> int:
        """
        Generate hash using MurmurHash3.
        
        Args:
            item: Item to hash
            seed: Hash seed
            
        Returns:
            Hash value
        """
        return mmh3.hash(item, seed) % self.size
    
    def _md5_hash(self, item: str, seed: int) -> int:
        """
        Generate hash using MD5 (fallback).
        
        Args:
            item: Item to hash
            seed: Hash seed
            
        Returns:
            Hash value
        """
        md5 = hashlib.md5()
        md5.update(f"{item}{seed}".encode('utf-8'))
        return int(md5.hexdigest(), 16) % self.size
    
    def add(self, item: str) -> None:
        """
        Add an item to the Bloom filter.
        
        Args:
            item: Item to add
        """
        for i in range(self.hash_count):
            index = self.hash_func(item, i)
            self.bit_array[index] = 1
    
    def contains(self, item: str) -> bool:
        """
        Check if an item might be in the Bloom filter.
        
        Args:
            item: Item to check
            
        Returns:
            True if item might be in the filter, False if definitely not
        """
        for i in range(self.hash_count):
            index = self.hash_func(item, i)
            if self.bit_array[index] == 0:
                return False
        return True


class BloomFilterDedup(DeduplicationMethod):
    """
    Deduplication using Bloom filters for fast exact duplicate detection.
    Unlike other methods, this focuses on exact duplicates only.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Bloom filter deduplication with configuration.
        
        Args:
            config: Configuration dictionary
        """
        super().__init__(config)
        
        # Bloom filter parameters
        self.capacity = config.get("capacity", 10000)
        self.error_rate = config.get("error_rate", 0.001)
        self.ignore_case = config.get("ignore_case", True)
        self.normalize_whitespace = config.get("normalize_whitespace", True)
        
        self.logger.info(f"Initialized Bloom Filter deduplication with capacity={self.capacity}, "
                         f"error_rate={self.error_rate}")
    
    def process(self, data: List[Dict[str, Any]], key: str = "conversation") -> List[Dict[str, Any]]:
        """
        Apply Bloom filter deduplication to the input data.
        
        Args:
            data: Input data to deduplicate
            key: The key to use for text extraction from items
            
        Returns:
            Deduplicated data
        """
        if not data:
            return []
            
        self.logger.info(f"Running Bloom Filter deduplication on {len(data)} items with key: {key}")
        
        # Create a bloom filter
        bloom_filter = BloomFilter(self.capacity, self.error_rate)
        
        # Track duplicate indices
        seen_texts = set()
        duplicates = set()
        
        # Find duplicates
        for idx, item in enumerate(data):
            text = self._extract_text(item, key)
            if not text:
                continue
                
            # Normalize text if needed
            text = self._normalize_text(text)
            
            # First check if text might be in the filter
            if bloom_filter.contains(text):
                # If potentially in filter, check exact duplicates
                if text in seen_texts:
                    duplicates.add(idx)
            else:
                # Add to filter and seen texts
                bloom_filter.add(text)
                seen_texts.add(text)
        
        # Create deduplicated data
        deduplicated = [item for idx, item in enumerate(data) if idx not in duplicates]
        
        self.logger.info(f"Deduplication reduced {len(data)} to {len(deduplicated)} items "
                         f"(removed {len(duplicates)} exact duplicates)")
        
        return deduplicated
    
    def _normalize_text(self, text: str) -> str:
        """
        Normalize text for comparison.
        
        Args:
            text: Input text
            
        Returns:
            Normalized text
        """
        if self.ignore_case:
            text = text.lower()
            
        if self.normalize_whitespace:
            text = re.sub(r'\s+', ' ', text).strip()
            
        return text
    
    def _is_similar(self, item1: Dict[str, Any], item2: Dict[str, Any], threshold: Optional[float] = None, key: str = "conversation") -> bool:
        """
        Check if two items are exact duplicates.
        
        Args:
            item1: First item
            item2: Second item
            threshold: Optional threshold override (not used in Bloom Filter)
            key: The key to use for text extraction from items
            
        Returns:
            True if items are exact duplicates, False otherwise
        """
        text1 = self._extract_text(item1, key)
        text2 = self._extract_text(item2, key)
        
        if not text1 or not text2:
            return False
            
        # Normalize texts
        text1 = self._normalize_text(text1)
        text2 = self._normalize_text(text2)
        
        # For Bloom Filter, items are similar only if they are exact matches
        return text1 == text2 