#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Deduplication manager for near-duplicate detection in chat data.
"""

import logging
from typing import Dict, List, Any, Type, Optional

from processor import BaseProcessor
from deduplication.dedup_base import DeduplicationMethod


class DedupManager(BaseProcessor):
    """
    Manager for deduplication methods.
    Selects and applies the appropriate deduplication method based on configuration.
    """
    
    # Registry of available deduplication methods
    _methods = {}
    
    @classmethod
    def register_method(cls, name: str, method_class: Type[DeduplicationMethod]) -> None:
        """
        Register a deduplication method.
        
        Args:
            name: Name of the method
            method_class: Deduplication method class
        """
        cls._methods[name] = method_class
        logging.getLogger(__name__).info(f"Registered deduplication method: {name}")
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the deduplication manager.
        
        Args:
            config: Configuration dictionary
        """
        super().__init__(config)
        
        self.enabled = config.get("enabled", True)
        self.method_name = config.get("method", "minhash_lsh")
        
        # Lazy import to avoid circular dependencies
        # Register available methods
        from deduplication.minhash_lsh import MinHashLSH
        self.register_method("minhash_lsh", MinHashLSH)
        
        # Import and register other methods conditionally
        try:
            from deduplication.simhash import SimHash
            self.register_method("simhash", SimHash)
        except ImportError:
            self.logger.warning("SimHash not available")
            
        try:
            from deduplication.semantic_dedup import SemanticDeduplicate
            self.register_method("semantic", SemanticDeduplicate)
        except ImportError:
            self.logger.warning("Semantic deduplication not available")
            
        try:
            from deduplication.suffix_array import SuffixArrayDedup
            self.register_method("suffix_array", SuffixArrayDedup)
        except ImportError:
            self.logger.warning("Suffix Array deduplication not available")
            
        try:
            from deduplication.dbscan_dedup import DBSCANDedup
            self.register_method("dbscan", DBSCANDedup)
        except ImportError:
            self.logger.warning("DBSCAN deduplication not available")
            
        try:
            from deduplication.bertopic_dedup import BERTopicDedup
            self.register_method("bertopic", BERTopicDedup)
        except ImportError:
            self.logger.warning("BERTopic deduplication not available")
            
        try:
            from deduplication.bloom_filter import BloomFilterDedup
            self.register_method("bloom_filter", BloomFilterDedup)
        except ImportError:
            self.logger.warning("Bloom Filter deduplication not available")
        
        # Initialize the selected method
        if not self.enabled:
            self.method = None
            self.logger.info("Deduplication disabled")
        elif self.method_name in self._methods:
            method_class = self._methods[self.method_name]
            self.method = method_class(config)
            self.logger.info(f"Using deduplication method: {self.method_name}")
        else:
            self.method = None
            self.logger.error(f"Deduplication method '{self.method_name}' not found")
    
    def process(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Apply deduplication to the input data.
        
        Args:
            data: Input data to deduplicate
            
        Returns:
            Deduplicated data
        """
        if not self.enabled or not self.method:
            self.logger.info("Deduplication skipped")
            return data
            
        if not self._validate_data(data):
            self.logger.error("Invalid input data format")
            return data
            
        self.logger.info(f"Starting deduplication on {len(data)} items")
        deduplicated_data = self.method.process(data)
        self.logger.info(f"Deduplication complete: {len(data)} -> {len(deduplicated_data)} items")
        
        return deduplicated_data 