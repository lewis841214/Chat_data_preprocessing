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
        # Only import and register the selected method to avoid dependency issues
        if self.enabled:
            if self.method_name == "minhash_lsh":
                from deduplication.minhash_lsh import MinHashLSH
                self.register_method("minhash_lsh", MinHashLSH)
            elif self.method_name == "simhash":
                try:
                    from deduplication.simhash import SimHash
                    self.register_method("simhash", SimHash)
                except ImportError:
                    self.logger.error("SimHash not available - missing dependencies")
                    self.enabled = False
            elif self.method_name == "semantic":
                try:
                    from deduplication.semantic_dedup import SemanticDeduplicate
                    self.register_method("semantic", SemanticDeduplicate)
                except ImportError:
                    self.logger.error("Semantic deduplication not available - missing dependencies")
                    self.enabled = False
            elif self.method_name == "suffix_array":
                try:
                    from deduplication.suffix_array import SuffixArrayDedup
                    self.register_method("suffix_array", SuffixArrayDedup)
                except ImportError:
                    self.logger.error("Suffix Array deduplication not available - missing dependencies")
                    self.enabled = False
            elif self.method_name == "dbscan":
                try:
                    from deduplication.dbscan_dedup import DBSCANDedup
                    self.register_method("dbscan", DBSCANDedup)
                except ImportError:
                    self.logger.error("DBSCAN deduplication not available - missing dependencies")
                    self.enabled = False
            elif self.method_name == "bertopic":
                try:
                    from deduplication.bertopic_dedup import BERTopicDedup
                    self.register_method("bertopic", BERTopicDedup)
                except ImportError:
                    self.logger.error("BERTopic deduplication not available - missing dependencies")
                    self.enabled = False
            elif self.method_name == "bloom_filter":
                try:
                    from deduplication.bloom_filter import BloomFilterDedup
                    self.register_method("bloom_filter", BloomFilterDedup)
                except ImportError:
                    self.logger.error("Bloom Filter deduplication not available - missing dependencies")
                    self.enabled = False
        
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
    
    def process(self, data: List[Dict[str, Any]], key: str = "conversation") -> List[Dict[str, Any]]:
        """
        Apply deduplication to the input data.
        
        Args:
            data: Input data to deduplicate
            key: The key to use for text extraction from items
            
        Returns:
            Deduplicated data
        """
        if not self.enabled or not self.method:
            self.logger.info("Deduplication skipped")
            return data
            
        if not self._validate_data(data):
            self.logger.error("Invalid input data format")
            return data
            
        self.logger.info(f"Starting deduplication on {len(data)} items using key: {key}")
        deduplicated_data = self.method.process(data, key=key)
        self.logger.info(f"Deduplication complete: {len(data)} -> {len(deduplicated_data)} items")
        
        return deduplicated_data 