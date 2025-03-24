#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Exact deduplication filter for chat data preprocessing.
"""

from typing import Dict, List, Any, Set
from collections import defaultdict

from processor import BaseProcessor


class ExactDeduplicate(BaseProcessor):
    """
    Filter that removes exact duplicates based on content hash.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the exact deduplication filter.
        
        Args:
            config: Configuration dictionary
        """
        super().__init__(config)
        self.hash_function = config.get("hash_function", "md5")
        self.logger.info(f"Initialized exact deduplication filter using {self.hash_function}")
        
    def process(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Remove exact duplicates from the input data.
        
        Args:
            data: Input data to deduplicate
            
        Returns:
            Deduplicated data
        """
        if not data:
            return []
            
        unique_items = {}
        hashes = defaultdict(list)
        
        for i, item in enumerate(data):
            if not self._validate_conversation(item):
                continue
                
            item_hash = self._hash_item(item)
            hashes[item_hash].append(i)
        
        # Keep only one item per hash
        deduplicated = []
        for hash_value, indices in hashes.items():
            # If multiple items have the same hash, select the first one
            deduplicated.append(data[indices[0]])
            
        self.logger.info(f"Exact deduplication: {len(data)} -> {len(deduplicated)} items")
        return deduplicated
    
    def _hash_item(self, item: Dict[str, Any]) -> str:
        """
        Generate a hash for the conversation content.
        
        Args:
            item: Conversation item
            
        Returns:
            Hash value
        """
        import hashlib
        
        if not self._validate_conversation(item):
            return ""
            
        # Create a deterministic string representation of the conversation
        conversation = item["conversation"]
        content_strings = []
        
        for message in conversation:
            role = message.get("role", "")
            content = message.get("content", "")
            content_strings.append(f"{role}:{content}")
            
        conversation_str = "||".join(content_strings)
        
        # Hash using the specified algorithm
        if self.hash_function == "md5":
            return hashlib.md5(conversation_str.encode('utf-8')).hexdigest()
        elif self.hash_function == "sha256":
            return hashlib.sha256(conversation_str.encode('utf-8')).hexdigest()
        else:
            # Default to md5
            return hashlib.md5(conversation_str.encode('utf-8')).hexdigest() 