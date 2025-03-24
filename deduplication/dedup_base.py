#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Base class for deduplication methods.
"""

import abc
from typing import Dict, List, Any, Set, Tuple, Optional

from processor import BaseProcessor


class DeduplicationMethod(BaseProcessor, abc.ABC):
    """
    Abstract base class for all deduplication methods.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the deduplication method.
        
        Args:
            config: Configuration dictionary
        """
        super().__init__(config)
        self.threshold = config.get("threshold", 0.8)
        
    @abc.abstractmethod
    def process(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Apply deduplication to the input data.
        
        Args:
            data: Input data to deduplicate
            
        Returns:
            Deduplicated data
        """
        pass
    
    def _extract_text(self, item: Dict[str, Any]) -> str:
        """
        Extract text from a conversation for deduplication.
        
        Args:
            item: Conversation item
            
        Returns:
            Extracted text
        """
        if not self._validate_conversation(item):
            return ""
            
        conversation = item["conversation"]
        texts = []
        
        for message in conversation:
            role = message.get("role", "")
            content = message.get("content", "")
            texts.append(f"{role}: {content}")
            
        return "\n".join(texts)
    
    def _select_representative(self, items: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Select a representative item from a group of near-duplicates.
        Default implementation selects the longest one.
        
        Args:
            items: List of near-duplicate items
            
        Returns:
            Representative item
        """
        if not items:
            return {}
            
        # Default strategy: select the one with the most total content
        return max(items, key=lambda x: sum(len(msg.get("content", "")) 
                                           for msg in x.get("conversation", [])))
    
    def _is_similar(self, item1: Dict[str, Any], item2: Dict[str, Any], threshold: Optional[float] = None) -> bool:
        """
        Check if two items are similar based on the method's similarity measure.
        Must be implemented by concrete deduplication methods.
        
        Args:
            item1: First item
            item2: Second item
            threshold: Optional threshold override
            
        Returns:
            True if items are similar, False otherwise
        """
        raise NotImplementedError("Similarity check must be implemented by deduplication methods") 