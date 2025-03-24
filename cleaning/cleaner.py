#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Main cleaning module for removing useless content from chat data.
"""

import re
from typing import Dict, List, Any, Set, Optional

from processor import BaseProcessor
from cleaning.stopwords_filter import StopwordsFilter
from cleaning.language_filter import LanguageFilter
from cleaning.url_filter import URLFilter
from cleaning.paragraph_filter import ParagraphFilter
from cleaning.exact_dedup import ExactDeduplicate


class Cleaner(BaseProcessor):
    """
    Main cleaner class that orchestrates all cleaning operations.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the cleaner with configuration.
        
        Args:
            config: Configuration dictionary for cleaning operations
        """
        super().__init__(config)
        
        # Initialize filters based on config
        self.filters = []
        
        if config.get("stopwords_enabled", True):
            self.filters.append(StopwordsFilter(config))
            
        if config.get("language_filter_enabled", True):
            self.filters.append(LanguageFilter(config))
            
        if config.get("url_filter_enabled", True):
            self.filters.append(URLFilter(config))
            
        if config.get("paragraph_filter_enabled", True):
            self.filters.append(ParagraphFilter(config))
            
        if config.get("exact_dedup_enabled", True):
            self.filters.append(ExactDeduplicate(config))
            
        # Basic length filters
        self.min_length = config.get("min_length", 10)
        self.max_length = config.get("max_length", 32768)
        
        self.logger.info(f"Initialized cleaner with {len(self.filters)} filters")
        
    def process(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Process the input data by applying all cleaning filters.
        
        Args:
            data: Input data to clean
            
        Returns:
            Cleaned data
        """
        if not self._validate_data(data):
            self.logger.error("Invalid input data format")
            return []
            
        self.logger.info(f"Starting cleaning process on {len(data)} items")
        
        # Apply length filtering
        length_filtered = self._apply_length_filtering(data)
        self.logger.info(f"Length filtering: {len(data)} -> {len(length_filtered)} items")
        
        # Apply all registered filters
        filtered_data = length_filtered
        for filter_obj in self.filters:
            filter_name = filter_obj.__class__.__name__
            filtered_data = filter_obj.process(filtered_data)
            self.logger.info(f"{filter_name}: {len(filtered_data)} items remaining")
            
        self.logger.info(f"Cleaning complete: {len(data)} -> {len(filtered_data)} items")
        return filtered_data
    
    def _apply_length_filtering(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Filter out conversations that are too short or too long.
        
        Args:
            data: Input data to filter
            
        Returns:
            Length-filtered data
        """
        filtered_data = []
        
        for item in data:
            if not self._validate_conversation(item):
                continue
                
            conversation = item["conversation"]
            
            # Check if any message is too short
            too_short = False
            for message in conversation:
                content = message.get("content", "")
                if len(content) < self.min_length:
                    too_short = True
                    break
                    
            if too_short:
                continue
                
            # Check total conversation length
            total_length = sum(len(message.get("content", "")) for message in conversation)
            if total_length > self.max_length:
                continue
                
            filtered_data.append(item)
            
        return filtered_data 