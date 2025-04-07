#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Base data processor module defining common interfaces for data processing components.
"""

import abc
import logging
from typing import Dict, List, Any, Optional

class BaseDataProcessor(abc.ABC):
    """
    Abstract base class for all data processors.
    Defines common interface that all data processing components must implement.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the data processor with configuration.
        
        Args:
            config: Configuration dictionary for this processor
        """
        self.config = config.get("data_processing", {})
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Initialize common filtering parameters
        self.filter_words = self.config.get("filter_words", [])
        self.min_length = self.config.get("min_length", 0)
        
    def process(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Process the input data by applying filters and transformations.
        
        Args:
            data: Input data to process
            
        Returns:
            Processed data
        """
        if not self._validate_data(data):
            self.logger.error("Invalid input data format")
            return []
            
        processed_data = []
        for item in data:
            try:
                processed_item = self.process_item(item)
                if processed_item:
                    processed_data.append(processed_item)
            except Exception as e:
                self.logger.error(f"Error processing item: {str(e)}")
                continue
                
        return processed_data
    
    @abc.abstractmethod
    def process_item(self, item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Process a single data item. Must be implemented by specific processors.
        
        Args:
            item: Data item to process
            
        Returns:
            Processed item or None if item should be filtered out
        """
        pass
    
    def _validate_data(self, data: List[Dict[str, Any]]) -> bool:
        """
        Validate that the input data conforms to expected format.
        
        Args:
            data: Input data to validate
            
        Returns:
            True if data is valid, False otherwise
        """
        if not isinstance(data, list):
            self.logger.error(f"Input data is not a list: {type(data)}")
            return False
        
        # Check if all items are dictionaries
        for i, item in enumerate(data):
            if not isinstance(item, dict):
                self.logger.error(f"Item at index {i} is not a dictionary: {type(item)}")
                return False
        
        return True
    
    def _should_filter_message(self, message: str) -> bool:
        """
        Check if a message should be filtered based on configured rules.
        
        Args:
            message: Message text to check
            
        Returns:
            True if message should be filtered out, False otherwise
        """
        # # Check minimum length
        # if len(message) < self.min_length:
        #     return True
        # Check for filtered words
        # if one of self.filter_words are in word, then return True
        if  any(word in message for word in self.filter_words):
            return True
        return False 