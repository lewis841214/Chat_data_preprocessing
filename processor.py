#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Base processor module defining common interfaces for all data processing components.
"""

import abc
import logging
from typing import Dict, List, Any, Optional


class BaseProcessor(abc.ABC):
    """
    Abstract base class for all data processors in the pipeline.
    Defines common interface that all components must implement.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the processor with configuration.
        
        Args:
            config: Configuration dictionary for this processor
        """
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
    
    @abc.abstractmethod
    def process(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Process the input data and return the processed output.
        
        Args:
            data: Input data to process
            
        Returns:
            Processed data
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
    
    def _validate_conversation(self, conv: Dict[str, Any]) -> bool:
        """
        Validate that a conversation object has the required structure.
        
        Args:
            conv: Conversation object to validate
            
        Returns:
            True if conversation is valid, False otherwise
        """
        if "conversation" not in conv:
            return False
            
        conversation = conv["conversation"]
        if not isinstance(conversation, list) or not conversation:
            return False
            
        for message in conversation:
            if not isinstance(message, dict):
                return False
                
            # Check required fields
            if "content" not in message or "role" not in message:
                return False
                
        return True 