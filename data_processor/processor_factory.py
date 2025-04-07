#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Factory for creating data processor instances.
"""

import logging
from typing import Dict, Any, Type

from data_processor.base_processor import BaseDataProcessor
from data_processor.processors.default_processor import DefaultDataProcessor

class DataProcessorFactory:
    """
    Factory class for creating data processor instances.
    """
    
    # Registry of available processors
    _processors = {
        "default": DefaultDataProcessor,
        # Add other processors as they are implemented
    }
    
    @classmethod
    def register_processor(cls, name: str, processor_class: Type[BaseDataProcessor]) -> None:
        """
        Register a new processor class.
        
        Args:
            name: Name of the processor
            processor_class: Processor class to register
        """
        cls._processors[name] = processor_class
        logging.getLogger(__name__).info(f"Registered data processor: {name}")
    
    @classmethod
    def get_processor(cls, config: Dict[str, Any]) -> BaseDataProcessor:
        """
        Get a data processor instance based on configuration.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            Instance of data processor
        """
        logger = logging.getLogger(__name__)
        
        # Get processor type from config, default to "default"
        processor_type = config.get("data_processing", {}).get("type", "default")
        
        # Check if the processor type is supported
        if processor_type not in cls._processors:
            logger.warning(f"Unsupported processor type: {processor_type}. Using default processor.")
            processor_type = "default"
            
        # Create and return the processor instance
        processor_class = cls._processors[processor_type]
        return processor_class(config) 