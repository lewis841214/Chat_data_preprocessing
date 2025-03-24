#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Base handler for platform-specific data processing.
"""

import abc
import os
import json
import logging
from typing import Dict, List, Any, Optional, Iterator

from processor import BaseProcessor


class BasePlatformHandler(BaseProcessor):
    """
    Base class for platform-specific data handlers.
    Provides common functionality for data loading and transformation.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the platform handler.
        
        Args:
            config: Configuration dictionary for this handler
        """
        super().__init__(config)
        self.input_path = config.get("input_path", "data/input/")
        self.batch_size = config.get("batch_size", 1000)
    
    def process(self, data: Optional[List[Dict[str, Any]]] = None) -> List[Dict[str, Any]]:
        """
        Process the input data and return standardized conversation data.
        If data is None, load from input_path.
        
        Args:
            data: Optional input data to process
            
        Returns:
            Processed data in standardized format
        """
        if data is None:
            # Load data from input path
            self.logger.info(f"Loading data from {self.input_path}")
            data = self.load_data()
        else:
            self.logger.info("Processing provided data")
            
        # Transform to standardized format
        transformed_data = self.transform(data)
        
        self.logger.info(f"Processed {len(transformed_data)} conversations")
        return transformed_data
    
    def load_data(self) -> List[Dict[str, Any]]:
        """
        Load data from the input path.
        
        Returns:
            List of data objects
        """
        data = []
        
        if os.path.isfile(self.input_path):
            # Single file
            data = self._load_file(self.input_path)
        elif os.path.isdir(self.input_path):
            # Directory with multiple files
            for filename in os.listdir(self.input_path):
                file_path = os.path.join(self.input_path, filename)
                if os.path.isfile(file_path):
                    try:
                        file_data = self._load_file(file_path)
                        data.extend(file_data)
                    except Exception as e:
                        self.logger.error(f"Error loading {file_path}: {str(e)}")
        else:
            self.logger.error(f"Input path not found: {self.input_path}")
            
        self.logger.info(f"Loaded {len(data)} data items from {self.input_path}")
        return data
    
    def _load_file(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Load data from a single file.
        
        Args:
            file_path: Path to the data file
            
        Returns:
            Data from the file
        """
        try:
            extension = os.path.splitext(file_path)[1].lower()
            
            if extension == '.json':
                with open(file_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            elif extension in ['.jsonl', '.ndjson']:
                data = []
                with open(file_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip():
                            data.append(json.loads(line))
                return data
            else:
                self.logger.warning(f"Unsupported file format: {extension}")
                return []
                
        except Exception as e:
            self.logger.error(f"Error loading {file_path}: {str(e)}")
            return []
    
    @abc.abstractmethod
    def transform(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Transform platform-specific data to standardized format.
        Must be implemented by platform-specific handlers.
        
        Args:
            data: Platform-specific data
            
        Returns:
            Data in standardized format
        """
        pass
    
    def batch_iterator(self, data: List[Dict[str, Any]]) -> Iterator[List[Dict[str, Any]]]:
        """
        Create batches from data for memory-efficient processing.
        
        Args:
            data: Data to batch
            
        Yields:
            Batches of data
        """
        for i in range(0, len(data), self.batch_size):
            yield data[i:i+self.batch_size] 