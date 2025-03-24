#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Stopwords filtering for chat data preprocessing.
"""

import re
from typing import Dict, List, Any, Set, Optional

from processor import BaseProcessor


class StopwordsFilter(BaseProcessor):
    """
    Filter that removes conversations containing a high percentage of stopwords.
    """
    
    # Common stopwords in multiple languages
    DEFAULT_STOPWORDS = {
        # English stopwords
        "the", "be", "to", "of", "and", "a", "in", "that", "have", "I", "it", 
        "for", "not", "on", "with", "he", "as", "you", "do", "at", "this", "but",
        "his", "by", "from", "they", "we", "say", "her", "she", "or", "an", "will",
        "my", "one", "all", "would", "there", "their", "what", "so", "up", "out", "if",
        "about", "who", "get", "which", "go", "me", "when", "make", "can", "like", "time",
        "no", "just", "him", "know", "take", "people", "into", "year", "your", "good", "some",
        "could", "them", "see", "other", "than", "then", "now", "look", "only", "come", "its",
        "over", "think", "also", "back", "after", "use", "two", "how", "our", "work", "first",
        "well", "way", "even", "new", "want", "because", "any", "these", "give", "day", "most", "us"
    }
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the stopwords filter.
        
        Args:
            config: Configuration dictionary
        """
        super().__init__(config)
        
        # Load custom stopwords if provided
        self.stopwords = set(self.DEFAULT_STOPWORDS)
        custom_stopwords = config.get("custom_stopwords", [])
        if custom_stopwords:
            self.stopwords.update(custom_stopwords)
            
        # Maximum allowed percentage of stopwords
        self.max_stopword_ratio = config.get("max_stopword_ratio", 0.5)
        
        # Minimum word count for filtering
        self.min_word_count = config.get("min_word_count", 20)
        
        self.logger.info(f"Initialized stopwords filter with {len(self.stopwords)} stopwords")
        
    def process(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Filter out conversations with a high percentage of stopwords.
        
        Args:
            data: Input data to filter
            
        Returns:
            Filtered data
        """
        filtered_data = []
        
        for item in data:
            if not self._validate_conversation(item):
                continue
                
            conversation = item["conversation"]
            
            # Skip filtering if there are very few words
            words = []
            for message in conversation:
                content = message.get("content", "")
                words.extend(re.findall(r'\b\w+\b', content.lower()))
                
            if len(words) < self.min_word_count:
                filtered_data.append(item)
                continue
                
            # Calculate stopword percentage
            stopword_count = sum(1 for word in words if word in self.stopwords)
            stopword_ratio = stopword_count / len(words) if words else 1.0
            
            # Keep if below threshold
            if stopword_ratio <= self.max_stopword_ratio:
                filtered_data.append(item)
                
        return filtered_data
    
    @classmethod
    def load_stopwords_from_file(cls, filepath: str) -> Set[str]:
        """
        Load stopwords from a file.
        
        Args:
            filepath: Path to the stopwords file (one word per line)
            
        Returns:
            Set of stopwords
        """
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                return {line.strip().lower() for line in f if line.strip()}
        except Exception as e:
            return set(cls.DEFAULT_STOPWORDS) 