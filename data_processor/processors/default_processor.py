#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Default data processor implementation.
"""

from typing import Dict, List, Any, Optional

from data_processor.base_processor import BaseDataProcessor
from data_processor.role_transfer import apply_role_transfer

class DefaultDataProcessor(BaseDataProcessor):
    """
    Default implementation of data processor.
    Handles basic message filtering and cleaning.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the default data processor.
        
        Args:
            config: Configuration dictionary
        """
        super().__init__(config)
        # Get role transfer configuration
        self.role_transfer_config = self.config.get("role_transfer", {})
        self.assistant_to_user = self.role_transfer_config.get("assistant_to_user", [])
        self.user_to_assistant = self.role_transfer_config.get("user_to_assistant", [])
    
    def process_item(self, item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Process a single conversation item.
        
        Args:
            item: Conversation data to process
            
        Returns:
            Processed conversation data or None if should be filtered
        """
        # Get messages from the conversation
        messages = item.get("conversation", [])
        if not messages:
            return None
            
        # Filter messages
        filtered_messages = []
        for message in messages:
            content = message.get("content", "")
            
            # Skip empty messages
            if not content:
                continue
                
            # Check if message should be filtered
            if self._should_filter_message(content):
                continue
                
            filtered_messages.append(message)
        # If no messages left after filtering, skip this conversation
        if not filtered_messages:
            return None
            
        # Apply role transfer if configured
        if self.assistant_to_user or self.user_to_assistant:
            filtered_messages = apply_role_transfer(
                filtered_messages,
                self.assistant_to_user,
                self.user_to_assistant
            )
            
        # Return processed item with filtered messages
        processed_item = item.copy()
        processed_item["conversation"] = filtered_messages
        return processed_item 