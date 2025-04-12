#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Role transfer functionality for conversation data.
"""

from typing import Dict, List, Any

def apply_role_transfer(messages: List[Dict[str, Any]], 
                       assistant_to_user: List[str],
                       user_to_assistant: List[str]) -> List[Dict[str, Any]]:
    """
    Apply role transfer to messages based on content patterns.
    
    Args:
        messages: List of message dictionaries
        assistant_to_user: List of patterns that trigger assistant to user transfer
        user_to_assistant: List of patterns that trigger user to assistant transfer
        
    Returns:
        List of messages with updated roles
    """
    processed_messages = []
    current_role = None
    
    for message in messages:
        content = message.get("content", "")
        role = message.get("role", "")
        
        # Skip empty messages
        if not content:
            continue
        # Apply role transfer if configured
        if role == "Assistant" and any(pattern in content for pattern in assistant_to_user):
            current_role = "User"
        elif role == "User":
            if  user_to_assistant != None:
                if any(pattern in content for pattern in user_to_assistant):
                    current_role = "Assistant"
        else:
            current_role = role
        # Create new message with potentially updated role
        processed_message = message.copy()
        processed_message["role"] = current_role
        processed_messages.append(processed_message)
        
    return processed_messages 