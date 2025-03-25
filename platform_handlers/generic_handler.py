#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Generic platform handler for processing data without platform-specific logic.
"""

from typing import Dict, List, Any, Optional

from platform_handlers.base_handler import BasePlatformHandler


class GenericHandler(BasePlatformHandler):
    """
    Generic handler for data that doesn't require platform-specific processing.
    Attempts to normalize various input formats to the standard conversation format.
    """
    
    def transform(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Transform generic data to standardized format.
        
        Args:
            data: Input data in various formats
            
        Returns:
            Data in standardized format
        """
        transformed_data = []
        
        for item in data:
            try:
                # Check if data is already in the correct format
                if self._validate_conversation(item):
                    transformed_data.append(item)
                    continue
                
                # Check common formats and convert
                if "messages" in item and isinstance(item["messages"], list):
                    # Convert messages format
                    transformed = self._transform_messages_format(item)
                    if transformed:
                        transformed_data.append(transformed)
                        
                elif "dialog" in item and isinstance(item["dialog"], list):
                    # Convert dialog format
                    transformed = self._transform_dialog_format(item)
                    if transformed:
                        transformed_data.append(transformed)
                        
                elif "text" in item and "user" in item:
                    # Convert simple QA format
                    transformed = self._transform_qa_format(item)
                    if transformed:
                        transformed_data.append(transformed)
                        
                else:
                    # Try best-effort transformation
                    transformed = self._transform_best_effort(item)
                    if transformed:
                        transformed_data.append(transformed)
                        
            except Exception as e:
                self.logger.error(f"Error transforming item: {str(e)}")
                continue
                
        self.logger.info(f"Transformed {len(transformed_data)} of {len(data)} items")
        return transformed_data
    
    def _transform_messages_format(self, item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Transform data in "messages" format.
        
        Args:
            item: Input data with "messages" field
            
        Returns:
            Transformed data or None if transformation failed
        """
        messages = item.get("messages", [])
        if not messages:
            return None
            
        conversation = []
        for msg in messages:
            if "role" in msg and "content" in msg:
                # Already in the right format
                conversation.append({
                    "role": msg["role"],
                    "content": msg["content"],
                    "do_train": True if "do_train" not in msg else msg["do_train"]
                })
            elif "sender" in msg and "text" in msg:
                # Convert sender/text format
                conversation.append({
                    "role": self._normalize_role(msg["sender"]),
                    "content": msg["text"],
                    "do_train": True
                })
                
        if not conversation:
            return None
            
        return {"conversation": conversation}
    
    def _transform_dialog_format(self, item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Transform data in "dialog" format.
        
        Args:
            item: Input data with "dialog" field
            
        Returns:
            Transformed data or None if transformation failed
        """
        dialog = item.get("dialog", [])
        if not dialog:
            return None
            
        conversation = []
        for turn in dialog:
            if isinstance(turn, dict):
                if "speaker" in turn and "text" in turn:
                    conversation.append({
                        "role": self._normalize_role(turn["speaker"]),
                        "content": turn["text"],
                        "do_train": True
                    })
                elif "role" in turn and "content" in turn:
                    conversation.append({
                        "role": turn["role"],
                        "content": turn["content"],
                        "do_train": True if "do_train" not in turn else turn["do_train"]
                    })
            elif isinstance(turn, str):
                # Alternate user/assistant roles for strings
                role = "User" if len(conversation) % 2 == 0 else "Assistant"
                conversation.append({
                    "role": role,
                    "content": turn,
                    "do_train": role == "Assistant"
                })
                
        if not conversation:
            return None
            
        return {"conversation": conversation}
    
    def _transform_qa_format(self, item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Transform data in simple QA format.
        
        Args:
            item: Input data with question/answer fields
            
        Returns:
            Transformed data or None if transformation failed
        """
        # Look for common QA field names
        question = item.get("text") or item.get("question") or item.get("query") or item.get("input")
        answer = item.get("response") or item.get("answer") or item.get("output") or item.get("completion")
        
        if not question or not answer:
            return None
            
        conversation = [
            {
                "role": "User",
                "content": question,
                "do_train": False
            },
            {
                "role": "Assistant",
                "content": answer,
                "do_train": True
            }
        ]
        
        return {"conversation": conversation}
    
    def _transform_best_effort(self, item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Make a best-effort attempt to transform unknown formats.
        
        Args:
            item: Input data in unknown format
            
        Returns:
            Transformed data or None if transformation failed
        """
        # Try to find text fields that could be messages
        conversation = []
        
        # If there's a query/response pattern
        for q_field in ["query", "question", "input", "prompt"]:
            if q_field in item and isinstance(item[q_field], str):
                conversation.append({
                    "role": "User",
                    "content": item[q_field],
                    "do_train": False
                })
                break
                
        for a_field in ["response", "answer", "output", "completion", "assistant"]:
            if a_field in item and isinstance(item[a_field], str):
                conversation.append({
                    "role": "Assistant",
                    "content": item[a_field],
                    "do_train": True
                })
                break
        
        # If we couldn't find a conversation structure, give up
        if len(conversation) < 2:
            return None
            
        return {"conversation": conversation}
    
    def _normalize_role(self, role: str) -> str:
        """
        Normalize role names to standard format.
        
        Args:
            role: Input role name
            
        Returns:
            Normalized role name
        """
        role = role.lower().strip()
        
        if role in ["user", "human", "client", "customer", "questioner"]:
            return "User"
        elif role in ["assistant", "ai", "bot", "chatbot", "model", "system", "answerer"]:
            return "Assistant"
        else:
            # Default to original role with first letter capitalized
            return role[0].upper() + role[1:] if role else "User" 