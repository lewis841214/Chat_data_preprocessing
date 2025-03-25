#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Facebook platform handler for processing downloaded Facebook message data.
"""

import os
import json
import logging
import datetime
import codecs
import re
from typing import Dict, List, Any, Optional, Tuple
from tqdm import tqdm

from platform_handlers.base_handler import BasePlatformHandler


class ChineseTextJSONDecoder(json.JSONDecoder):
    """Custom JSON decoder with special handling for Chinese text."""
    
    def __init__(self, *args, **kwargs):
        json.JSONDecoder.__init__(self, *args, **kwargs)
        # Use the default JSONDecoder's scan_once
        self.scan_once = json.scanner.py_make_scanner(self)
        
    def decode(self, s, **kwargs):
        """Decode JSON string with special handling for Chinese characters."""
        result = super().decode(s, **kwargs)
        return self._decode_chinese_text(result)
    
    def _decode_chinese_text(self, obj):
        """Recursively decode Chinese text in JSON objects."""
        if isinstance(obj, str):
            # Try to fix common CJK encoding issues
            try:
                # Check for UTF-8 bytes misinterpreted as latin1/cp1252
                return obj.encode('latin1').decode('utf-8', errors='replace')
            except (UnicodeDecodeError, UnicodeEncodeError):
                return obj
        elif isinstance(obj, list):
            return [self._decode_chinese_text(item) for item in obj]
        elif isinstance(obj, dict):
            return {k: self._decode_chinese_text(v) for k, v in obj.items()}
        return obj


class FacebookHandler(BasePlatformHandler):
    """
    Handler for processing Facebook data downloaded from the "Download Your Information" feature.
    Processes message files from the messages/inbox directory structure.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the Facebook handler.
        
        Args:
            config: Configuration dictionary for this handler
        """
        super().__init__(config)
        # Get Facebook-specific config values
        self.platform = config.get("platform", "facebook")
        self.platform_data_path = config.get("platform_data_path", "data/this_profile's_activity_across_facebook/messages/inbox/")
        self.input_formatted_path = config.get("input_formated_path", "data/formated_data/")
        self.output_path = config.get("output_path", "data/output/")
        self.user_name = config.get("user_name", None)  # User's Facebook name to identify their messages
        
        # Set default encoding to UTF-8
        import sys
        if sys.stdout.encoding != 'utf-8':
            sys.stdout.reconfigure(encoding='utf-8')
        
        # Log the configuration for debugging
        self.logger.info(f"Initialized FacebookHandler with platform: {self.platform}")
        self.logger.info(f"Data path: {self.platform_data_path}")
        self.logger.info(f"Formatted data path: {self.input_formatted_path}")
        self.logger.info(f"Output path: {self.output_path}")
        
        if not os.path.isdir(self.platform_data_path):
            self.logger.warning(f"Facebook data path not found: {self.platform_data_path}")
        
        # Ensure output directories exist
        os.makedirs(self.input_formatted_path, exist_ok=True)
        os.makedirs(self.output_path, exist_ok=True)
    
    def load_data(self) -> List[Dict[str, Any]]:
        """
        Load Facebook message data from the inbox directory.
        
        Returns:
            List of conversation data with source file paths
        """
        conversations = []
        
        # Check if platform_data_path exists
        if not os.path.isdir(self.platform_data_path):
            self.logger.error(f"Facebook data path not found: {self.platform_data_path}")
            return []
            
        # Get all folder names first
        folder_names = [d for d in os.listdir(self.platform_data_path) 
                        if os.path.isdir(os.path.join(self.platform_data_path, d))]
        
        self.logger.info(f"Found {len(folder_names)} conversation folders to process")
        
        # Process folders with progress bar
        for folder_name in tqdm(folder_names, desc="Loading conversations", unit="folder"):
            folder_path = os.path.join(self.platform_data_path, folder_name)
            
            # Look for message files in each conversation folder
            message_files = [f for f in os.listdir(folder_path) 
                            if f.startswith("message_") and f.endswith(".json")]
            
            for message_file in message_files:
                file_path = os.path.join(folder_path, message_file)
                try:
                    # Use codecs to enforce UTF-8 and read raw content first
                    with codecs.open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                        
                    # Clean problematic control characters that cause JSON parsing errors
                    # Remove null bytes, escape controls, and other problematic characters
                    content = content.replace('\x00', '')  # Remove null bytes
                    
                    # Replace other control characters that can cause parsing errors
                    content = re.sub(r'[\x00-\x1F\x7F]', '', content)
                    
                    # Fix common Unicode escaping issues
                    content = re.sub(r'\\u([0-9a-fA-F]{4})', lambda m: chr(int(m.group(1), 16)), content)
                    
                    # Try to parse the JSON with our custom decoder
                    try:
                        conversation_data = json.loads(content, cls=ChineseTextJSONDecoder)
                    except json.JSONDecodeError as e:
                        # If regular parsing fails, use a more robust approach for problematic files
                        self.logger.warning(f"Standard JSON parsing failed for {file_path}, using fallback method")
                        
                        # More aggressive cleanup for problematic files
                        content = re.sub(r'[\x00-\x1F\x7F-\x9F]', '', content)  # Remove all control chars
                        content = re.sub(r'\\(?!["\\/bfnrt]|u[0-9a-fA-F]{4})', r'\\\\', content)  # Fix invalid escapes
                        
                        try:
                            conversation_data = json.loads(content, cls=ChineseTextJSONDecoder)
                        except json.JSONDecodeError as e2:
                            # If still fails, log and skip
                            self.logger.error(f"Failed to parse {file_path} even with fallback method: {str(e2)}")
                            continue
                        
                    # Store the original file path and folder for later reference
                    conversation_data["_source_file"] = file_path
                    conversation_data["_source_filename"] = message_file
                    conversation_data["_source_folder"] = folder_name
                    conversations.append(conversation_data)
                    
                except Exception as e:
                    self.logger.error(f"Error loading {file_path}: {str(e)}")
        
        self.logger.info(f"Loaded {len(conversations)} Facebook conversations")
        return conversations
    
    def transform(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Transform Facebook data to standardized format.
        
        Args:
            data: Facebook conversation data
            
        Returns:
            Data in standardized format
        """
        transformed_data = []
        
        # Use tqdm to show progress
        for conversation in tqdm(data, desc="Transforming conversations", unit="conv"):
            try:
                conversation_id = conversation.get("thread_path", "unknown")
                participants = conversation.get("participants", [])
                messages = conversation.get("messages", [])
                source_file = conversation.get("_source_file", "")
                source_filename = conversation.get("_source_filename", "")
                source_folder = conversation.get("_source_folder", "")
                
                if not messages:
                    self.logger.debug(f"Skipping conversation with no messages: {conversation_id}")
                    continue
                
                # Get participant names
                participant_names = [p.get("name", "Unknown") for p in participants]
                
                # Ensure participant names are properly decoded
                participant_names = [self._ensure_unicode(name) for name in participant_names]
                
                # Identify the user and other participants
                user_name = self._identify_user(participant_names)
                
                # Process the messages
                formatted_conversation = self._process_messages(messages, user_name, conversation_id, participant_names)
                
                if formatted_conversation:
                    # Use the original filename for the output
                    output_filename = source_filename
                    if not output_filename:
                        # Fallback to a generated name if original filename isn't available
                        output_filename = f"facebook_conversation_{conversation_id.replace('/', '_')}.json"
                    
                    # Create folder structure that mirrors the original
                    output_folder = os.path.join(self.input_formatted_path, source_folder)
                    os.makedirs(output_folder, exist_ok=True)
                    
                    # Keep the same filename but place it in the output directory with the same folder structure
                    output_file = os.path.join(output_folder, output_filename)
                    
                    try:
                        # Custom JSON serialization to ensure proper Unicode handling
                        json_str = json.dumps(formatted_conversation, ensure_ascii=False, indent=2)
                        
                        # Write with explicit UTF-8 encoding
                        with open(output_file, 'wb') as f:
                            f.write(json_str.encode('utf-8'))
                        
                        transformed_data.append(formatted_conversation)
                    except Exception as e:
                        self.logger.error(f"Error saving conversation to {output_file}: {str(e)}")
                
            except Exception as e:
                self.logger.error(f"Error transforming conversation: {str(e)}")
                import traceback
                self.logger.error(traceback.format_exc())
                continue
        
        self.logger.info(f"Transformed {len(transformed_data)} Facebook conversations")
        return transformed_data
    
    def _ensure_unicode(self, text):
        """
        Ensure text is properly decoded as unicode.
        
        Args:
            text: Text to decode
            
        Returns:
            Properly decoded unicode text
        """
        if text is None:
            return ""
            
        if isinstance(text, bytes):
            # Try different encodings common for CJK characters
            for encoding in ['utf-8', 'cp936', 'gbk', 'gb2312', 'big5']:
                try:
                    return text.decode(encoding, errors='replace')
                except UnicodeDecodeError:
                    continue
            # Default fallback
            return text.decode('utf-8', errors='replace')
            
        # For string objects, check if they need fixing
        if isinstance(text, str):
            # Check for double-encoded UTF-8
            try:
                # Sometimes text is double-encoded, try to fix that
                encoded = text.encode('latin1')
                return encoded.decode('utf-8', errors='replace')
            except (UnicodeDecodeError, UnicodeEncodeError):
                pass
                
        return text
    
    def _identify_user(self, participant_names: List[str]) -> str:
        """
        Identify the user's name from the list of participants.
        
        Args:
            participant_names: List of participant names
            
        Returns:
            The identified user name
        """
        # If user_name is provided in config, use it - this is the most reliable method
        if self.user_name:
            # Exact match
            if self.user_name in participant_names:
                self.logger.debug(f"User identified by exact config match: {self.user_name}")
                return self.user_name
                
            # Check for partial matches (common with display name variations)
            for name in participant_names:
                if self.user_name in name or name in self.user_name:
                    self.logger.debug(f"User identified by partial config match: {name} (config: {self.user_name})")
                    return name
        
        # The Facebook data is usually from the perspective of the account owner
        # So the owner is often referenced in special ways across conversations
        # We'll try some heuristics to identify the account owner
        
        # Heuristic 1: In Facebook data exports, the account owner's name appears consistently 
        # across all conversations. If we have participant frequency data from previous calls,
        # use that to identify the most common participant.
        if hasattr(self, '_participant_frequency'):
            most_common = max(self._participant_frequency.items(), key=lambda x: x[1])[0]
            self.logger.debug(f"User identified by frequency: {most_common}")
            return most_common
        
        # Heuristic 2: For Facebook group chats with many participants,
        # the account owner can be among any of the participants
        if len(participant_names) > 2:
            # Fallback to user_name from config if provided, even if not in participants
            # (might be a nickname or alternate form)
            if self.user_name:
                self.logger.debug(f"Using configured user name despite not matching: {self.user_name}")
                return self.user_name
            else:
                # In a group chat without a configured name, it's hard to identify the user
                # Just use the first participant and log a warning
                self.logger.warning(f"Group chat with {len(participant_names)} participants but no configured user_name. Using {participant_names[0]} as User.")
                return participant_names[0]
                
        # Heuristic 3: For one-on-one conversations with two participants,
        # we need to determine which one is the account owner
        if len(participant_names) == 2:
            # In a one-on-one chat export from Facebook, the first participant is often
            # the conversation partner, and the second is the account owner
            user = participant_names[1]
            self.logger.debug(f"User identified as second participant in 1:1 chat: {user}")
            return user
        
        # Fallback to first participant if only one exists (unlikely in Facebook data)
        self.logger.warning(f"Could not confidently identify user from {participant_names}. Using {participant_names[0]}")
        return participant_names[0] if participant_names else "Unknown"
    
    def _process_messages(self, messages: List[Dict[str, Any]], user_name: str, 
                         conversation_id: str, participant_names: List[str]) -> Dict[str, Any]:
        """
        Process messages in a conversation and convert to standardized format.
        
        Args:
            messages: List of Facebook messages
            user_name: The name of the Facebook account owner as identified from config
            conversation_id: Conversation identifier
            participant_names: List of participant names
            
        Returns:
            Standardized conversation data
        """
        conversation = []
        
        # Sort messages by timestamp (oldest first)
        sorted_messages = sorted(messages, key=lambda m: m.get("timestamp_ms", 0))
        
        # Track participant frequency across conversations to help with user identification
        if not hasattr(self, '_participant_frequency'):
            self._participant_frequency = {}
            
        # Count message senders to help identify the user in future conversations
        sender_counts = {}
        
        # Process each message
        for message in sorted_messages:
            sender_name = message.get("sender_name", "Unknown")
            content = message.get("content", "")
            timestamp_ms = message.get("timestamp_ms", 0)
            
            # Skip empty messages
            if not content:
                continue
            
            # Update sender frequency counts
            sender_counts[sender_name] = sender_counts.get(sender_name, 0) + 1
            
            # Ensure content is properly decoded
            content = self._ensure_unicode(content)
            sender_name = self._ensure_unicode(sender_name)
            
            # Additional processing for common CJK encoding issues
            if re.search(r'[\u4e00-\u9fff\u3400-\u4dbf\u20000-\u2a6df\u2a700-\u2b73f\u2b740-\u2b81f\u2b820-\u2ceaf\uf900-\ufaff\u2f800-\u2fa1f]', content):
                # This contains CJK characters, apply special handling
                self.logger.debug(f"Handling CJK content: {content[:20]}...")
                
                # Try to fix common issues with CJK characters
                # 1. Replace known problematic character sequences
                content = re.sub(r'\\u([0-9a-fA-F]{4})', lambda m: chr(int(m.group(1), 16)), content)
                
                # 2. Replace Unicode escape sequences with actual characters
                content = re.sub(r'&#([0-9]+);', lambda m: chr(int(m.group(1))), content)
            
            # Determine role based on exact rule:
            # If sender name matches the user_name from config, then "Assistant", otherwise "User"
            if sender_name == self.user_name:
                role = "Assistant"  # Config's user_name is Assistant
                self.logger.debug(f"Role 'Assistant' assigned to: {sender_name} (matches config user_name)")
            else:
                role = "User"  # Everyone else is User
                self.logger.debug(f"Role 'User' assigned to: {sender_name} (does not match config user_name: {self.user_name})")
            
            # Create message entry
            message_entry = {
                "role": role,
                "content": content
            }
            
            conversation.append(message_entry)
        
        # Update global participant frequency counters
        for sender, count in sender_counts.items():
            self._participant_frequency[sender] = self._participant_frequency.get(sender, 0) + count
        
        # Create the formatted conversation object
        timestamp = sorted_messages[0].get("timestamp_ms", 0) / 1000 if sorted_messages else 0
        last_timestamp = sorted_messages[-1].get("timestamp_ms", 0) / 1000 if sorted_messages else 0
        
        formatted_conversation = {
            "platform": self.platform,  # Use the platform from config instead of hardcoding
            "conversation_id": conversation_id,
            "participants": participant_names,
            "created_at": datetime.datetime.fromtimestamp(timestamp).isoformat(),
            "last_message_at": datetime.datetime.fromtimestamp(last_timestamp).isoformat(),
            "conversation": conversation
        }
        
        return formatted_conversation
    
    def process(self) -> List[Dict[str, Any]]:
        """
        Process Facebook data and return in standardized format.
        This method is called by the main pipeline.
        
        Returns:
            Processed data in standardized format
        """
        self.logger.info("Starting Facebook data processing")
        
        # Set UTF-8 encoding for stdout
        import sys
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
        os.environ["PYTHONIOENCODING"] = "utf-8"
        
        try:
            # Load data from Facebook files
            conversations = self.load_data()
            self.logger.info(f"Loaded {len(conversations)} conversations")
            
            # Transform to standardized format
            transformed_conversations = self.transform(conversations)
            self.logger.info(f"Transformed {len(transformed_conversations)} conversations")
            
            # Calculate statistics
            num_messages = sum(len(conv.get("conversation", [])) for conv in transformed_conversations)
            num_users = len(set(p for conv in transformed_conversations for p in conv.get("participants", [])))
            
            self.logger.info(f"Processing summary:")
            self.logger.info(f"- Total conversations: {len(transformed_conversations)}")
            self.logger.info(f"- Total messages: {num_messages}")
            self.logger.info(f"- Total unique participants: {num_users}")
            
            return transformed_conversations
            
        except Exception as e:
            self.logger.error(f"Error in Facebook data processing: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return [] 