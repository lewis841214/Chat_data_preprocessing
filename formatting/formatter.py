#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Text formatter for chat data normalization.
"""

import re
import unicodedata
import string
from typing import Dict, List, Any
from tqdm import tqdm
import html
import ftfy
from datetime import datetime

from processor import BaseProcessor

class Formatter(BaseProcessor):
    """
    Formatter for text normalization in chat data.
    Handles punctuation, HTML cleaning, and Unicode fixes.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the formatter.
        
        Args:
            config: Configuration dictionary
        """
        super().__init__(config)
        self.logger.info("Initializing text formatter")
        
        # Set formatting options from config
        self.remove_punctuation = config.get("remove_punctuation", False)
        self.clean_html = config.get("clean_html", True)
        self.fix_unicode = config.get("fix_unicode", True)
        
        # Additional optional formatting settings
        self.normalize_whitespace = config.get("normalize_whitespace", True)
        self.lowercase = config.get("lowercase", False)
        self.replace_urls = config.get("replace_urls", False)
        self.url_replacement = config.get("url_replacement", "[URL]")
        self.replace_emails = config.get("replace_emails", False)
        self.email_replacement = config.get("email_replacement", "[EMAIL]")
        self.replace_numbers = config.get("replace_numbers", False)
        self.number_replacement = config.get("number_replacement", "[NUMBER]")
        
        # Date handling options - either normalize or replace
        self.normalize_dates = config.get("normalize_dates", True)
        self.date_format = config.get("date_format", "MM/DD")  # Output date format
        self.replace_dates = config.get("replace_dates", False)
        self.date_replacement = config.get("date_replacement", "[DATE]")
        
        # If both normalize_dates and replace_dates are True, replace takes precedence
        if self.normalize_dates and self.replace_dates:
            self.logger.warning("Both normalize_dates and replace_dates are enabled. replace_dates will take precedence.")
            self.normalize_dates = False
        
        # Compile regex patterns
        self.html_tag_pattern = re.compile(r'<[^>]*>')
        self.whitespace_pattern = re.compile(r'\s+')
        self.url_pattern = re.compile(r'https?://\S+|www\.\S+')
        self.email_pattern = re.compile(r'\S+@\S+\.\S+')
        self.number_pattern = re.compile(r'\d+(?:\.\d+)?')
        
        # Date patterns for detection and replacement
        # Common date formats for Chinese/English contexts
        self.date_patterns = [
            # M/D~D (month/day~day) - e.g., 2/3~28
            re.compile(r'(\d{1,2})/(\d{1,2})~(\d{1,2})'),
            # Short form date ranges without repeating month: e.g., 2/20~21
            re.compile(r'(\d{1,2})/(\d{1,2})[-~](\d{1,2})(?!/|-)'),
            # M/D (month/day) - e.g., 2/15
            re.compile(r'(\d{1,2})/(\d{1,2})'),
            # M-D (month-day) - e.g., 12-25
            re.compile(r'(\d{1,2})-(\d{1,2})'),
            # D~D with context - e.g., 11~28 in a date context
            re.compile(r'(\d{1,2})~(\d{1,2})'),
            # Y/M/D (year/month/day) - e.g., 2023/12/25
            re.compile(r'(20\d{2})/(\d{1,2})/(\d{1,2})'),
            # Y-M-D (year-month-day) - e.g., 2023-12-25
            re.compile(r'(20\d{2})-(\d{1,2})-(\d{1,2})')
        ]
        
        # Punctuation translator for removal
        self.punctuation_translator = str.maketrans('', '', string.punctuation)
        
        self.logger.info(f"Formatter configured with: clean_html={self.clean_html}, "
                        f"fix_unicode={self.fix_unicode}, remove_punctuation={self.remove_punctuation}, "
                        f"normalize_dates={self.normalize_dates}, replace_dates={self.replace_dates}")
        
    def clean_html_tags(self, text: str) -> str:
        """
        Remove HTML tags and decode HTML entities.
        
        Args:
            text: Input text
            
        Returns:
            Text with HTML tags removed and entities decoded
        """
        # First decode HTML entities
        text = html.unescape(text)
        # Then remove HTML tags
        return self.html_tag_pattern.sub('', text)
    
    def normalize_unicode(self, text: str) -> str:
        """
        Fix Unicode issues using ftfy and normalize to NFC.
        
        Args:
            text: Input text
            
        Returns:
            Text with normalized Unicode
        """
        # Use ftfy to fix various Unicode issues
        text = ftfy.fix_text(text)
        # Normalize to NFC form (canonical composition)
        return unicodedata.normalize('NFC', text)
    
    def replace_date_patterns(self, text: str) -> str:
        """
        Replace date patterns with a placeholder token.
        
        Args:
            text: Input text with dates
            
        Returns:
            Text with dates replaced by placeholder
        """
        if not text:
            return text
            
        # Process the text line by line
        lines = text.split('\n')
        processed_lines = []
        
        for line in lines:
            # Replace all date patterns with the placeholder
            for pattern in self.date_patterns:
                line = pattern.sub(self.date_replacement, line)
            
            # Additional pattern for formats like "2/20~21" (short-form date ranges)
            line = re.sub(r'(\d{1,2})/(\d{1,2})[-~](\d{1,2})', self.date_replacement, line)
            
            # Also handle isolated numbers like "12", "14", "15" that appear between dates
            # This is tricky because we don't want to replace all numbers, just those in date context
            if self.date_replacement in line:
                # Replace consecutive single or double digit numbers in date contexts
                # For example: "中壢:2/7、12、14、15、2/17~21"
                line = re.sub(
                    f'{self.date_replacement}[、,，] ?(\\d{{1,2}})[、,，]',
                    f'{self.date_replacement}、{self.date_replacement}、',
                    line
                )
                # Replace the last number in a sequence
                line = re.sub(
                    f'{self.date_replacement}[、,，] ?(\\d{{1,2}})',
                    f'{self.date_replacement}、{self.date_replacement}',
                    line
                )
                
            # Also look for Chinese/Japanese date separators and replace date-like contexts
            # This handles cases like "2/3、2/15" (dates separated by、)
            if '、' in line and re.search(r'\d{1,2}/\d{1,2}', line):
                parts = []
                current = ""
                # Split by the delimiter but preserve it
                for char in line:
                    if char == '、':
                        if current and re.search(r'\d{1,2}/\d{1,2}', current):
                            parts.append(self.date_replacement)
                        else:
                            parts.append(current)
                        current = ""
                        parts.append(char)
                    else:
                        current += char
                
                if current:
                    if re.search(r'\d{1,2}/\d{1,2}', current):
                        parts.append(self.date_replacement)
                    else:
                        parts.append(current)
                
                line = "".join(parts)
            
            processed_lines.append(line)
            
        return '\n'.join(processed_lines)
    
    def normalize_date_text(self, text: str) -> str:
        """
        Normalize date formats in the text.
        
        Args:
            text: Input text with dates like "2/3~28" or "2/2~9、11~28"
            
        Returns:
            Text with normalized date formats
        """
        if not text:
            return text
        
        # Process the text line by line to maintain context
        lines = text.split('\n')
        normalized_lines = []
        
        for line in lines:
            current_month = None
            
            # First, try to extract the current month from any full date in the line
            month_match = re.search(r'(\d{1,2})/', line)
            if month_match:
                current_month = month_match.group(1)
            
            # Process full date ranges (M/D~D)
            if current_month:
                # Handle date ranges with explicit month/day format (M/D~D)
                line = re.sub(
                    r'(\d{1,2})/(\d{1,2})~(\d{1,2})',
                    lambda m: self._format_date_range(m.group(1), m.group(2), m.group(3)),
                    line
                )
                
                # Handle short-form date ranges (like 2/20~21) without repeating month
                line = re.sub(
                    r'(\d{1,2})/(\d{1,2})[-~](\d{1,2})(?!/|-)',
                    lambda m: self._format_date_range(m.group(1), m.group(2), m.group(3)),
                    line
                )
            
                # Now process parts with implicit month information
                # Look for patterns like "、11~28" or "，11~28"
                # We need to preserve the delimiter
                
                # First, we'll split the text, but keep track of the delimiters
                parts = []
                current_part = ""
                i = 0
                while i < len(line):
                    if line[i] in ['、', '，', ',']:
                        if current_part:
                            parts.append(current_part)
                            current_part = ""
                        parts.append(line[i])  # Keep the delimiter as a separate part
                    else:
                        current_part += line[i]
                    i += 1
                if current_part:
                    parts.append(current_part)
                
                # Process each part
                for i in range(len(parts)):
                    # Skip delimiters
                    if parts[i] in ['、', '，', ',']:
                        continue
                    
                    # Process patterns like "11~28"
                    if re.search(r'^\s*(\d{1,2})~(\d{1,2})\s*$', parts[i]):
                        match = re.search(r'^\s*(\d{1,2})~(\d{1,2})\s*$', parts[i])
                        if match:
                            start_day = match.group(1)
                            end_day = match.group(2)
                            parts[i] = self._format_date_range(current_month, start_day, end_day)
                
                # Rejoin the parts, preserving delimiters
                line = "".join(parts)
            
            normalized_lines.append(line)
        
        return '\n'.join(normalized_lines)
    
    def _format_date_range(self, month: str, day_start: str, day_end: str) -> str:
        """
        Format a date range according to the configured date format.
        
        Args:
            month: Month number
            day_start: Starting day
            day_end: Ending day
            
        Returns:
            Formatted date range string
        """
        if self.date_format == "MM/DD":
            return f"{month}/{day_start} to {month}/{day_end}"
        elif self.date_format == "MM-DD":
            return f"{month}-{day_start} to {month}-{day_end}"
        else:  # Default to more verbose format
            return f"from {month}/{day_start} to {month}/{day_end}"
    
    def format_text(self, text: str) -> str:
        """
        Apply all configured formatting operations to a text.
        
        Args:
            text: Input text
            
        Returns:
            Formatted text
        """
        if not text:
            return text
        
        # Fix Unicode if enabled
        if self.fix_unicode:
            text = self.normalize_unicode(text)
        
        # Clean HTML if enabled
        if self.clean_html:
            text = self.clean_html_tags(text)
        
        # Handle date formatting - replace takes precedence over normalize
        if self.replace_dates:
            text = self.replace_date_patterns(text)
        elif self.normalize_dates:
            text = self.normalize_date_text(text)
        
        # Replace URLs if enabled
        if self.replace_urls:
            text = self.url_pattern.sub(self.url_replacement, text)
        
        # Replace emails if enabled
        if self.replace_emails:
            text = self.email_pattern.sub(self.email_replacement, text)
        
        # Replace numbers if enabled (but only if we're not normalizing dates,
        # as that would interfere with the date numbers)
        if self.replace_numbers and not self.normalize_dates and not self.replace_dates:
            text = self.number_pattern.sub(self.number_replacement, text)
        
        # Remove punctuation if enabled
        if self.remove_punctuation:
            text = text.translate(self.punctuation_translator)
        
        # Normalize whitespace if enabled
        if self.normalize_whitespace:
            text = self.whitespace_pattern.sub(' ', text).strip()
        
        # Convert to lowercase if enabled
        if self.lowercase:
            text = text.lower()
        
        return text
    
    def process(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Apply text formatting to the input data.
        
        Args:
            data: Input data to format
            
        Returns:
            Formatted data
        """
        self.logger.info(f"Applying text formatting to {len(data)} items")
        
        formatted_data = []
        format_counts = {
            "html_cleaned": 0,
            "unicode_fixed": 0,
            "punctuation_removed": 0,
            "dates_normalized": 0,
            "dates_replaced": 0
        }
        
        for item in tqdm(data, desc="Formatting text"):
            # Make a copy to avoid modifying original
            formatted_item = item.copy()
            
            # Handle conversation messages (the actual structure of the data)
            if "conversation" in item and isinstance(item["conversation"], list):
                formatted_conversations = []
                for message in item["conversation"]:
                    # Copy the message to avoid modifying original
                    formatted_message = message.copy()
                    
                    # Format message content if present
                    message_content = message.get("content", "")
                    if message_content:
                        # Save original for comparison
                        original_message_content = message_content
                        
                        # Apply formatting
                        formatted_message_content = self.format_text(message_content)
                        formatted_message["content"] = formatted_message_content
                        
                        # Track changes for logging
                        if self.clean_html and original_message_content != formatted_message_content:
                            format_counts["html_cleaned"] += 1
                        if self.fix_unicode and original_message_content != formatted_message_content:
                            format_counts["unicode_fixed"] += 1
                        if self.remove_punctuation and original_message_content != formatted_message_content:
                            format_counts["punctuation_removed"] += 1
                        if self.normalize_dates and original_message_content != formatted_message_content:
                            format_counts["dates_normalized"] += 1
                        if self.replace_dates and original_message_content != formatted_message_content:
                            format_counts["dates_replaced"] += 1
                    
                    formatted_conversations.append(formatted_message)
                
                # Update the conversation list with formatted messages
                formatted_item["conversation"] = formatted_conversations
            
            # Format other text fields if needed
            # Example: if your data has additional text fields like "title", "summary", etc.
            for text_field in ["title", "summary", "description"]:
                if text_field in item and item[text_field]:
                    formatted_item[text_field] = self.format_text(item[text_field])
            
            formatted_data.append(formatted_item)
        
        # Log formatting statistics
        log_msg = (f"Text formatting complete: {format_counts['html_cleaned']} items had HTML cleaned, "
                  f"{format_counts['unicode_fixed']} had Unicode fixed, "
                  f"{format_counts['punctuation_removed']} had punctuation removed")
                  
        if self.normalize_dates:
            log_msg += f", {format_counts['dates_normalized']} had dates normalized"
        elif self.replace_dates:
            log_msg += f", {format_counts['dates_replaced']} had dates replaced"
            
        self.logger.info(log_msg)
        return formatted_data 