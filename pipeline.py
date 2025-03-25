#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Main pipeline orchestration script for chat data preprocessing.
This script manages the entire flow from raw data to processed corpus.
"""

import os
import argparse
import logging
import json
import yaml  # Added for debugging
import codecs  # Added for UTF-8 handling
from typing import Dict, List, Any, Optional
from tqdm import tqdm

from config_handler import ConfigHandler
from platform_handlers.handler_factory import PlatformHandlerFactory
from cleaning.cleaner import Cleaner
from filtering.filter_manager import FilterManager
from formatting.formatter import Formatter
from deduplication.dedup_manager import DedupManager


class Pipeline:
    """
    Main pipeline class for orchestrating the chat data preprocessing flow.
    """
    
    def __init__(self, config_path: str):
        """
        Initialize the pipeline with configuration.
        
        Args:
            config_path: Path to the configuration file
        """
        self.logger = self._setup_logging()
        self.config = ConfigHandler(config_path).get_config()
        self.logger.info(f"Initialized pipeline with config from {config_path}")
        
        # Debug: Print the platform configuration
        if "platform" in self.config:
            self.logger.info(f"Platform config: {self.config['platform']}")
            self.logger.info(f"Platform type: {self.config['platform'].get('type', 'Not specified')}")
        else:
            self.logger.warning("No platform section found in config")
        
        # Initialize components
        self.platform_handler = None
        if self.is_platform_configured():
            self.platform_handler = PlatformHandlerFactory.get_handler(self.config)
            self.logger.info(f"Using platform handler: {self.platform_handler.__class__.__name__}")
        else:
            self.logger.info("Platform not configured, will load from formatted data path")
        
        self.cleaner = Cleaner(self.config["cleaning"])
        self.filter_manager = FilterManager(self.config["filtering"])
        self.formatter = Formatter(self.config["formatting"])
        self.dedup_manager = DedupManager(self.config["deduplication"])
        
    def _setup_logging(self) -> logging.Logger:
        """Set up logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler("pipeline.log", encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
        return logging.getLogger("ChatPipeline")
    
    def is_platform_configured(self) -> bool:
        """
        Check if platform is properly configured.
        
        Returns:
            bool: True if platform is configured, False otherwise
        """
        return (
            "platform" in self.config 
            and self.config["platform"].get("type") is not None
            and self.config["platform"].get("platform_data_path") is not None
        )
    
    def load_formatted_data(self) -> List[Dict[str, Any]]:
        """
        Load data directly from the formatted data path.
        
        Returns:
            List[Dict[str, Any]]: The loaded formatted data
        """
        if "platform" not in self.config or "input_formated_path" not in self.config["platform"]:
            self.logger.error("No input_formated_path specified in config")
            return []
            
        input_path = self.config["platform"]["input_formated_path"]
        self.logger.info(f"Loading data from formatted input path: {input_path}")
        
        conversation_data = []
        
        if not os.path.exists(input_path):
            self.logger.error(f"Formatted input path does not exist: {input_path}")
            return []
            
        try:
            if os.path.isdir(input_path):
                # If it's a directory, recursively search for all JSON files
                for root, dirs, files in tqdm(os.walk(input_path)):
                    for filename in files:
                        if filename.endswith('.json'):
                            file_path = os.path.join(root, filename)
                            self.logger.debug(f"Loading conversation data from: {file_path}")
                            try:
                                with open(file_path, 'r', encoding='utf-8') as f:
                                    file_data = json.load(f)
                                    if isinstance(file_data, list):
                                        conversation_data.extend(file_data)
                                    else:
                                        conversation_data.append(file_data)
                            except Exception as file_e:
                                self.logger.warning(f"Error loading file {file_path}: {str(file_e)}")
            else:
                # If it's a file, load it directly
                with open(input_path, 'r', encoding='utf-8') as f:
                    conversation_data = json.load(f)
            self.logger.info(f"Loaded {len(conversation_data)} conversations from formatted input")
            return conversation_data
            
        except Exception as e:
            self.logger.error(f"Error loading formatted conversation data: {str(e)}")
            return []

    def _generate_final_corpus(self, data: List[Dict[str, Any]], type: str = "conversation") -> None:
        """
        Generate the final corpus in the required JSON format.
        
        Args:
            conversation_data: Processed conversation data to be saved
        """
        output_path = self.config["output"][f"{type}_path"]
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Use proper UTF-8 encoding when writing the output file
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        self.logger.info(f"Final corpus saved to {output_path}")
        self.logger.info(f"Total {type}: {len(data)}")
    
    def run(self) -> None:
        """
        Execute the full pipeline process.
        """
        self.logger.info("Starting pipeline execution")
        
        # Step 1: Get conversation data - either from platform handler or formatted input
        if self.is_platform_configured() and self.platform_handler is not None:
            self.logger.info("Step 1: Platform-specific conversation data handling")
            conversation_data = self.platform_handler.process()
        else:
            self.logger.info("Step 1: Loading from formatted data path (skipping platform processing)")
            conversation_data = self.load_formatted_data()
            
        if not conversation_data:
            self.logger.error("No conversation data loaded. Pipeline execution failed.")
            return
        
        if self.config["mode"] == "testing":
            conversation_data = conversation_data[:500]

        # Step 2: Clean conversation data (remove useless content)
        self.logger.info("Step 2: Cleaning - removing useless content from conversations")
        cleaned_conversations = self.cleaner.process(conversation_data)
        
        # Step 3: Quality filtering
        self.logger.info("Step 3: Quality filtering of conversations")
        filtered_conversations = self.filter_manager.process(cleaned_conversations)
        
        # Step 4: Text formatting
        self.logger.info("Step 4: Text formatting for conversations")
        formatted_conversations = self.formatter.process(filtered_conversations)
        
        # Generate question-answer pairs (as a separate process that doesn't affect the main flow)
        self.logger.info("Generating question-answer pairs from conversations")
        qa_pairs = self._conversations_to_qa_pairs(formatted_conversations)
        self.logger.info(f"Generated {len(qa_pairs)} question-answer pairs from {len(formatted_conversations)} conversations")
    
        # Save QA pairs to a separate output file if configured
        qa_output_path = self.config["output"].get("qa_path")
        if qa_output_path:
            os.makedirs(os.path.dirname(qa_output_path), exist_ok=True)
            with open(qa_output_path, 'w', encoding='utf-8') as f:
                json.dump(qa_pairs, f, ensure_ascii=False, indent=2)
            self.logger.info(f"QA pairs saved to {qa_output_path}")

        
        # Step 5: Deduplication (optional based on config)
        if self.config["deduplication"]["enabled"]:
            self.logger.info("Step 5: Deduplication of conversations")
            deduplicated_conversations = self.dedup_manager.process(formatted_conversations, key="conversation")
            deduplicated_qa_pairs = self.dedup_manager.process(qa_pairs, key="question")
        else:
            self.logger.info("Step 5: Conversation deduplication skipped")
            deduplicated_conversations = formatted_conversations
            deduplicated_qa_pairs = qa_pairs

        
        # Step 6: Generate final corpus
        self.logger.info("Step 6: Generating final conversation corpus")
        self._generate_final_corpus(deduplicated_conversations, type="conversation")
        self._generate_final_corpus(deduplicated_qa_pairs, type="qa")
        
        self.logger.info("Conversation pipeline execution completed successfully")
    
    def _conversations_to_qa_pairs(self, conversations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Transform conversations into question-answer pairs.
        
        Rules:
        1. The "User" messages are treated as questions
        2. All subsequent "Assistant" messages until the next "User" message are concatenated as the answer
        
        Args:
            conversations: List of conversation data
        
        Returns:
            List[Dict[str, Any]]: Question-answer pairs derived from conversations
        """
        qa_pairs = []
        
        for conversation in conversations:
            if "conversation" not in conversation:
                self.logger.warning(f"Conversation missing 'messages' field: {conversation}")
                continue
                
            messages = conversation.get("conversation", [])
            current_question = None
            current_answers = []
            
            for message in messages:
                role = message.get("role")
                content = message.get("content", "")
                
                if role == "User":
                    # If we have a previous question and collected answers, save the QA pair
                    if current_question and current_answers:
                        qa_pairs.append({
                            "question": current_question,
                            "answer": " ".join(current_answers),
                            "conversation_id": conversation.get("conversation_id", ""),
                            "metadata": conversation.get("metadata", {})
                        })
                    
                    # Start a new QA pair
                    current_question = content
                    current_answers = []
                
                elif role == "Assistant" and current_question:
                    # Add this answer to the current collection
                    if content.strip():  # Only add non-empty content
                        current_answers.append(content + '\n')
            
            # Don't forget the last QA pair in the conversation
            if current_question and current_answers:
                qa_pairs.append({
                    "question": current_question,
                    "answer": " ".join(current_answers),
                    "conversation_id": conversation.get("conversation_id", ""),
                    "metadata": conversation.get("metadata", {})
                })
        return qa_pairs

def main():
    """Main entry point for the pipeline."""
    parser = argparse.ArgumentParser(description="Chat Data Preprocessing Pipeline")
    parser.add_argument("--config", "-c", required=True, help="Path to configuration file")
    args = parser.parse_args()
    
    # Debug - print the raw config file
    with codecs.open(args.config, 'r', encoding='utf-8') as f:
        config_text = f.read()
        parsed_config = yaml.safe_load(config_text)
        print(f"Raw config from {args.config}:")
        print(f"Platform section: {parsed_config.get('platform', 'Not found')}")
    
    pipeline = Pipeline(args.config)
    pipeline.run()


if __name__ == "__main__":
    main() 