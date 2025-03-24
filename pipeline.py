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

from config_handler import ConfigHandler
from platform.handler_factory import PlatformHandlerFactory
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
        self.platform_handler = PlatformHandlerFactory.get_handler(self.config)
        self.logger.info(f"Using platform handler: {self.platform_handler.__class__.__name__}")
        
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
    
    def run(self) -> None:
        """
        Execute the full pipeline process.
        """
        self.logger.info("Starting pipeline execution")
        
        # Step 1: Platform-specific data handling
        self.logger.info("Step 1: Platform-specific data handling")
        platform_data = self.platform_handler.process()
        
        # Step 2: Clean data (remove useless content)
        self.logger.info("Step 2: Cleaning - removing useless content")
        cleaned_data = self.cleaner.process(platform_data)
        
        # Step 3: Quality filtering
        self.logger.info("Step 3: Quality filtering")
        filtered_data = self.filter_manager.process(cleaned_data)
        
        # Step 4: Text formatting
        self.logger.info("Step 4: Text formatting")
        formatted_data = self.formatter.process(filtered_data)
        
        # Step 5: Deduplication (optional based on config)
        if self.config["deduplication"]["enabled"]:
            self.logger.info("Step 5: Deduplication")
            deduplicated_data = self.dedup_manager.process(formatted_data)
        else:
            self.logger.info("Step 5: Deduplication skipped")
            deduplicated_data = formatted_data
        
        # Step 6: Generate final corpus
        self.logger.info("Step 6: Generating final corpus")
        self._generate_final_corpus(deduplicated_data)
        
        self.logger.info("Pipeline execution completed successfully")
    
    def _generate_final_corpus(self, data: List[Dict[str, Any]]) -> None:
        """
        Generate the final corpus in the required JSON format.
        
        Args:
            data: Processed data to be saved
        """
        output_path = self.config["output"]["path"]
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Use proper UTF-8 encoding when writing the output file
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        self.logger.info(f"Final corpus saved to {output_path}")
        self.logger.info(f"Total conversations: {len(data)}")


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