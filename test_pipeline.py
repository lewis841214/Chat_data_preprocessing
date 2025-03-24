#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Test script for the chat data preprocessing pipeline.
"""

import os
import json
import logging
import argparse
from typing import List, Dict, Any

from config_handler import ConfigHandler, generate_default_config
from pipeline import Pipeline


def create_test_data(output_path: str, num_items: int = 10) -> None:
    """
    Create test data for the pipeline.
    
    Args:
        output_path: Path to save the test data
        num_items: Number of test items to create
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    test_data = []
    
    # Create some unique conversations
    for i in range(num_items - 3):
        item = {
            "conversation": [
                {
                    "role": "User",
                    "content": f"Tell me about topic {i}",
                    "do_train": False
                },
                {
                    "role": "Assistant",
                    "content": f"Here is information about topic {i}. It is very interesting.",
                    "do_train": True
                }
            ]
        }
        test_data.append(item)
    
    # Add some duplicates
    duplicate = {
        "conversation": [
            {
                "role": "User",
                "content": "What is the capital of France?",
                "do_train": False
            },
            {
                "role": "Assistant",
                "content": "The capital of France is Paris.",
                "do_train": True
            }
        ]
    }
    test_data.append(duplicate)
    test_data.append(duplicate.copy())
    
    # Add one with too much stopwords
    stopword_heavy = {
        "conversation": [
            {
                "role": "User",
                "content": "The the the and and and of of of in in in to to to",
                "do_train": False
            },
            {
                "role": "Assistant",
                "content": "I am not sure what you mean by that. Could you please clarify?",
                "do_train": True
            }
        ]
    }
    test_data.append(stopword_heavy)
    
    # Save test data
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(test_data, f, ensure_ascii=False, indent=2)
    
    print(f"Created {len(test_data)} test items at {output_path}")


def main():
    """Main entry point for the test script."""
    parser = argparse.ArgumentParser(description="Test the chat data preprocessing pipeline")
    parser.add_argument("--create-data", action="store_true", help="Create test data")
    parser.add_argument("--config", default="configs/default_config.yaml", help="Path to configuration file")
    parser.add_argument("--data-path", default="data/input/test_data.json", help="Path to test data")
    args = parser.parse_args()
    
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create directories
    os.makedirs("data/input", exist_ok=True)
    os.makedirs("data/output", exist_ok=True)
    os.makedirs("configs", exist_ok=True)
    
    # Create test data if requested
    if args.create_data:
        create_test_data(args.data_path)
    
    # Create default config if it doesn't exist
    if not os.path.exists(args.config):
        generate_default_config(args.config)
        
    # Ensure config has correct input path
    config = ConfigHandler(args.config).get_config()
    config["platform"]["input_path"] = args.data_path
    
    # Create temp config with updated path
    test_config_path = "configs/test_config.yaml"
    with open(test_config_path, 'w') as f:
        import yaml
        yaml.dump(config, f, default_flow_style=False)
    
    # Run pipeline
    pipeline = Pipeline(test_config_path)
    pipeline.run()


if __name__ == "__main__":
    main() 