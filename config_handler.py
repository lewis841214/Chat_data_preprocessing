#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Configuration handler for the chat data preprocessing pipeline.
Provides functionality to load, validate, and access configuration settings.
"""

import os
import yaml
import logging
from typing import Dict, Any, Optional


class ConfigHandler:
    """
    Handles loading and validation of pipeline configuration.
    """
    
    # Default configuration for each component
    DEFAULT_CONFIG = {
        "platform": {
            "type": "generic",
            "input_path": "data/input/",
            "batch_size": 1000
        },
        "cleaning": {
            "stopwords_enabled": True,
            "language_filter_enabled": True,
            "url_filter_enabled": True,
            "paragraph_filter_enabled": True,
            "exact_dedup_enabled": True,
            "min_length": 10,
            "max_length": 32768
        },
        "filtering": {
            "fasttext_enabled": True,
            "fasttext_model_path": "models/fasttext/lid.176.bin",
            "kenlm_enabled": True,
            "kenlm_model_path": {
                "en": "models/kenlm/en.arpa.bin",
                "zh": "models/kenlm/zh.arpa.bin"
            },
            "gpt_evaluation_enabled": False,
            "logistic_regression_enabled": True,
            "target_languages": ["en", "zh"]
        },
        "formatting": {
            "remove_punctuation": False,
            "clean_html": True,
            "fix_unicode": True
        },
        "deduplication": {
            "enabled": True,
            "method": "minhash_lsh",  # Options: minhash_lsh, simhash, semantic, suffix_array, dbscan, bertopic, bloom_filter
            "threshold": 0.8,
            "ngram_size": 5,
            "num_permutations": 128,
            "band_size": 8
        },
        "output": {
            "path": "data/output/corpus.json",
            "format": "json"
        }
    }
    
    def __init__(self, config_path: str):
        """
        Initialize the configuration handler.
        
        Args:
            config_path: Path to the YAML configuration file
        """
        self.logger = logging.getLogger("ConfigHandler")
        self.config_path = config_path
        self.config = self._load_config()
        
    def _load_config(self) -> Dict[str, Any]:
        """
        Load configuration from YAML file and merge with defaults.
        
        Returns:
            Dict containing the configuration
        """
        if not os.path.exists(self.config_path):
            self.logger.warning(f"Config file {self.config_path} not found. Using default configuration.")
            return self.DEFAULT_CONFIG.copy()
        
        try:
            with open(self.config_path, 'r') as f:
                user_config = yaml.safe_load(f)
                
            # Merge user config with default config
            merged_config = self.DEFAULT_CONFIG.copy()
            
            for section, values in user_config.items():
                if section in merged_config:
                    if isinstance(values, dict) and isinstance(merged_config[section], dict):
                        merged_config[section].update(values)
                    else:
                        merged_config[section] = values
                else:
                    merged_config[section] = values
            
            self.logger.info(f"Loaded configuration from {self.config_path}")
            return merged_config
            
        except Exception as e:
            self.logger.error(f"Failed to load config from {self.config_path}: {str(e)}")
            self.logger.warning("Using default configuration")
            return self.DEFAULT_CONFIG.copy()
    
    def get_config(self) -> Dict[str, Any]:
        """
        Get the complete configuration.
        
        Returns:
            Dict containing the configuration
        """
        return self.config
    
    def get_section(self, section: str) -> Dict[str, Any]:
        """
        Get a specific section of the configuration.
        
        Args:
            section: The section name
            
        Returns:
            Dict containing the section configuration
        """
        return self.config.get(section, {})
    
    def save_config(self, output_path: Optional[str] = None) -> None:
        """
        Save the current configuration to a YAML file.
        
        Args:
            output_path: Path to save the configuration (defaults to original path)
        """
        path = output_path or self.config_path
        
        try:
            with open(path, 'w') as f:
                yaml.dump(self.config, f, default_flow_style=False)
            self.logger.info(f"Configuration saved to {path}")
        except Exception as e:
            self.logger.error(f"Failed to save config to {path}: {str(e)}")
            

def generate_default_config(output_path: str) -> None:
    """
    Generate a default configuration file.
    
    Args:
        output_path: Path to save the default configuration
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w') as f:
        yaml.dump(ConfigHandler.DEFAULT_CONFIG, f, default_flow_style=False)
    
    print(f"Default configuration saved to {output_path}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate default configuration")
    parser.add_argument("--output", "-o", default="configs/default_config.yaml", 
                        help="Path to save the default configuration")
    args = parser.parse_args()
    
    generate_default_config(args.output) 