#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Filter manager for quality assessment of chat data.
"""

import os
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import logging
from tqdm import tqdm

from processor import BaseProcessor

# Handle torch import more carefully
torch = None
try:
    # Define a simple platform module replacement if needed
    class SimplePlatform:
        @staticmethod
        def system():
            return "Linux"  # Default to Linux
    
    # Try direct import first
    try:
        import torch
    except AttributeError as e:
        # If we get platform.system() error, use our custom platform
        if "platform" in str(e) and "system" in str(e):
            import sys
            sys.modules['platform'] = SimplePlatform()
            import torch
except Exception as e:
    logging.error(f"Failed to import torch: {e}")
    # Create a minimal torch replacement for CPU-only operation
    class DummyTorch:
        class nn:
            class Sequential:
                def __init__(self, *args):
                    self.layers = args
                
                def __call__(self, x):
                    return 0.5  # Default quality score
                
                def to(self, device):
                    return self
            
            class Linear:
                def __init__(self, in_features, out_features):
                    pass
                
                class weight:
                    class data:
                        pass
                
                class bias:
                    class data:
                        pass
            
            class Sigmoid:
                def __init__(self):
                    pass
        
        class Tensor:
            def __init__(self, data, **kwargs):
                self.data = data
            
            def item(self):
                return 0.5  # Default quality score
        
        @staticmethod
        def tensor(data, **kwargs):
            return DummyTorch.Tensor(data)
        
        @staticmethod
        def no_grad():
            class NoGradContext:
                def __enter__(self):
                    return self
                
                def __exit__(self, *args):
                    pass
            
            return NoGradContext()
        
        def device(device_str):
            return device_str
    
    torch = DummyTorch()
    logging.warning("Using dummy torch implementation with limited functionality")


class FastTextLanguageDetector:
    """Language detection using Facebook's FastText model."""
    
    def __init__(self, model_path: str):
        """
        Initialize the FastText language detector.
        
        Args:
            model_path: Path to the FastText language identification model
        """
        self.logger = logging.getLogger(__name__)
        self.model_path = model_path
        self.model = None
        
        # Check if model exists and warn with explicit path
        if not os.path.exists(model_path):
            abs_path = os.path.abspath(model_path)
            self.logger.error(f"FastText model not found at {abs_path}")
            self.logger.info(f"Please download the model using: curl -L 'https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin' -o '{abs_path}'")
            return
            
        try:
            import fasttext
            self.logger.info(f"Loading FastText model from {model_path}")
            self.model = fasttext.load_model(model_path)
            self.logger.info("FastText model loaded successfully")
        except ImportError:
            self.logger.error("FastText package not installed. Please install via pip install fasttext")
    
    def detect_language(self, text: str) -> Tuple[str, float]:
        """
        Detect the language of the given text.
        
        Args:
            text: Input text
            
        Returns:
            Tuple of (language_code, confidence_score)
        """
        if not self.model or not text or len(text.strip()) == 0:
            return ("unknown", 0.0)
        
        # FastText expects a cleaned input
        cleaned_text = text.strip().replace("\n", " ")
        prediction = self.model.predict(cleaned_text, k=1)
        
        # Extract language code (removing '__label__' prefix)
        lang_code = prediction[0][0].replace('__label__', '')
        confidence = float(prediction[1][0])
        
        return (lang_code, confidence)


class KenLMPerplexityScorer:
    """Text quality scoring using KenLM perplexity models."""
    
    def __init__(self, model_paths: Dict[str, str]):
        """
        Initialize the KenLM perplexity scorer.
        
        Args:
            model_paths: Dictionary mapping language codes to KenLM model paths
        """
        self.logger = logging.getLogger(__name__)
        self.model_paths = model_paths
        self.models = {}
        
        if not model_paths:
            self.logger.warning("No KenLM model paths provided.")
            return
            
        try:
            import kenlm
            for lang, path in model_paths.items():
                if os.path.exists(path):
                    self.logger.info(f"Loading KenLM model for {lang} from {path}")
                    self.models[lang] = kenlm.Model(path)
                    self.logger.info(f"KenLM model for {lang} loaded successfully")
                else:
                    abs_path = os.path.abspath(path)
                    self.logger.error(f"KenLM model for {lang} not found at {abs_path}")
                    if lang == "en":
                        self.logger.info("Download an English KenLM model from: https://github.com/kpu/kenlm/archive/master.zip")
                        self.logger.info("Or build your own: https://github.com/kpu/kenlm#building-language-models")
                    elif lang == "zh":
                        self.logger.info("For Chinese models, consider using: https://github.com/kpu/kenlm#building-language-models")
        except ImportError:
            self.logger.error("KenLM package not installed. Please install via pip install https://github.com/kpu/kenlm/archive/master.zip")
    
    def get_perplexity(self, text: str, lang: str) -> float:
        """
        Calculate the perplexity score for text in the given language.
        
        Args:
            text: Input text
            lang: Language code
            
        Returns:
            Perplexity score (lower is better)
        """
        if lang not in self.models or not text:
            return float('inf')
        
        model = self.models[lang]
        
        # Normalize text for scoring
        normalized = text.strip().replace("\n", " ")
        
        # Calculate log10 probability
        log_prob = model.score(normalized) / len(normalized.split())
        
        # Convert to perplexity: 10^(-log10(P))
        perplexity = 10 ** (-log_prob)
        
        return perplexity


class LogisticRegressionClassifier:
    """Quality classification using logistic regression."""
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize the logistic regression classifier.
        
        Args:
            model_path: Path to the pre-trained logistic regression model (optional)
        """
        self.logger = logging.getLogger(__name__)
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Define features used by the model
        self.features = [
            "lang_confidence", 
            "perplexity_score", 
            "char_ratio",
            "avg_word_length",
            "special_char_ratio"
        ]
        
        if model_path and os.path.exists(model_path):
            try:
                self.logger.info(f"Loading logistic regression model from {model_path}")
                self.model = torch.load(model_path, map_location=self.device)
                self.logger.info("Logistic regression model loaded successfully")
            except Exception as e:
                self.logger.error(f"Failed to load logistic regression model: {str(e)}")
        else:
            # Create a simple logistic regression model
            self.logger.info("Creating new logistic regression model")
            self._create_new_model()
    
    def _create_new_model(self):
        """Create a new logistic regression model."""
        input_dim = len(self.features)
        self.model = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 1),
            torch.nn.Sigmoid()
        ).to(self.device)
        
        # Initialize with some reasonable defaults
        with torch.no_grad():
            # Set initial weights to prioritize perplexity and language confidence
            self.model[0].weight.data = torch.tensor([[0.8, -0.7, 0.5, 0.3, -0.4]], device=self.device)
            self.model[0].bias.data = torch.tensor([0.0], device=self.device)
    
    def extract_features(self, text: str, lang_confidence: float, perplexity: float) -> torch.Tensor:
        """
        Extract features for the logistic regression model.
        
        Args:
            text: Input text
            lang_confidence: Language detection confidence
            perplexity: Perplexity score
            
        Returns:
            Tensor of features
        """
        # Additional features to help assess translation quality
        char_count = len(text)
        word_count = len(text.split())
        avg_word_length = char_count / max(1, word_count)
        
        # Ratio of special characters
        special_chars = sum(1 for c in text if not c.isalnum() and not c.isspace())
        special_char_ratio = special_chars / max(1, char_count)
        
        # Ratio of characters to words (can help detect machine translations)
        char_ratio = char_count / max(1, word_count)
        
        # Normalize perplexity to 0-1 range (lower is better)
        norm_perplexity = min(1.0, perplexity / 1000.0)
        
        features = [
            lang_confidence,
            norm_perplexity,
            min(5.0, char_ratio) / 5.0,  # Normalize
            min(15.0, avg_word_length) / 15.0,  # Normalize
            special_char_ratio
        ]
        
        return torch.tensor([features], dtype=torch.float32, device=self.device)
    
    def predict(self, text: str, lang_confidence: float, perplexity: float) -> float:
        """
        Predict the quality score for a text.
        
        Args:
            text: Input text
            lang_confidence: Language detection confidence
            perplexity: Perplexity score
            
        Returns:
            Quality score between 0.0 and 1.0
        """
        if not self.model:
            # Default logic if no model is available
            # Good quality = high language confidence and low perplexity
            return max(0.0, min(1.0, lang_confidence * (1.0 - min(1.0, perplexity / 1000.0))))
        
        with torch.no_grad():
            features = self.extract_features(text, lang_confidence, perplexity)
            score = self.model(features).item()
            return score


class FilterManager(BaseProcessor):
    """
    Manager for quality filtering of chat data.
    Orchestrates various filters for language detection, perplexity scoring, etc.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the filter manager.
        
        Args:
            config: Configuration dictionary
        """
        super().__init__(config)
        self.logger.info("Initializing filter manager with quality assessment components")
        
        # Initialize components based on config
        self.fasttext_enabled = config.get("fasttext_enabled", False)
        self.kenlm_enabled = config.get("kenlm_enabled", False)
        self.logistic_regression_enabled = config.get("logistic_regression_enabled", False)
        self.target_languages = set(config.get("target_languages", ["en"]))
        
        # Quality threshold - conversations with score below this will be filtered out
        self.quality_threshold = config.get("quality_threshold", 0.5)
        
        # Flag to track if we have working models
        self.models_available = False
        
        # Initialize FastText language detector if enabled
        self.fasttext = None
        if self.fasttext_enabled:
            model_path = config.get("fasttext_model_path")
            if model_path:
                self.fasttext = FastTextLanguageDetector(model_path)
                if self.fasttext.model is not None:
                    self.models_available = True
            else:
                self.logger.warning("FastText enabled but no model path provided")
        
        # Initialize KenLM perplexity scorer if enabled
        self.kenlm = None
        if self.kenlm_enabled:
            model_paths = config.get("kenlm_model_path", {})
            if model_paths:
                self.kenlm = KenLMPerplexityScorer(model_paths)
                if any(self.kenlm.models):
                    self.models_available = True
            else:
                self.logger.warning("KenLM enabled but no model paths provided")
        
        # Initialize logistic regression classifier if enabled
        self.lr_classifier = None
        if self.logistic_regression_enabled:
            model_path = config.get("logistic_regression_model_path")
            self.lr_classifier = LogisticRegressionClassifier(model_path)
            if hasattr(self.lr_classifier, 'model') and self.lr_classifier.model is not None:
                self.models_available = True
    
    def process(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Apply quality filtering to the input data.
        
        Args:
            data: Input data to filter
            
        Returns:
            Filtered data
        """
        self.logger.info(f"Applying quality filtering on {len(data)} items")
        
        # Check if any filtering methods are enabled and models available
        if not (self.fasttext_enabled or self.kenlm_enabled or self.logistic_regression_enabled):
            self.logger.warning("No filtering methods enabled, skipping quality assessment")
            return data
            
        # If no models are available but filtering is enabled, warn but continue without filtering
        if not self.models_available:
            self.logger.warning("Filtering methods are enabled but no models are available. Running in basic mode.")
            # Add quality metrics with default values but don't filter
            for item in data:
                text = item.get("content", "")
                item["quality_metrics"] = {
                    "language": "unknown",
                    "lang_confidence": 0.0,
                    "perplexity": float('inf'),
                    "quality_score": 0.5  # Neutral score
                }
            return data
        
        filtered_data = []
        rejected_count = 0
        
        for item in tqdm(data, desc="Quality filtering"):
            # Get the text to evaluate
            text = item.get("content", "")
            if not text or len(text.strip()) == 0:
                rejected_count += 1
                continue
            
            # Default quality values
            item["quality_metrics"] = {
                "language": "unknown",
                "lang_confidence": 0.0,
                "perplexity": float('inf'),
                "quality_score": 0.0
            }
            
            # Step 1: Language detection with FastText
            language = "unknown"
            lang_confidence = 0.0
            
            if self.fasttext_enabled and self.fasttext and self.fasttext.model:
                language, lang_confidence = self.fasttext.detect_language(text)
                item["quality_metrics"]["language"] = language
                item["quality_metrics"]["lang_confidence"] = lang_confidence
                
                # Skip if not in target languages
                if language not in self.target_languages:
                    rejected_count += 1
                    continue
            
            # Step 2: Perplexity scoring with KenLM
            perplexity = float('inf')
            
            if self.kenlm_enabled and self.kenlm and self.kenlm.models and language in self.kenlm.models:
                perplexity = self.kenlm.get_perplexity(text, language)
                item["quality_metrics"]["perplexity"] = perplexity
            
            # Step 3: Overall quality assessment with logistic regression
            quality_score = 0.0
            
            if self.logistic_regression_enabled and self.lr_classifier:
                quality_score = self.lr_classifier.predict(text, lang_confidence, perplexity)
                item["quality_metrics"]["quality_score"] = quality_score
            
            # Accept or reject based on quality score (if we have working models)
            if self.models_available:
                if quality_score >= self.quality_threshold:
                    filtered_data.append(item)
                else:
                    rejected_count += 1
            else:
                # If no models, accept everything
                filtered_data.append(item)
        
        self.logger.info(f"Quality filtering complete. Accepted: {len(filtered_data)}, Rejected: {rejected_count}")
        return filtered_data 