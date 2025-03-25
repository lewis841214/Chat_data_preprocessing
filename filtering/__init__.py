"""
Filtering module for quality assessment of chat data.
Includes language detection, perplexity scoring, and ML classifiers.
"""

from filtering.filter_manager import (
    FilterManager,
    FastTextLanguageDetector,
    KenLMPerplexityScorer,
    LogisticRegressionClassifier
)

__all__ = [
    'FilterManager',
    'FastTextLanguageDetector',
    'KenLMPerplexityScorer',
    'LogisticRegressionClassifier'
] 