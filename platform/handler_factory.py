#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Factory for creating platform-specific data handlers.
"""

import logging
from typing import Dict, Any, Type

from platform.base_handler import BasePlatformHandler
from platform.generic_handler import GenericHandler


class PlatformHandlerFactory:
    """
    Factory class for creating platform-specific data handlers.
    """
    
    # Registry of available handlers
    _handlers = {
        "generic": GenericHandler,
        # Add other handlers as they are implemented
        # "reddit": RedditHandler,
        # "twitter": TwitterHandler,
        # "discord": DiscordHandler,
    }
    
    @classmethod
    def register_handler(cls, name: str, handler_class: Type[BasePlatformHandler]) -> None:
        """
        Register a new handler class.
        
        Args:
            name: Name of the handler
            handler_class: Handler class to register
        """
        cls._handlers[name] = handler_class
        logging.getLogger(__name__).info(f"Registered platform handler: {name}")
    
    @classmethod
    def get_handler(cls, config: Dict[str, Any]) -> BasePlatformHandler:
        """
        Get a platform handler instance based on configuration.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            Instance of platform handler
        
        Raises:
            ValueError: If handler type is not supported
        """
        handler_type = config.get("type", "generic")
        
        if handler_type not in cls._handlers:
            logger = logging.getLogger(__name__)
            logger.warning(f"Unsupported handler type: {handler_type}. Falling back to generic handler.")
            handler_type = "generic"
            
        handler_class = cls._handlers[handler_type]
        return handler_class(config) 