#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Factory for creating platform-specific data handlers.
"""

import logging
from typing import Dict, Any, Type

from platform_handlers.base_handler import BasePlatformHandler
from platform_handlers.generic_handler import GenericHandler
from platform_handlers.facebook_handler import FacebookHandler


class PlatformHandlerFactory:
    """
    Factory class for creating platform-specific data handlers.
    """
    
    # Registry of available handlers
    _handlers = {
        "generic": GenericHandler,
        "facebook": FacebookHandler,
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
            config: Configuration dictionary that should contain a platform section
            
        Returns:
            Instance of platform handler
        
        Raises:
            ValueError: If handler type is not supported
        """
        logger = logging.getLogger(__name__)
        
        # Extract the platform configuration section
        if "platform" in config and isinstance(config["platform"], dict):
            platform_config = config["platform"]
            logger.info(f"Found platform section in config: {platform_config}")
        else:
            logger.warning("No platform section found in config, using empty config")
            platform_config = {}
        
        # Determine the handler type to use
        handler_type = None
        
        # First check for explicit 'type' field
        if "type" in platform_config and platform_config["type"]:
            handler_type = platform_config["type"]
            logger.info(f"Using handler type from 'type' field: {handler_type}")
        
        # Then check for platform field 
        elif "platform" in platform_config and platform_config["platform"]:
            handler_type = platform_config["platform"]
            logger.info(f"Using handler type from 'platform' field: {handler_type}")
        
        # Fall back to generic if no type specified
        if not handler_type:
            logger.warning("No handler type specified, using generic handler")
            handler_type = "generic"
        
        logger.info(f"Selected handler type: {handler_type}")
        
        # Check if the handler type is supported
        if handler_type not in cls._handlers:
            logger.warning(f"Unsupported handler type: {handler_type}. Falling back to generic handler.")
            handler_type = "generic"
            
        # Create and return the handler instance
        handler_class = cls._handlers[handler_type]
        return handler_class(platform_config) 