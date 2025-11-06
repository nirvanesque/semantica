"""
Core Orchestration Module

This module provides the main orchestration capabilities for the Semantica framework,
including the primary Semantica class, configuration management, lifecycle management,
and plugin system.

Main Components:
    - Semantica: Main framework class for orchestration and knowledge base building
    - Config: Configuration data class with validation
    - ConfigManager: Configuration loading, validation, and management
    - LifecycleManager: System lifecycle management with hooks and health monitoring
    - PluginRegistry: Dynamic plugin discovery, loading, and management

Example Usage:
    >>> from semantica.core import Semantica, ConfigManager
    >>> # Initialize framework
    >>> framework = Semantica()
    >>> # Load configuration
    >>> config_manager = ConfigManager()
    >>> config = config_manager.load_from_file("config.yaml")

Author: Semantica Contributors
License: MIT
"""

from .orchestrator import Semantica
from .config_manager import Config, ConfigManager
from .lifecycle import LifecycleManager, SystemState, HealthStatus
from .plugin_registry import PluginRegistry, PluginInfo, LoadedPlugin

__all__ = [
    # Main orchestrator
    "Semantica",
    # Configuration
    "Config",
    "ConfigManager",
    # Lifecycle
    "LifecycleManager",
    "SystemState",
    "HealthStatus",
    # Plugins
    "PluginRegistry",
    "PluginInfo",
    "LoadedPlugin",
]