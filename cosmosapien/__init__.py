"""Cosmosapien CLI - A modular command-line tool for multiple LLM providers."""

__version__ = "0.1.0"
__author__ = "Cosmosapien CLI Team"

from .core.config import Config
from .core.router import Router

__all__ = ["Config", "Router"] 