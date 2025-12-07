"""
Routes package for Water Intelligence API.

Contains modular API routers for different feature domains.
"""

from .generative import router as generative_router

__all__ = ["generative_router"]
