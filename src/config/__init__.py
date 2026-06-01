"""
src/config
----------
Configuration package.  Import from here throughout the codebase::

    from src.config import settings, AppSettings
"""
from src.config.settings import AppSettings, settings

__all__ = ["AppSettings", "settings"]
