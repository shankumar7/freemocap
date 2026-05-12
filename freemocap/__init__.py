"""A free and open source markerless motion capture system for everyone 💀✨"""

import logging

__author__ = """Skelly FreeMoCap"""
__email__ = "info@freemocap.org"
__version__ = "v1.8.2"
__description__ = "A free and open source markerless motion capture system for everyone 💀✨"

__package_name__ = "freemocap"
__repo_url__ = f"https://github.com/freemocap/{__package_name__}/"
__repo_issues_url__ = f"{__repo_url__}issues"

logger = logging.getLogger(__name__)

try:
	from freemocap.system.logging.configure_logging import configure_logging, LogLevel

	configure_logging(LogLevel.TRACE)
except Exception as exc:
	logger.debug("Skipping automatic logging configuration during import: %s", exc)

logger.info(f"Initializing {__package_name__} package, version: {__version__}, from file: {__file__}")
