###########################################################################
# KAMINO: Utilities: Message Logging
###########################################################################
"""
Copyright (c) 2025 Walt Disney Enterprises, Inc.
Disney Research, Zurich. All rights reserved.

Authors:
    Josefine Klintberg (josefine.klintberg@disney.com)
    Vassilios Tsounis (vassilios.tsounis@disney.com)
"""

import logging
from enum import IntEnum


class LogLevel(IntEnum):
    """Enumeration for log levels."""
    DEBUG = logging.DEBUG
    INFO = logging.INFO
    WARNING = logging.WARNING
    ERROR = logging.ERROR
    CRITICAL = logging.CRITICAL


class Logger(logging.Formatter):
    """Base logger with color highlighting for log levels."""

    GREY = "\x1b[38;20m"
    RED = "\x1b[31;20m"
    BOLD_RED = "\x1b[31;1m"
    BLUE = "\x1b[34;20m"
    BOLD_BLUE = "\x1b[34;1m"
    GREEN = "\x1b[32;20m"
    BOLD_GREEN = "\x1b[32;1m"
    YELLOW = "\x1b[33;20m"
    BOLD_YELLOW = "\x1b[33;1m"
    RESET = "\x1b[0m"

    # TODO: How to include the filename and line number in the log messages when called from the func wrappers?
    LINE_FORMAT = ("[%(asctime)s][%(filename)s:%(lineno)d][%(levelname)s]: %(message)s")
    """Line format for the log messages, including timestamp, filename, line number, log level, and message."""

    FORMATS = {
        logging.DEBUG: BLUE + LINE_FORMAT + RESET,
        logging.INFO: GREY + LINE_FORMAT + RESET,
        logging.WARNING: YELLOW + LINE_FORMAT + RESET,
        logging.ERROR: RED + LINE_FORMAT + RESET,
        logging.CRITICAL: BOLD_RED + LINE_FORMAT + RESET,
    }
    """Dictionary mapping log levels to their respective formats."""

    def __init__(self):
        """Initialize the Logger with a stream handler and set the default logging level."""
        super().__init__()
        # Create a stream handler with the custom logger format.
        self._streamhandler = logging.StreamHandler()
        self._streamhandler.setFormatter(self)
        # Set the default logging level to DEBUG
        logging.basicConfig(
            handlers=[self._streamhandler],
            level=logging.WARNING,  # Default level set to WARNING
        )

    def format(self, record):
        """Format the log record with the appropriate color based on the log level."""
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)

    def get(self):
        """Get the global logger instance."""
        return logging.getLogger()


LOGGER: Logger | None = None
"""Global logger instance for the application."""


def get_default_logger() -> logging.Logger:
    """Initialize the global logger instance."""
    global LOGGER
    if LOGGER is None:
        LOGGER = Logger()
    return LOGGER.get()


def set_log_level(level: LogLevel):
    """Set the logging level for the default logger."""
    get_default_logger().setLevel(level)
    get_default_logger().info(f"Log level set to: {logging.getLevelName(level)}")


def info(msg: str, *args, **kwargs):
    """Log an info message."""
    get_default_logger().info(msg, *args, **kwargs, stacklevel=2)


def debug(msg: str, *args, **kwargs):
    """Log a debug message."""
    get_default_logger().debug(msg, *args, **kwargs, stacklevel=2)


def warning(msg: str, *args, **kwargs):
    """Log a warning message."""
    get_default_logger().warning(msg, *args, **kwargs, stacklevel=2)


def error(msg: str, *args, **kwargs):
    """Log an error message."""
    get_default_logger().error(msg, *args, **kwargs, stacklevel=2)


def critical(msg: str, *args, **kwargs):
    """Log a critical message."""
    get_default_logger().critical(msg, *args, **kwargs, stacklevel=2)
