"""
Centralized structured logger for the DNS Attack Detection system.
Uses python-json-logger for structured JSON output and supports
rotating file handlers with configurable log levels.
"""
import logging
import logging.handlers
import os
import sys
from typing import Optional

from pythonjsonlogger import jsonlogger


_LOGGERS: dict[str, logging.Logger] = {}

_LOG_LEVEL_MAP = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL,
}


def get_logger(
    name: str,
    level: str = "INFO",
    log_file: Optional[str] = None,
    max_bytes: int = 10_485_760,
    backup_count: int = 5,
    fmt: str = "json",
) -> logging.Logger:
    """Return (or create) a named logger with optional file rotation.

    Args:
        name: Logger name (typically ``__name__``).
        level: Log level string (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        log_file: Optional path to the rotating log file.
        max_bytes: Maximum file size before rotation (default 10 MB).
        backup_count: Number of backup log files to retain.
        fmt: Output format – ``"json"`` (default) or ``"text"``.

    Returns:
        Configured :class:`logging.Logger` instance.
    """
    if name in _LOGGERS:
        return _LOGGERS[name]

    logger = logging.getLogger(name)
    logger.setLevel(_LOG_LEVEL_MAP.get(level.upper(), logging.INFO))
    logger.propagate = False

    if fmt == "json":
        formatter = jsonlogger.JsonFormatter(
            fmt="%(asctime)s %(name)s %(levelname)s %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S",
        )
    else:
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler (optional)
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8",
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    _LOGGERS[name] = logger
    return logger
