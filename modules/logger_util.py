import logging
from typing import Optional


def get_logger(name: Optional[str] = None, level: str = "INFO") -> logging.Logger:
    if name is None:
        name = "sam2-playground"
    logger = logging.getLogger(name)
    logger.setLevel(level.upper())

    return logger

