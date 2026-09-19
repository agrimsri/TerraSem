"""Logging helpers for TerraSem.

TODO: implement structured JSON logging, TensorBoard/W&B integration.
"""

import logging


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Return a named logger with a simple console handler."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s %(name)s — %(message)s")
        )
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger
