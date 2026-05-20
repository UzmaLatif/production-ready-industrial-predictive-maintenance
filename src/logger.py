import logging
import os
import sys
from datetime import datetime


def get_logger(name: str, log_file: str = "logs/pipeline.log") -> logging.Logger:
    """
    Creates and returns a logger that writes to both
    the terminal AND a log file simultaneously.

    Args:
        name: Name of the module calling the logger
        log_file: Path to the log file

    Returns:
        Configured logger object
    """
    # ── Create logs directory if it doesn't exist ──────────────────────────
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    # ── Create logger ───────────────────────────────────────────────────────
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # ── Format: timestamp | level | module | message ────────────────────────
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # ── Handler 1: Write to log file ────────────────────────────────────────
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(formatter)

    # ── Handler 2: Print to terminal ────────────────────────────────────────
    console_handler = logging.StreamHandler(
        open(sys.stdout.fileno(), mode="w", encoding="utf-8", buffering=1)
    )
    console_handler.setFormatter(formatter)

    # ── Attach handlers to logger ───────────────────────────────────────────
    if not logger.handlers:
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

    return logger