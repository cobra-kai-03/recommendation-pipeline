"""Shared utilities: configuration loading, path management, logging.

Every pipeline stage imports from here so that paths, the random seed, and the
logging format are defined in exactly one place (single source of truth).
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict

import yaml

# Project root = parent of the `src` package directory.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"


def load_config(config_path: str | os.PathLike | None = None) -> Dict[str, Any]:
    """Load the YAML configuration file into a plain dict."""
    path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    with open(path, "r", encoding="utf-8") as fh:
        config = yaml.safe_load(fh)
    return config


def resolve_path(relative: str) -> Path:
    """Resolve a config path (relative to the project root) to an absolute Path."""
    p = Path(relative)
    return p if p.is_absolute() else (PROJECT_ROOT / p)


def ensure_dir(path: str | os.PathLike) -> Path:
    """Create a directory (and parents) if it does not exist; return it."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


_LOGGERS: Dict[str, logging.Logger] = {}


def get_logger(name: str, log_dir: str | os.PathLike | None = None) -> logging.Logger:
    """Return a configured logger that writes to both console and a file.

    The file handler writes to ``logs/pipeline.log`` so that a complete,
    timestamped run record is always available as a deliverable artifact.
    """
    if name in _LOGGERS:
        return _LOGGERS[name]

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(name)-22s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    if not logger.handlers:
        console = logging.StreamHandler()
        console.setFormatter(fmt)
        logger.addHandler(console)

        logs_path = ensure_dir(log_dir or (PROJECT_ROOT / "logs"))
        file_handler = logging.FileHandler(logs_path / "pipeline.log")
        file_handler.setFormatter(fmt)
        logger.addHandler(file_handler)

    _LOGGERS[name] = logger
    return logger


def banner(logger: logging.Logger, title: str) -> None:
    """Log a visually distinct stage banner."""
    line = "=" * 70
    logger.info(line)
    logger.info(title.upper())
    logger.info(line)
