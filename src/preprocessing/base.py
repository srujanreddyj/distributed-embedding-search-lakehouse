# src/preprocessing/base.py

from abc import ABC, abstractmethod
from typing import Any


class BasePreprocessor(ABC):
    """Stateful Ray actor for batch preprocessing."""

    @abstractmethod
    def __init__(self) -> None:
        """One-time setup: load models, tokenizers, etc."""
        ...

    @abstractmethod
    def __call__(self, batch: dict[str, list]) -> dict[str, list]:
        """Process a batch. Input/output are columnar (Arrow) batches."""
        ...
