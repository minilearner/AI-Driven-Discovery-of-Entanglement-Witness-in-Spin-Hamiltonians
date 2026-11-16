from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

import torch


class QuantumState(ABC):
    """Minimal state interface consumed by the observable engine."""

    N: int

    @abstractmethod
    def expectation(self, sites: Sequence[int], operators: Sequence[torch.Tensor]) -> torch.Tensor:
        """Return an expectation value without constructing a dense many-body state."""

