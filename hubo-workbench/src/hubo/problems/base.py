from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Dict

from hubo.encoding import HUBOModel, QUBOModel, rosenberg_reduction, ReductionResult


class Problem(ABC):
    """Base problem interface for encoding to HUBO/QUBO."""

    @abstractmethod
    def encode_hubo(self) -> HUBOModel:
        """Return HUBO encoding of the problem."""
        pass

    def encode_qubo(
        self, penalty_scale: float = 10.0
    ) -> tuple[QUBOModel, ReductionResult]:
        """
        Return QUBO encoding (via Rosenberg reduction if needed).
        Returns (qubo, reduction_result).
        """
        hubo = self.encode_hubo()
        result = rosenberg_reduction(hubo, penalty_scale=penalty_scale)
        return result.qubo, result

    @abstractmethod
    def validate_assignment(self, assignment: dict[str, int]) -> bool:
        """Check if assignment satisfies hard constraints."""
        pass

    @abstractmethod
    def objective_value(self, assignment: dict[str, int]) -> float:
        """Compute objective value for the assignment (lower is better)."""
        pass

    @abstractmethod
    def metadata(self) -> Dict[str, Any]:
        """Return problem metadata for tracking."""
        pass
