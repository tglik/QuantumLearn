from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Any, Dict

from .base import Problem
from hubo.encoding import HUBOModel, HUBOTerm, one_hot_penalty


@dataclass
class AssignmentProblem(Problem):
    """
    Bipartite assignment: assign n left nodes to n right nodes, minimize total cost.
    Variables: x_{i,j} = 1 if left i assigned to right j, else 0.
    Constraints: sum_j x_{i,j} = 1 for all i (one-hot per left node).
    Objective: sum_{i,j} cost[i,j] * x_{i,j}
    """

    cost_matrix: np.ndarray  # shape (n, n)
    penalty_strength: float = 100.0

    def __post_init__(self):
        assert self.cost_matrix.ndim == 2, "cost_matrix must be 2D"
        assert (
            self.cost_matrix.shape[0] == self.cost_matrix.shape[1]
        ), "cost_matrix must be square"

    @property
    def size(self) -> int:
        return self.cost_matrix.shape[0]

    def var_name(self, i: int, j: int) -> str:
        return f"x_{i}_{j}"

    def encode_hubo(self) -> HUBOModel:
        n = self.size
        variables = [self.var_name(i, j) for i in range(n) for j in range(n)]
        terms = []
        constant = 0.0

        # Objective: sum_{i,j} cost[i,j] * x_{i,j}
        for i in range(n):
            for j in range(n):
                terms.append(
                    HUBOTerm(
                        vars=(self.var_name(i, j),), coeff=float(self.cost_matrix[i, j])
                    )
                )

        # Constraints: one-hot for each left node
        for i in range(n):
            row_vars = [self.var_name(i, j) for j in range(n)]
            penalty_terms = one_hot_penalty(row_vars, self.penalty_strength)
            terms.extend(penalty_terms)
            constant += self.penalty_strength  # from (1 - sum)^2 expansion

        return HUBOModel(
            variables=variables,
            terms=terms,
            constant=constant,
            metadata=self.metadata(),
        )

    def validate_assignment(self, assignment: dict[str, int]) -> bool:
        """Check that each left node assigned to exactly one right node."""
        n = self.size
        for i in range(n):
            row_sum = sum(assignment.get(self.var_name(i, j), 0) for j in range(n))
            if row_sum != 1:
                return False
        return True

    def objective_value(self, assignment: dict[str, int]) -> float:
        """Sum of selected costs."""
        n = self.size
        total = 0.0
        for i in range(n):
            for j in range(n):
                if assignment.get(self.var_name(i, j), 0) == 1:
                    total += self.cost_matrix[i, j]
        return total

    def metadata(self) -> Dict[str, Any]:
        return {
            "problem_type": "assignment",
            "size": self.size,
            "num_variables": self.size * self.size,
            "penalty_strength": self.penalty_strength,
        }

    @staticmethod
    def random(
        n: int, seed: int = 42, cost_range: tuple[float, float] = (1.0, 10.0)
    ) -> "AssignmentProblem":
        """Generate random assignment problem."""
        rng = np.random.default_rng(seed)
        cost = rng.uniform(cost_range[0], cost_range[1], size=(n, n))
        return AssignmentProblem(
            cost_matrix=cost, penalty_strength=cost_range[1] * 10.0
        )
