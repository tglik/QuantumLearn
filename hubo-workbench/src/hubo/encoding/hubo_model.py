from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict


@dataclass
class HUBOTerm:
    """A single term: c * Π_{i in vars} x_i"""

    vars: tuple[str, ...]
    coeff: float

    def __post_init__(self):
        assert len(self.vars) > 0, "HUBOTerm must have at least one variable"
        assert len(self.vars) == len(
            set(self.vars)
        ), "HUBOTerm variables must be unique"


@dataclass
class HUBOModel:
    """HUBO: sum_{S} c_S * Π_{i in S} x_i"""

    variables: list[str]
    terms: list[HUBOTerm]
    constant: float = 0.0
    metadata: Dict = field(default_factory=dict)

    def degree(self) -> int:
        """Maximum term degree (k for k-local)"""
        return max((len(t.vars) for t in self.terms), default=0)

    def energy(self, assignment: dict[str, int]) -> float:
        """Evaluate HUBO at binary assignment {var: 0 or 1}"""
        assert all(
            v in assignment for v in self.variables
        ), "Assignment must cover all variables"
        assert all(
            val in (0, 1) for val in assignment.values()
        ), "Values must be binary"

        total = self.constant
        for term in self.terms:
            product = 1
            for var in term.vars:
                product *= assignment[var]
            total += term.coeff * product
        return total

    def num_terms(self) -> int:
        return len(self.terms)

    def num_variables(self) -> int:
        return len(self.variables)


@dataclass
class QUBOModel:
    """QUBO: sum_{i,j} Q_{ij} x_i x_j (quadratic only, degree<=2)"""

    variables: list[str]
    Q: Dict[tuple[str, str], float] = field(default_factory=dict)
    linear: Dict[str, float] = field(default_factory=dict)
    constant: float = 0.0
    metadata: Dict = field(default_factory=dict)

    def energy(self, assignment: dict[str, int]) -> float:
        """Evaluate QUBO at binary assignment."""
        assert all(
            v in assignment for v in self.variables
        ), "Assignment must cover all variables"
        assert all(
            val in (0, 1) for val in assignment.values()
        ), "Values must be binary"

        total = self.constant
        for var, coeff in self.linear.items():
            total += coeff * assignment[var]
        for (vi, vj), coeff in self.Q.items():
            total += coeff * assignment[vi] * assignment[vj]
        return total

    def degree(self) -> int:
        return 2 if self.Q else (1 if self.linear else 0)

    def num_terms(self) -> int:
        return len(self.linear) + len(self.Q)

    def num_variables(self) -> int:
        return len(self.variables)
