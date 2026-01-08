from __future__ import annotations
from dataclasses import dataclass, field
import itertools
from typing import Dict

from .hubo_model import HUBOModel, HUBOTerm, QUBOModel


@dataclass
class ReductionResult:
    """Result of HUBO->QUBO reduction."""

    qubo: QUBOModel
    ancilla_count: int
    ancilla_map: Dict[tuple[str, ...], str]  # term vars -> ancilla name
    penalty_strength: float
    overhead_factor: float  # (QUBO vars) / (HUBO vars)


def rosenberg_reduction(
    hubo: HUBOModel, penalty_scale: float = 10.0
) -> ReductionResult:
    """
    Reduce HUBO to QUBO using Rosenberg reduction.
    For each k-local term (k>2), introduce k-2 ancillas and penalty constraints.
    Penalty ensures ancilla = product of original variables at optimum.
    """
    max_degree = hubo.degree()
    if max_degree <= 2:
        # Already quadratic; convert directly
        return _convert_quadratic_hubo(hubo)

    # Determine penalty strength: must dominate coefficient magnitudes
    max_abs_coeff = max((abs(t.coeff) for t in hubo.terms), default=1.0)
    penalty = penalty_scale * max_abs_coeff

    ancilla_map: Dict[tuple[str, ...], str] = {}
    ancilla_count = 0
    qubo_vars = list(hubo.variables)
    Q: Dict[tuple[str, str], float] = {}
    linear: Dict[str, float] = {}
    constant = hubo.constant

    for term in hubo.terms:
        if len(term.vars) <= 2:
            _add_term_to_qubo(term.vars, term.coeff, Q, linear)
        else:
            # Reduce k-local to quadratic with ancillas
            # Use chain: z1 = x1*x2, z2 = z1*x3, ..., z_{k-2} = z_{k-3}*x_{k-1}, final = z_{k-2}*x_k
            variables_list = list(term.vars)
            prev = variables_list[0]
            for idx in range(1, len(variables_list) - 1):
                ancilla_name = f"anc_{ancilla_count}"
                ancilla_count += 1
                qubo_vars.append(ancilla_name)
                x = prev
                y = variables_list[idx]
                z = ancilla_name
                ancilla_map[tuple(sorted([x, y]))] = z
                # Penalty: P*(z - x*y)^2 = P*(z^2 + x^2*y^2 - 2*x*y*z)
                # Binary: x^2=x, so: P*(z + x*y - 2*x*y*z) = P*z + P*x*y - 2P*x*y*z
                linear[z] = linear.get(z, 0.0) + penalty
                _add_quadratic_term(Q, x, y, penalty)
                _add_quadratic_term(Q, x, z, -2 * penalty)
                _add_quadratic_term(Q, y, z, -2 * penalty)
                prev = ancilla_name

            # Final product: prev * last_var gets original coefficient
            last_var = variables_list[-1]
            _add_quadratic_term(Q, prev, last_var, term.coeff)

    qubo = QUBOModel(
        variables=qubo_vars,
        Q=Q,
        linear=linear,
        constant=constant,
        metadata={**hubo.metadata, "reduced_from_hubo": True},
    )
    overhead = len(qubo_vars) / len(hubo.variables) if hubo.variables else 1.0
    return ReductionResult(
        qubo=qubo,
        ancilla_count=ancilla_count,
        ancilla_map=ancilla_map,
        penalty_strength=penalty,
        overhead_factor=overhead,
    )


def _convert_quadratic_hubo(hubo: HUBOModel) -> ReductionResult:
    """Convert HUBO with degree<=2 directly to QUBO (no ancillas)."""
    Q: Dict[tuple[str, str], float] = {}
    linear: Dict[str, float] = {}
    for term in hubo.terms:
        _add_term_to_qubo(term.vars, term.coeff, Q, linear)

    qubo = QUBOModel(
        variables=list(hubo.variables),
        Q=Q,
        linear=linear,
        constant=hubo.constant,
        metadata={**hubo.metadata, "reduced_from_hubo": False},
    )
    return ReductionResult(
        qubo=qubo,
        ancilla_count=0,
        ancilla_map={},
        penalty_strength=0.0,
        overhead_factor=1.0,
    )


def _add_term_to_qubo(vars: tuple[str, ...], coeff: float, Q: Dict, linear: Dict):
    if len(vars) == 1:
        linear[vars[0]] = linear.get(vars[0], 0.0) + coeff
    elif len(vars) == 2:
        _add_quadratic_term(Q, vars[0], vars[1], coeff)
    else:
        raise ValueError(f"Cannot add term of degree {len(vars)} to QUBO directly")


def _add_quadratic_term(
    Q: Dict[tuple[str, str], float], vi: str, vj: str, coeff: float
):
    key = tuple(sorted([vi, vj]))
    Q[key] = Q.get(key, 0.0) + coeff
