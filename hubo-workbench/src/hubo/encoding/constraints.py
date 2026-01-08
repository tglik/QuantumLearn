from __future__ import annotations
from .hubo_model import HUBOModel, HUBOTerm


def one_hot_penalty(variables: list[str], penalty: float) -> list[HUBOTerm]:
    """
    One-hot constraint: exactly one of variables must be 1.
    Penalty form: P * (1 - sum_i x_i)^2
    Expands to: P * (1 + sum_i x_i^2 - 2*sum_i x_i + 2*sum_{i<j} x_i*x_j)
    Since x_i^2 = x_i (binary), this becomes:
    P * (1 + sum_i x_i - 2*sum_i x_i + 2*sum_{i<j} x_i*x_j)
    = P * (1 - sum_i x_i + 2*sum_{i<j} x_i*x_j)
    """
    terms = []
    # Linear: -P * x_i
    for v in variables:
        terms.append(HUBOTerm(vars=(v,), coeff=-penalty))
    # Quadratic: 2P * x_i * x_j
    for i, vi in enumerate(variables):
        for vj in variables[i + 1 :]:
            terms.append(HUBOTerm(vars=(vi, vj), coeff=2 * penalty))
    # Constant handled outside (add P)
    return terms


def at_most_one_penalty(variables: list[str], penalty: float) -> list[HUBOTerm]:
    """
    At-most-one constraint: sum_i x_i <= 1.
    Penalty: P * (sum_i x_i - 1)^2 if sum > 1, else 0.
    Simpler: just penalize pairs: P * sum_{i<j} x_i * x_j
    """
    terms = []
    for i, vi in enumerate(variables):
        for vj in variables[i + 1 :]:
            terms.append(HUBOTerm(vars=(vi, vj), coeff=penalty))
    return terms


def capacity_penalty(
    variables: list[str], capacity: int, penalty: float
) -> list[HUBOTerm]:
    """
    Capacity constraint: sum_i x_i <= capacity.
    Penalty: P * max(0, sum_i x_i - capacity)^2
    Approximation: P * (sum_i x_i - capacity)^2 (always active).
    Expands to: P * (sum_i x_i^2 + capacity^2 - 2*capacity*sum_i x_i + 2*sum_{i<j} x_i*x_j)
    = P * (sum_i x_i + capacity^2 - 2*capacity*sum_i x_i + 2*sum_{i<j} x_i*x_j)
    = P * ((1 - 2*capacity)*sum_i x_i + 2*sum_{i<j} x_i*x_j) + P*capacity^2
    """
    terms = []
    # Linear
    for v in variables:
        terms.append(HUBOTerm(vars=(v,), coeff=penalty * (1 - 2 * capacity)))
    # Quadratic
    for i, vi in enumerate(variables):
        for vj in variables[i + 1 :]:
            terms.append(HUBOTerm(vars=(vi, vj), coeff=2 * penalty))
    # Constant: P * capacity^2 added outside
    return terms
