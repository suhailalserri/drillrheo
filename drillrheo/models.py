"""Rheological model equations for drilling fluids.

Each function computes shear stress (tau, in Pa) from shear rate (gamma_dot,
in 1/s) and a set of model parameters. All functions accept either a scalar
or a numpy array for ``shear_rate`` and return the same shape.

Sign convention: ``shear_rate`` is the magnitude of the shear rate
(what the source literature calls -dv/dr), always >= 0. This matches how
Fann/Chan viscometer readings are reported and how the validation papers
define their equations.

References:
    Wisniowski, R.; Skrzypaszek, K.; Malachowski, T. Selection of a Suitable
    Rheological Model for Drilling Fluid Using Applied Numerical Methods.
    Energies 2020, 13, 3192.

    Wisniowski, R.; Skrzypaszek, K.; Toczek, P. Vom Berg and Hahn-Eyring
    Drilling Fluid Rheological Models. Energies 2022, 15, 5583.
"""
from __future__ import annotations

from typing import Union

import numpy as np

ArrayLike = Union[float, np.ndarray]


def bingham(shear_rate: ArrayLike, tau_y: float, mu_p: float) -> ArrayLike:
    """Bingham plastic model: tau = tau_y + mu_p * gamma_dot.

    Two-parameter linear model. Describes fluids with a yield stress that,
    once exceeded, flow with constant (Newtonian) plastic viscosity.

    Args:
        shear_rate: Shear rate(s) in 1/s. Must be >= 0.
        tau_y: Yield stress in Pa.
        mu_p: Plastic viscosity in Pa.s.

    Returns:
        Shear stress in Pa, same shape as ``shear_rate``.
    """
    shear_rate = np.asarray(shear_rate, dtype=float)
    return tau_y + mu_p * shear_rate


def power_law(shear_rate: ArrayLike, K: float, n: float) -> ArrayLike:
    """Ostwald-de Waele (power law) model: tau = K * gamma_dot^n.

    Two-parameter model with no yield stress. ``n`` < 1 indicates
    shear-thinning behavior (typical of most drilling fluids), ``n`` = 1
    reduces to a Newtonian fluid, and ``n`` > 1 indicates shear-thickening.

    Args:
        shear_rate: Shear rate(s) in 1/s. Must be >= 0.
        K: Consistency index in Pa.s^n.
        n: Flow behavior index, dimensionless.

    Returns:
        Shear stress in Pa, same shape as ``shear_rate``.
    """
    shear_rate = np.asarray(shear_rate, dtype=float)
    return K * np.power(shear_rate, n)


def herschel_bulkley(
    shear_rate: ArrayLike, tau_0: float, K: float, n: float
) -> ArrayLike:
    """Herschel-Bulkley (yield power law) model: tau = tau_0 + K * gamma_dot^n.

    Three-parameter model combining a yield stress with power-law behavior
    above that yield stress. Widely reported as the best general-purpose
    fit for drilling fluids and is one of the three models recommended by
    API RP 13D (along with Bingham and power law).

    Args:
        shear_rate: Shear rate(s) in 1/s. Must be >= 0.
        tau_0: Yield stress in Pa.
        K: Consistency index in Pa.s^n.
        n: Flow behavior index, dimensionless.

    Returns:
        Shear stress in Pa, same shape as ``shear_rate``.
    """
    shear_rate = np.asarray(shear_rate, dtype=float)
    return tau_0 + K * np.power(shear_rate, n)


def vom_berg(shear_rate: ArrayLike, tau_y: float, D: float, C: float) -> ArrayLike:
    """Vom Berg model: tau = tau_y + D * asinh(gamma_dot / C).

    Three-parameter model using an inverse hyperbolic sine rather than a
    power function to describe the curve above yield stress. Originally
    developed for cement pastes (Vom Berg, 1979) and shown by Wisniowski
    et al. (2020, 2022) to fit cement slurries and some drilling muds
    better than Herschel-Bulkley.

    Args:
        shear_rate: Shear rate(s) in 1/s. Must be >= 0.
        tau_y: Yield stress in Pa.
        D: Curve-shape parameter in Pa.
        C: Curve-shape parameter in 1/s (must be > 0).

    Returns:
        Shear stress in Pa, same shape as ``shear_rate``.
    """
    shear_rate = np.asarray(shear_rate, dtype=float)
    return tau_y + D * np.arcsinh(shear_rate / C)


def hahn_eyring(shear_rate: ArrayLike, E: float, D: float, C: float) -> ArrayLike:
    """Hahn-Eyring model: tau = E * gamma_dot + D * asinh(gamma_dot / C).

    Three-parameter model with no explicit yield stress term; instead
    combines a Newtonian-like linear term with an inverse hyperbolic sine
    term. Reported by Wisniowski et al. (2022) to give the strongest
    correlation of any model tested for cement slurries.

    Args:
        shear_rate: Shear rate(s) in 1/s. Must be >= 0.
        E: Linear (high-shear-rate) viscosity term in Pa.s.
        D: Curve-shape parameter in Pa.
        C: Curve-shape parameter in 1/s (must be > 0).

    Returns:
        Shear stress in Pa, same shape as ``shear_rate``.
    """
    shear_rate = np.asarray(shear_rate, dtype=float)
    return E * shear_rate + D * np.arcsinh(shear_rate / C)


# Registry mapping model name -> (callable, ordered param names).
# Used by fitting.py and model_selection.py to iterate over all models
# generically without hardcoding each name in multiple places.
MODEL_REGISTRY = {
    "Bingham": (bingham, ("tau_y", "mu_p")),
    "PowerLaw": (power_law, ("K", "n")),
    "HerschelBulkley": (herschel_bulkley, ("tau_0", "K", "n")),
    "VomBerg": (vom_berg, ("tau_y", "D", "C")),
    "HahnEyring": (hahn_eyring, ("E", "D", "C")),
}
