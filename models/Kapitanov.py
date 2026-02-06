"""Kapitanov TMDD 3-compartment PK/PD model (Berkeley Madonna reconstruction).

Conventions:
- Time: days
- Drug amounts D1,D2,D3: nmol
- Free receptors R1,R2,R3: nM
- Complex amounts DR1,DR2,DR3: nmol
- Volumes V1,V2,V3: L

Derived:
- C1 = D1/V1 (nM)
- Central_ugml = C1 * MW * 1e-6 (µg/mL), with MW in g/mol
- RO3 = (DR3/V3) / (R3 + DR3/V3)

Notes:
- Loading-dose constructs in the original .mmd are intentionally not implemented.
- n_doses is handled by the runner (fixed to 25).
"""
from __future__ import annotations

from typing import Dict, Any, List
import numpy as np

STATE_NAMES: List[str] = [
    "D1", "D2", "D3",
    "R1", "R2", "R3",
    "DR1", "DR2", "DR3",
    "AUC_total",
]

DEFAULTS: Dict[str, float] = {
    # Volumes (L)
    "V1": 2.29,
    "V2": 12.4,
    "V3": 0.563,

    # PK macro (days, dimensionless)
    "t_half": 32.8,
    "td_12": 55.9/24.0,
    "td_13": 30.0/24.0,
    "P12": 0.352,
    "P13": 0.3,

    # Receptor baseline (nM)
    "CR_1": 0.00605,
    "CR_2": 0.127,
    "CR_3": 2.02,

    # Receptor degradation (1/day)
    "kdeg": 1.0 * 24.0,

    # Binding kinetics (kon intended units: 1/(nM*day))
    "kon": 1e-3 * 86400.0,
    "koff": 3.3e-5 * 86400.0,

    # Constants
    "MW": 146899.0,          # g/mol
    "body_weight_kg": 70.0,  # fixed
}

def apply_dose(y, dose_mgkg: float, params):
    """
    Bolus into central drug amount D1.
    y[0] is assumed to be D1 in nmol.
    """
    MW = float(params["MW"])               # g/mol
    bw = float(params["body_weight_kg"])   # kg
    delta_nmol = dose_mgkg * bw / MW * 1e6  # nmol
    y[0] += delta_nmol
    return y


def compute_micro_params(p: Dict[str, float]) -> Dict[str, float]:
    """Compute micro-rate constants from the Berkeley Madonna formulas."""
    V1, V2, V3 = p["V1"], p["V2"], p["V3"]
    td_12, td_13 = p["td_12"], p["td_13"]
    P12, P13 = p["P12"], p["P13"]
    t_half = p["t_half"]

    ln2 = np.log(2.0)

    k12 = (ln2/td_12) * (P12*V2) / (V1 + P12*V2)
    k21 = (ln2/td_12) * (V1)     / (V1 + P12*V2)

    k13 = (ln2/td_13) * (P13*V3) / (V1 + P13*V3)
    k31 = (ln2/td_13) * (V1)     / (V1 + P13*V3)

    kel = ln2 / t_half

    return {"k12": k12, "k21": k21, "k13": k13, "k31": k31, "kel": kel}

def validate_params(params: Dict[str, Any]) -> None:
    required = set(DEFAULTS.keys())
    missing = required - set(params.keys())
    if missing:
        raise ValueError(f"Missing required params: {sorted(missing)}")
    for k in ["V1","V2","V3","t_half","td_12","td_13","kon","koff","kdeg","MW","body_weight_kg"]:
        if params[k] <= 0:
            raise ValueError(f"Parameter {k} must be > 0 (got {params[k]})")

def initial_conditions(params: Dict[str, float]) -> np.ndarray:
    """Initial state vector y0."""
    y0 = np.zeros(len(STATE_NAMES), dtype=float)
    y0[3] = params["CR_1"]
    y0[4] = params["CR_2"]
    y0[5] = params["CR_3"]
    return y0

def rhs(t: float, y: np.ndarray, params: Dict[str, float]) -> np.ndarray:
    """Right-hand side dy/dt."""
    D1, D2, D3 = y[0], y[1], y[2]
    R1, R2, R3 = y[3], y[4], y[5]
    DR1, DR2, DR3 = y[6], y[7], y[8]

    V1, V2, V3 = params["V1"], params["V2"], params["V3"]
    kon, koff, kdeg = params["kon"], params["koff"], params["kdeg"]

    micro = compute_micro_params(params)
    k12, k21, k13, k31, kel = micro["k12"], micro["k21"], micro["k13"], micro["k31"], micro["kel"]

    ksyn1 = params["CR_1"] * kdeg
    ksyn2 = params["CR_2"] * kdeg
    ksyn3 = params["CR_3"] * kdeg

    C1 = D1 / V1
    C2 = D2 / V2
    C3 = D3 / V3

    Bind1 = kon * C1 * R1
    Bind2 = kon * C2 * R2
    Bind3 = kon * C3 * R3

    dD1 = (k21*D2 + k31*D3) - (k12 + k13 + kel)*D1 - (Bind1*V1) + (koff*DR1)
    dD2 = (k12*D1) - (k21 + kel)*D2 - (Bind2*V2) + (koff*DR2)
    dD3 = (k13*D1) - (k31 + kel)*D3 - (Bind3*V3) + (koff*DR3)

    dR1 = ksyn1 - kdeg*R1 - Bind1 + koff*(DR1/V1)
    dR2 = ksyn2 - kdeg*R2 - Bind2 + koff*(DR2/V2)
    dR3 = ksyn3 - kdeg*R3 - Bind3 + koff*(DR3/V3)

    dDR1 = (Bind1*V1) - (koff + kdeg)*DR1
    dDR2 = (Bind2*V2) - (koff + kdeg)*DR2
    dDR3 = (Bind3*V3) - (koff + kdeg)*DR3

    MW = params["MW"]
    Central_ugml = C1 * MW * 1e-6
    dAUC_total = Central_ugml

    return np.array([dD1, dD2, dD3, dR1, dR2, dR3, dDR1, dDR2, dDR3, dAUC_total], dtype=float)

def derived(t: np.ndarray, y: np.ndarray, params: Dict[str, float]) -> Dict[str, np.ndarray]:
    """Compute derived outputs on arrays."""
    V1, V3 = params["V1"], params["V3"]
    MW = params["MW"]

    D1 = y[0, :]
    R3 = y[5, :]
    DR3 = y[8, :]

    C1 = D1 / V1
    Central_ugml = C1 * MW * 1e-6

    DR3_c = DR3 / V3
    RO3 = DR3_c / (R3 + DR3_c)

    return {"Central_ugml": Central_ugml, "RO3": RO3, "C1_nM": C1}
