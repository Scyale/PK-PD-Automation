"""Lebri (lebrikizumab) PK/PD model reconstructed from Berkeley Madonna .mmd.

Source: FDA_review_reconstruction.mmd (model equations section; XML/UI settings ignored).

Conventions (matches workflow runner):
- Time: days
- SC depot, central, peripheral amounts: µg
- Central concentration: Central_ugml = Central / Vc_ml (µg/mL)
- PD: EASI (absolute score), with baseline EASI0
  - EASI_pctchg = 100*(EASI/EASI0 - 1)  (negative = improvement vs baseline)

State vector:
  y = [Dep, Cent, Peri, EASI, AUC]

Notes:
- The original .mmd uses PULSE/SQUAREPULSE-style inputs to build the dosing regimen.
  In this workflow, dosing events are discrete and handled by runner.py; apply_dose()
  injects a bolus into the SC depot at each dose time.
"""

from __future__ import annotations

from typing import Dict, Any
import numpy as np


DEFAULTS: Dict[str, Any] = {
    # Fixed workflow
    "body_weight_kg": 70.0,

    # Initial conditions
    "EASI0": 30.0,     # baseline EASI score
    "AUC0": 0.0,       # exposure accumulator initial value

    # PK parameters (from .mmd)
    "Fsc": 0.86,        # bioavailability
    "ka": 0.303,        # 1/day
    "Vc_ml": 3860.0,    # mL
    "Vp_ml": 1280.0,    # mL
    "Q_ml_day": 525.0,  # mL/day
    "Cl_ml_day": 156.0, # mL/day

    # PD parameters (from .mmd)
    "kout": 0.0523,     # 1/day
    "Imax": 0.83,       # maximal inhibitory effect
    "IC50_ugml": 16.5,  # µg/mL

    # Placebo effect constant (from .mmd): 1/(1+exp(-0.702))
    "Pbo": float(1.0 / (1.0 + np.exp(-0.702))),
}


def validate_params(p: Dict[str, Any]) -> None:
    required = [
        "body_weight_kg",
        "EASI0", "AUC0",
        "Fsc", "ka", "Vc_ml", "Vp_ml", "Q_ml_day", "Cl_ml_day",
        "kout", "Imax", "IC50_ugml", "Pbo",
    ]
    missing = [k for k in required if k not in p]
    if missing:
        raise KeyError(f"Lebri: missing params {missing}")

    if float(p["body_weight_kg"]) <= 0:
        raise ValueError("body_weight_kg must be > 0")

    for k in ["ka", "Vc_ml", "Vp_ml", "Q_ml_day", "Cl_ml_day", "kout", "IC50_ugml"]:
        if float(p[k]) <= 0:
            raise ValueError(f"{k} must be > 0 (got {p[k]})")

    if not (0 <= float(p["Fsc"]) <= 1.5):
        raise ValueError("Fsc should be in a reasonable range (0..~1).")

    if not (0 <= float(p["Imax"]) <= 1.5):
        raise ValueError("Imax should be in a reasonable range (0..~1).")


def initial_conditions(p: Dict[str, Any]) -> np.ndarray:
    EASI0 = float(p["EASI0"])
    AUC0 = float(p.get("AUC0", 0.0))
    # Dep, Cent, Peri, EASI, AUC
    return np.array([0.0, 0.0, 0.0, EASI0, AUC0], dtype=float)


def apply_dose(y: np.ndarray, dose_mgkg: float, p: Dict[str, Any]) -> np.ndarray:
    """SC bolus into depot with bioavailability applied at the event."""
    y = y.copy()
    bw = float(p["body_weight_kg"])
    Fsc = float(p["Fsc"])
    delta_ug = float(dose_mgkg) * bw * 1000.0  # mg/kg * kg * (µg/mg) = µg
    y[0] += Fsc * delta_ug
    return y


def rhs(t: float, y: np.ndarray, p: Dict[str, Any]) -> np.ndarray:
    Dep, Cent, Peri, EASI, AUC = y

    # PK
    ka = float(p["ka"])
    Vc = float(p["Vc_ml"])
    Vp = float(p["Vp_ml"])
    Q = float(p["Q_ml_day"])
    Cl = float(p["Cl_ml_day"])

    Absorption = ka * Dep
    Arteries = (Q / Vc) * Cent
    Veins = (Q / Vp) * Peri
    Clearance = (Cl / Vc) * Cent

    dDep = -Absorption
    dCent = Absorption + Veins - Arteries - Clearance
    dPeri = Arteries - Veins

    # Concentration (µg/mL)
    C2 = Cent / Vc

    # PD
    EASI0 = float(p["EASI0"])
    kout = float(p["kout"])
    kin = kout * EASI0
    Imax = float(p["Imax"])
    IC50 = float(p["IC50_ugml"])
    Pbo = float(p["Pbo"])

    # Inhibitory effect term from .mmd: EFF = 1 - Imax*C2/(IC50 + C2)
    EFF = 1.0 - Imax * (C2 / (IC50 + C2)) if (IC50 + C2) > 0 else 1.0

    Gain = kin * EFF * (1.0 - Pbo)
    Loss = kout * EASI
    dEASI = Gain - Loss

    # Exposure accumulator
    dAUC = C2

    return np.array([dDep, dCent, dPeri, dEASI, dAUC], dtype=float)


def derived(t: np.ndarray, y: np.ndarray, p: Dict[str, Any]) -> Dict[str, np.ndarray]:
    # y shape: (n_states, n_timepoints) or (n_states,) for single time
    Dep = y[0, :]
    Cent = y[1, :]
    Peri = y[2, :]
    EASI = y[3, :]
    AUC = y[4, :]

    Vc = float(p["Vc_ml"])
    Central_ugml = Cent / Vc
    Pbo = float(p["Pbo"])
    Imax = float(p["Imax"])
    
    EASI0 = float(p["EASI0"])
    EASI_red = (EASI0 - EASI) / EASI0
    EASI_norm = (EASI_red - Pbo) / (1.0 - Pbo)

    return {
        "Central_ugml": Central_ugml,
        "EASI_red": EASI_red,
        "EASI_norm": EASI_norm,
        "Dep_ug": Dep,
        "Cent_ug": Cent,
        "Peri_ug": Peri,
        "EASI": EASI,
        "AUC": AUC,
}


def microconstants(p: Dict[str, Any]) -> Dict[str, float]:
    """Small helper for notebook parameter readout."""
    EASI0 = float(p["EASI0"])
    kout = float(p["kout"])
    kin = kout * EASI0
    return {"kin": kin}
