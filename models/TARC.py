"""
EASI.py — Dupilumab PK/PD implementation to predict EASI change from baseline.

Based on Lebrikizumab model capturing drug effect.
Calibrated with phase 3 data from Dupixent ClinPharm review.

PK: Two-compartment model:
     SC depot -> central/peripheral (2-comp) 
     Linear clearance (Cl) + TMDD-like MM elimination (Vmax/Km).
PD: Indirect response model:
     dEASI = kin*(1 - Imax*C/(IC50+C)) - kout*EASI
     EASI_red = (EASI0 - EASI)/EASI0

Conventions:
- time: days
- amounts: ug
- volumes: mL
- concentrations: ug/mL

Runner API implemented:
- DEFAULTS, validate_params, initial_conditions, apply_dose, rhs, derived

TARC.py - Dupilumab PK/PD implementation to predict TARC change from baseline.

TARC Gain rate inhibited proportionally to "Central" concentration to capturing drug effect.
PD parameters were fitted to digitized SAD PKPD data from ClinPharm Review of Dupixent.
Calibrated with phase 2 data  

Structure:
PK: Two-compartment model:
     SC depot -> central/peripheral (2-comp) 
     Linear clearance (Cl) + TMDD-like MM elimination (Vmax/Km).
PD: Indirect response model:
     dTARC = kin*(1 - Imax*C/(IC50+C)) - kout*TARC
     TARC_red = (TARC0 - TARC) / TARC0

Conventions:
- time: days
- amounts: ug
- volumes: mL
- concentrations: ug/mL

"""

from __future__ import annotations

from typing import Dict, Any
import numpy as np


# Default parameters
DEFAULTS: Dict[str, Any] = {
    # Fixed workflow
    "body_weight_kg": 70.0,

    # PK
    "ka": 0.26,        # 1/day
    "F": 0.63,         # -
    "Vc_ml": 2790.0,   # mL
    "Cl_ml_day": 117.0,# mL/day
    "Q_ml_day": 263.0, # mL/day
    "Vp_ml": 1656.0,   # mL
    "Vmax_ugml_day": 1.39, # ug/mL/day
    "Km_ugml": 2.41,       # ug/mL

    # PD
    "TARC0": 1.0,      # baseline
    "kout": 0.26,      # 1/day
    "Imax": 0.8,       # -
    "IC50_ugml": 6.05, # ug/mL
}

def apply_dose(y, dose_mgkg: float, params):
    """
    SC dose into depot with bioavailability applied at the event:
      Dep += F * Delta_ug
    y[0] is Dep in ug.
    """
    bw = float(params["body_weight_kg"])   # kg (fixed 70 in your workflow)
    F = float(params["F"])                # -
    delta_ug = float(dose_mgkg) * bw * 1000.0  # mg/kg * kg * (ug/mg) = ug
    y[0] += F * delta_ug
    return y


def validate_params(p: Dict[str, Any]) -> None:
    required = [
        "body_weight_kg",
        "ka", "F", "Vc_ml", "Cl_ml_day", "Q_ml_day", "Vp_ml", "Vmax_ugml_day", "Km_ugml",
        "TARC0", "kout", "Imax", "IC50_ugml",
    ]
    missing = [k for k in required if k not in p]
    if missing:
        raise ValueError(f"Missing TARC params: {missing}")

    # Simple sanity checks
    if p["Vc_ml"] <= 0 or p["Vp_ml"] <= 0:
        raise ValueError("Vc_ml and Vp_ml must be > 0.")
    if p["ka"] <= 0 or p["Cl_ml_day"] < 0 or p["Q_ml_day"] < 0:
        raise ValueError("ka must be > 0; Cl_ml_day and Q_ml_day must be >= 0.")
    if p["Km_ugml"] <= 0 or p["Vmax_ugml_day"] < 0:
        raise ValueError("Km_ugml must be > 0; Vmax_ugml_day must be >= 0.")
    if not (0 <= p["F"] <= 1.5):
        raise ValueError("F should be in a reasonable range (0..~1).")
    if p["kout"] <= 0:
        raise ValueError("kout must be > 0.")
    if not (0 <= p["Imax"] <= 1.5):
        raise ValueError("Imax should be in a reasonable range (0..~1).")
    if p["IC50_ugml"] <= 0:
        raise ValueError("IC50_ugml must be > 0.")


def initial_conditions(p: Dict[str, Any]) -> np.ndarray:
    # Dep, Cent, Peri start at 0; TARC starts at baseline
    Dep0 = 0.0
    Cent0 = 0.0
    Peri0 = 0.0
    TARC0 = float(p["TARC0"])
    return np.array([Dep0, Cent0, Peri0, TARC0], dtype=float)


def rhs(t: float, y: np.ndarray, p: Dict[str, Any]) -> np.ndarray:
    Dep, Cent, Peri, TARC = y

    ka = float(p["ka"])
    Vc = float(p["Vc_ml"])
    Vp = float(p["Vp_ml"])
    Cl = float(p["Cl_ml_day"])
    Q  = float(p["Q_ml_day"])
    Vmax = float(p["Vmax_ugml_day"])
    Km = float(p["Km_ugml"])

    kout = float(p["kout"])
    TARC0 = float(p["TARC0"])
    Imax = float(p["Imax"])
    IC50 = float(p["IC50_ugml"])

    Central = Cent / Vc  # ug/mL

    # Indirect response: kin = kout*TARC0
    kin = kout * TARC0

    # ODEs
    dDep = -ka * Dep

    # MM elimination term written as in your spec: -Cent * Vmax/(Km + Central)
    # Units: Cent [ug] * Vmax [ug/mL/day] / [ug/mL] => ug/day
    dCent = (
        ka * Dep
        - ((Q + Cl) / Vc) * Cent
        + (Q / Vp) * Peri
        - Cent * (Vmax / (Km + Central))
    )

    dPeri = (Q / Vc) * Cent - (Q / Vp) * Peri

    inhib = 1.0 - Imax * (Central / (IC50 + Central))
    dTARC = kin * inhib - kout * TARC

    return np.array([dDep, dCent, dPeri, dTARC], dtype=float)


def derived(t: np.ndarray, y: np.ndarray, p: Dict[str, Any]) -> Dict[str, np.ndarray]:
    Dep = y[0, :]
    Cent = y[1, :]
    Peri = y[2, :]
    TARC = y[3, :]

    Vc = float(p["Vc_ml"])
    Central = Cent / Vc  # ug/mL

    Imax = float(p["Imax"])
    IC50 = float(p["IC50_ugml"])
    inhib = 1.0 - Imax * (Central / (IC50 + Central))

    TARC0 = float(p["TARC0"])
    TARC_red = (TARC0 - TARC) / TARC0

    return {
        "Central_ugml": Central,
        "TARC_red": TARC_red,
        "Dep_ug": Dep,
        "Cent_ug": Cent,
        "Peri_ug": Peri,
        "TARC": TARC,
        "inhib": inhib,
    }
