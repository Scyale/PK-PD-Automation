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
"""

from __future__ import annotations
from typing import Dict, Any
import numpy as np

DEFAULTS: Dict[str, Any] = {
    # subject
    "body_weight_kg": 70.0,

    # PK 
    "ka_day": 0.26,          # 1/day
    "F": 0.63,               # bioavailability
    "Vc_ml": 2790.0,         # central vol (mL)
    "Vp_ml": 1656.0,         # peripheral vol (mL)
    "Cl_ml_day": 117.0,      # linear clearance from central (mL/day)
    "Q_ml_day": 263.0,       # intercompartmental clearance (mL/day)
    "Vmax_ugml_day": 1.39,   # nonlinear Vmax (ug/mL/day)
    "Km_ugml": 2.41,         # nonlinear Km (ug/mL)

    # PD 
    "EASI0": 30.0,           # baseline EASI
    "Imax": 0.8,             # maximal inhibitory effect (fraction)
    "IC50_ugml": 6.05,       # ug/mL
    "kout_day": 0.05,        # 1/day

    # limits
    "EASI_red_cap": 1.0,
}

def validate_params(p: Dict[str, Any]) -> None:
    req = ["ka_day","F","Vc_ml","Vp_ml","Cl_ml_day","Q_ml_day","Vmax_ugml_day","Km_ugml",
           "EASI0","Imax","IC50_ugml","kout_day","EASI_red_cap","body_weight_kg"]
    missing = [k for k in req if k not in p]
    if missing:
        raise KeyError(f"EASI model: missing params {missing}")
    if float(p["Vc_ml"])<=0 or float(p["Vp_ml"])<=0:
        raise ValueError("Volumes must be > 0")
    for k in ["ka_day","Q_ml_day","Cl_ml_day","kout_day","IC50_ugml","Km_ugml"]:
        if float(p[k]) <= 0:
            raise ValueError(f"{k} must be > 0")

def initial_conditions(p: Dict[str, Any]) -> np.ndarray:
    # States:
    # 0 Dep (ug)
    # 1 Cent (ug)
    # 2 Peri (ug)
    # 3 EASI (score units)
    # 4 AUC (ug*day/mL)
    EASI0 = float(p["EASI0"])
    return np.array([0.0, 0.0, 0.0, EASI0, 0.0], dtype=float)

def apply_dose(y: np.ndarray, dose_mgkg: float, p: Dict[str, Any]) -> np.ndarray:
    """
    SC bolus into depot; F applied at administration time.
    dose_mgkg * body_weight_kg * 1000 -> ug
    """
    y = y.copy()
    bw = float(p["body_weight_kg"])
    F = float(p["F"])
    delta_ug = float(dose_mgkg) * bw * 1000.0
    y[0] += F * delta_ug
    return y

def rhs(t: float, y: np.ndarray, p: Dict[str, Any]) -> np.ndarray:
    Dep, Cent, Peri, EASI, AUC = y

    # PK params
    ka = float(p["ka_day"])
    Vc = float(p["Vc_ml"])
    Vp = float(p["Vp_ml"])
    Q = float(p["Q_ml_day"])
    Cl = float(p["Cl_ml_day"])
    Vmax = float(p["Vmax_ugml_day"])
    Km = float(p["Km_ugml"])

    # concentrations
    C = Cent / Vc  # ug/mL

    # depot absorption
    dDep = -ka * Dep

    # linear elimination (Cl in mL/day -> amount/day = Cent/Vc * Cl)
    elim_lin = (Cl / Vc) * Cent  # ug/day 

    # nonlinear (MM) elimination (amount-based)
    denom = (Km + C)
    elim_nl = Cent * (Vmax / denom) if denom > 0 else 0.0

    # central dynamics: absorption + distribution - elim
    dCent = ka * Dep - (Q/Vc) * Cent + (Q/Vp) * Peri - elim_lin - elim_nl

    # peripheral
    dPeri = (Q/Vc) * Cent - (Q/Vp) * Peri

    # PD parameters
    EASI0 = float(p["EASI0"])
    Imax = float(p["Imax"])
    IC50 = float(p["IC50_ugml"])
    kout = float(p["kout_day"])
    kin = kout * EASI0

    # EASI turnover (inhibitory effect on kin)
    effect = Imax * (C / (IC50 + C)) if (IC50 + C) > 0 else 0.0
    dEASI = kin * (1.0 - effect) - kout * EASI

    # AUC of central concentration (ug/mL * day)
    dAUC = C

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
    
    EASI0 = float(p["EASI0"])
    EASI_red = (EASI0 - EASI) / EASI0
    EASI_red = np.clip(EASI_red, 0.0, float(p.get("EASI_red_cap", 1.0)))
    # Positive values represent improvement (reduction from baseline).
    EASI_pct_red = EASI_red * 100.0

    return {
        "Central_ugml": Central_ugml,
        "EASI_pct_red": EASI_pct_red,
        "EASI_red": EASI_red,
        "Dep_ug": Dep,
        "Cent_ug": Cent,
        "Peri_ug": Peri,
        "EASI": EASI,
        "AUC": AUC,
}
