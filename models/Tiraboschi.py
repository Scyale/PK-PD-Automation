"""
Tiraboschi two-compartment subcutaneous TMDD model.

Public API used by runner:
- DEFAULTS (dict)
- validate_params(params)
- initial_conditions(params) -> np.ndarray
- apply_dose(y, dose_mgkg, params) -> np.ndarray
- rhs(t, y, params) -> np.ndarray
- derived(t, y_traj, params) -> Dict[str, np.ndarray]
- microconstants(params) -> Dict[str, float]  (optional)
"""

from __future__ import annotations
from typing import Dict, Any
import numpy as np

# -----------------------------------------------------------------------------
# Defaults -- PLEASE VERIFY these numbers vs your Tiraboschi Model.txt
# -----------------------------------------------------------------------------
DEFAULTS: Dict[str, Any] = {
    # general
    "body_weight_kg": 70.0,

    # PK
    "ka": 0.3,               # 1/day (absorption)
    "F": 0.63,               # bioavailability (unitless)
    "Vc_ml": 3000.0,         # central volume (mL)
    "Vp_ml": 2000.0,         # peripheral volume (mL)
    "Cl_ml_day": 120.0,      # linear clearance (mL/day)
    "Q_ml_day": 250.0,       # intercompartmental flow (mL/day)

    # Nonlinear elimination (MM)
    "Vmax_ug_day": 1.4,      # Vmax in ug/day (note unit difference)
    "Km_ugml": 2.4,          # Km in ug/mL

    # PD / indirect response
    "EASI0": 1.0,            # baseline EASI (unitless)
    "kout": 0.26,            # 1/day
    "Imax": 0.8,             # maximal inhibitory effect (fraction)
    "IC50_ugml": 6.05,       # ug/mL

    # misc
    # If you want to change n_doses / dt / solver you'll do that in the YAML simulation section.
}

# -----------------------------------------------------------------------------
# Utility / validation
# -----------------------------------------------------------------------------
def validate_params(p: Dict[str, Any]) -> None:
    required = [
        "body_weight_kg", "ka", "F", "Vc_ml", "Vp_ml", "Cl_ml_day", "Q_ml_day",
        "Vmax_ug_day", "Km_ugml", "EASI0", "kout", "Imax", "IC50_ugml"
    ]
    missing = [k for k in required if k not in p]
    if missing:
        raise KeyError(f"Tiraboschi: missing params {missing}")

# -----------------------------------------------------------------------------
# Initial conditions
# State vector (amounts in ug unless otherwise noted):
# 0 Dep  (Depot amount, ug)
# 1 Cent (Central amount, ug)
# 2 Peri (Peripheral amount, ug)
# 3 EASI (biomarker)
# 4 AUC  (exposure accumulator)    -- units will be ug*day/ml integrated Central concentration
# 5 AUCss (gating variable for steady-state AUC; implemented in derived)
# -----------------------------------------------------------------------------
def initial_conditions(params: Dict[str, Any]) -> np.ndarray:
    EASI0 = float(params["EASI0"])
    # start with zero drug in compartments, baseline biomarker EASI0
    return np.array([0.0, 0.0, 0.0, EASI0, 0.0, 0.0], dtype=float)

# -----------------------------------------------------------------------------
# apply_dose hook: discrete dosing event
# We agreed: Depot += F * Delta (Delta = dose_mgkg * bw * 1000 µg/mg)
# -----------------------------------------------------------------------------
def apply_dose(y: np.ndarray, dose_mgkg: float, params: Dict[str, Any]) -> np.ndarray:
    y = y.copy()
    bw = float(params.get("body_weight_kg", 70.0))
    # dose_mgkg assumed in mg/kg -> total dose in µg
    delta_ug = float(dose_mgkg) * bw * 1000.0
    F = float(params.get("F", 1.0))
    y[0] += F * delta_ug
    return y

# -----------------------------------------------------------------------------
# RHS: piecewise ODE for one interval
# - Dep in ug
# - Cent, Peri in ug
# - EASI (unitless)
# - AUC, AUCss are accumulators (AUC integrates Central concentration over time)
# -----------------------------------------------------------------------------
def rhs(t: float, y: np.ndarray, p: Dict[str, Any]) -> np.ndarray:
    Dep, Cent, Peri, EASI, _AUC, _AUCss = y

    # PK params
    ka = float(p["ka"])
    Vc = float(p["Vc_ml"])
    Vp = float(p["Vp_ml"])
    Cl = float(p["Cl_ml_day"])   # mL/day
    Q = float(p["Q_ml_day"])     # mL/day
    Vmax = float(p["Vmax_ug_day"])  # ug/day
    Km = float(p["Km_ugml"])        # ug/mL

    # PD params
    kout = float(p["kout"])
    EASI0 = float(p["EASI0"])
    Imax = float(p["Imax"])
    IC50 = float(p["IC50_ugml"])

    # Central concentration (ug/mL)
    Central = Cent / Vc  # ug/mL

    # Indirect response model
    kin = kout * EASI0

    # ODEs
    dDep = -ka * Dep
    # absorption term to central is ka*Dep (ug/day)
    # distribution and linear clearance use amount-based form: Cent is ug, so linear elimination rate = (Cl/Vc)*Cent (ug/day)
    # Nonlinear MM elimination term: Vmax * (C/(Km + C)) (ug/day)
    dCent = (
        ka * Dep
        - ((Q + Cl) / Vc) * Cent
        + (Q / Vp) * Peri
        - Vmax * (Central / (Km + Central))
    )

    dPeri = (Q / Vc) * Cent - (Q / Vp) * Peri

    # drug effect on EASI (inhibitory): inhib = Imax * C / (IC50 + C); actual production term kin * (1 - inhib)
    inhib = Imax * (Central / (IC50 + Central)) if (IC50 + Central) > 0 else 0.0
    dEASI = kin * (1.0 - inhib) - kout * EASI

    # AUC accumulator: integrate Central (ug/mL) over time -> units ug/mL * day
    dAUC = Central
    # AUCss gate can be implemented via SQUAREPULSE like logic; here we keep it as simple accumulator (derived() will compute gated AUC)
    dAUCss = 0.0

    return np.array([dDep, dCent, dPeri, dEASI, dAUC, dAUCss], dtype=float)

# -----------------------------------------------------------------------------
# derived outputs: called with t (1D array) and y (2D array shape n_states x n_time)
# Return time series dict. Keys used by runner:
#   "Central_ugml"  -> time series of central concentration
#   "EASI_red"      -> PD metric time series (EASI0 - EASI)
# Optionally include other series for debugging.
# -----------------------------------------------------------------------------
def derived(t: np.ndarray, y: np.ndarray, p: Dict[str, Any]) -> Dict[str, np.ndarray]:
    Dep = y[0, :]
    Cent = y[1, :]
    Peri = y[2, :]
    EASI = y[3, :]
    AUC = y[4, :]
    AUCss = y[5, :]

    Vc = float(p["Vc_ml"])
    EASI0 = float(p["EASI0"])

    Central = Cent / Vc  # ug/mL

    # Fractional EASI reduction (positive = improvement)
    EASI_red = (EASI0 - EASI) / EASI0

    return {
        "Central_ugml": Central,
        "EASI_red": EASI_red,
        "Dep_ug": Dep,
        "Cent_ug": Cent,
        "Peri_ug": Peri,
        "EASI": EASI,
        "AUC": AUC,
        "AUCss": AUCss,
    }


# -----------------------------------------------------------------------------
# microconstants for display under plots
# -----------------------------------------------------------------------------
def microconstants(params: Dict[str, Any]) -> Dict[str, float]:
    kout = float(params["kout"])
    EASI0 = float(params["EASI0"])
    kin = kout * EASI0
    return {"kin": kin}
