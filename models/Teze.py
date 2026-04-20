"""
Tezepelumab PK/PD FeNO model adapted from:
Ly et al. (2021), Pharmacokinetic and Pharmacodynamic Modeling of
Tezepelumab to Guide Phase 3 Dose Selection for Patients With Severe Asthma.

Implemented for the workflow runner API:
- DEFAULTS
- validate_params
- initial_conditions
- apply_dose
- rhs
- derived

Model structure used here
-------------------------
PK:
- 2-compartment model
- first-order SC absorption
- first-order linear elimination from central compartment

PD (FeNO):
- direct concentration-response model (no delay state)
- paper equation on the log scale:
    log(FeNO(t)) = log(base) + log(Emax_ratio) * C / (EC50 + C)
  which corresponds to:
    FeNO(t) = base * Emax_ratio ** (C / (EC50 + C))

Conventions in this implementation
----------------------------------
- time: days
- amounts: ug
- volumes: mL
- concentrations: ug/mL
- FeNO: ppb

Notes
-----
- The source paper reports fixed SC doses (e.g. 210 mg Q4W). This workflow's
  runner uses dose_mgkg. Therefore dosing is handled as mg/kg and converted to
  total micrograms using body_weight_kg.
- For a 70 kg subject, 210 mg fixed dose corresponds to 3.0 mg/kg.
- The published PK-PD model also included some biomarker covariate effects on
  baseline FeNO and Emax. To keep the model simple and robust for this
  workflow, the default implementation uses the typical-population parameters
  reported in the paper and exposes FeNO_base_ppb directly as a parameter.
"""

from __future__ import annotations

from typing import Dict, Any
import numpy as np


DEFAULTS: Dict[str, Any] = {
    # subject
    "body_weight_kg": 70.0,

    # PK typical values from the population PK model (Table 3)
    "ka_day": 0.247,        # 1/day
    "F": 0.645,             # bioavailability
    "Vc_ml": 4280.0,        # 4.28 L
    "Vp_ml": 2290.0,        # 2.29 L
    "Cl_ml_day": 201.0,     # 0.201 L/day
    "Q_ml_day": 726.0,      # 0.726 L/day (distribution clearance)

    # PD typical values from the PK-FeNO model (Table 4)
    "FeNO_base_ppb": 21.1,  # typical baseline FeNO
    "Emax_ratio": 0.722,    # asymptotic FeNO / baseline ratio at very high C
    "EC50_ugml": 2.50,      # ug/mL

    # output cap
    "FeNO_red_cap": 1.0,
}


def validate_params(p: Dict[str, Any]) -> None:
    req = [
        "body_weight_kg",
        "ka_day",
        "F",
        "Vc_ml",
        "Vp_ml",
        "Cl_ml_day",
        "Q_ml_day",
        "FeNO_base_ppb",
        "Emax_ratio",
        "EC50_ugml",
        "FeNO_red_cap",
    ]
    missing = [k for k in req if k not in p]
    if missing:
        raise KeyError(f"Teze: missing params {missing}")

    for k in ["body_weight_kg", "ka_day", "Vc_ml", "Vp_ml", "Cl_ml_day", "Q_ml_day", "FeNO_base_ppb", "EC50_ugml"]:
        if float(p[k]) <= 0:
            raise ValueError(f"{k} must be > 0")

    F = float(p["F"])
    if not (0.0 < F <= 1.5):
        raise ValueError("F must be in a reasonable range (0, 1.5]")

    emax_ratio = float(p["Emax_ratio"])
    if not (0.0 < emax_ratio <= 1.0):
        raise ValueError("Emax_ratio must be in the range (0, 1]")

    cap = float(p["FeNO_red_cap"])
    if cap < 0:
        raise ValueError("FeNO_red_cap must be >= 0")


def initial_conditions(p: Dict[str, Any]) -> np.ndarray:
    # States:
    # 0 Dep (ug)
    # 1 Cent (ug)
    # 2 Peri (ug)
    # 3 AUC (ug*day/mL)
    return np.array([0.0, 0.0, 0.0, 0.0], dtype=float)



def apply_dose(y: np.ndarray, dose_mgkg: float, p: Dict[str, Any]) -> np.ndarray:
    """
    SC bolus into depot; bioavailability applied at administration time.
    dose_mgkg * body_weight_kg * 1000 -> ug
    """
    y = y.copy()
    bw = float(p["body_weight_kg"])
    F = float(p["F"])
    delta_ug = float(dose_mgkg) * bw * 1000.0
    y[0] += F * delta_ug
    return y



def rhs(t: float, y: np.ndarray, p: Dict[str, Any]) -> np.ndarray:
    Dep, Cent, Peri, AUC = y

    ka = float(p["ka_day"])
    Vc = float(p["Vc_ml"])
    Vp = float(p["Vp_ml"])
    Cl = float(p["Cl_ml_day"])
    Q = float(p["Q_ml_day"])

    C = Cent / Vc  # ug/mL

    dDep = -ka * Dep
    dCent = ka * Dep - ((Q + Cl) / Vc) * Cent + (Q / Vp) * Peri
    dPeri = (Q / Vc) * Cent - (Q / Vp) * Peri
    dAUC = C

    return np.array([dDep, dCent, dPeri, dAUC], dtype=float)



def derived(t: np.ndarray, y: np.ndarray, p: Dict[str, Any]) -> Dict[str, np.ndarray]:
    Dep = y[0, :]
    Cent = y[1, :]
    Peri = y[2, :]
    AUC = y[3, :]

    Vc = float(p["Vc_ml"])
    C = Cent / Vc  # ug/mL

    FeNO_base = float(p["FeNO_base_ppb"])
    Emax_ratio = float(p["Emax_ratio"])
    EC50 = float(p["EC50_ugml"])

    frac = C / (EC50 + C)
    FeNO_ppb = FeNO_base * np.power(Emax_ratio, frac)

    FeNO_red = (FeNO_base - FeNO_ppb) / FeNO_base
    FeNO_red = np.clip(FeNO_red, 0.0, float(p.get("FeNO_red_cap", 1.0)))
    FeNO_pct_red = FeNO_red * 100.0

    return {
        "Central_ugml": C,
        "FeNO_ppb": FeNO_ppb,
        "FeNO_red": FeNO_red,
        "FeNO_pct_red": FeNO_pct_red,
        "Dep_ug": Dep,
        "Cent_ug": Cent,
        "Peri_ug": Peri,
        "AUC": AUC,
    }
