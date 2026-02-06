"""Runner for Kapitanov sweeps (reference run included).

Usage (from repo root):
  python scripts/run_sweep.py --config configs/Kapitanov_sweep.yaml --dose_mgkg 1.0 --interval_weeks 2

Outputs:
  results/Kapitanov_sweep/1.0mgkg_q2w/
    run.h5
    run_config.json
    run_summary.json
"""
from __future__ import annotations

import sys
from pathlib import Path as _Path
# Ensure repo root is on sys.path so `models.*` imports work when running as a script
sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

import json
from pathlib import Path
from typing import Dict, Any
import importlib

import numpy as np
import yaml
import h5py
from scipy.integrate import solve_ivp


def _repo_root() -> Path:
    """Repo root = folder containing configs/, scripts/, models/, notebooks/."""
    return Path(__file__).resolve().parents[1]

def _resolve_under_repo(p: str | Path) -> Path:
    """If p is relative, interpret it relative to repo root."""
    p = Path(p)
    return p if p.is_absolute() else (_repo_root() / p)

def _run_one_worker(args):
    """
    Picklable worker for Windows multiprocessing.
    args = (cfg, dose, interval, out_root, folder_template, overwrite)
    """
    cfg, dose, interval, out_root, folder_template, overwrite = args

    folder = folder_template.format(dose_mgkg=dose, interval_weeks=interval)
    out_dir = out_root / folder
    summary_path = out_dir / "run_summary.json"

    if summary_path.exists() and not overwrite:
        with open(summary_path, "r", encoding="utf-8") as f:
            return json.load(f)

    return run_one(cfg, dose, interval, out_dir)


def load_config(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def format_run_folder(template: str, dose_mgkg: float, interval_weeks: int) -> str:
    return template.format(dose_mgkg=dose_mgkg, interval_weeks=interval_weeks)

def make_last_interval_grid(t_start: float, t_end: float, dt: float) -> np.ndarray:
    # Fixed output grid; solver remains adaptive internally.
    n = int(np.floor((t_end - t_start) / dt))
    t = t_start + dt * np.arange(n + 1)
    if t[-1] < t_end - 1e-12:
        t = np.append(t, t_end)
    else:
        t[-1] = t_end
    return t

    
def run_one(config: Dict[str, Any], dose_mgkg: float, interval_weeks: int, out_dir: Path) -> Dict[str, Any]:
    model_module = importlib.import_module(config["model"]["module"])
    params = dict(model_module.DEFAULTS)
    params.update(config.get("params", {}))
    model_module.validate_params(params)

    sim = config["simulation"]
    n_doses = int(sim["n_doses"])
    if n_doses != 25:
        raise ValueError("This workflow is locked to n_doses=25.")
    start_days = float(sim.get("start_days", 0.0))

    method = sim["solver"]["method"]
    rtol = float(sim["solver"]["rtol"])
    atol = float(sim["solver"]["atol"])
    dt_days = float(sim["dt_days"])

    Iota_days = int(interval_weeks) * 7.0

    y = model_module.initial_conditions(params)

    # Required model hook
    if not hasattr(model_module, "apply_dose"):
        raise AttributeError(
            f"Model {config['model']['module']} must define apply_dose(y, dose_mgkg, params)."
        )

    dose_times = [start_days + k * Iota_days for k in range(n_doses)]
    t_end = start_days + n_doses * Iota_days
    t_last_start = t_end - Iota_days

    t_last = None
    y_last = None

    for k, t_dose in enumerate(dose_times):
        # Apply dose as discrete event at interval start
        y = y.copy()
        y = model_module.apply_dose(y, dose_mgkg, params)

        t0 = t_dose
        t1 = min(t_dose + Iota_days, t_end)
        is_last = (t0 >= t_last_start - 1e-12)

        t_eval = make_last_interval_grid(t0, t1, dt_days) if is_last else None

        sol = solve_ivp(
            fun=lambda t, yy: model_module.rhs(t, yy, params),
            t_span=(t0, t1),
            y0=y,
            method=method,
            t_eval=t_eval,
            rtol=rtol,
            atol=atol,
        )
        if not sol.success:
            raise RuntimeError(f"Solver failed on interval {k+1}/{n_doses}: {sol.message}")

        # state at end of interval
        y = sol.y[:, -1]

        # store last-interval trajectory only
        if is_last:
            t_last, y_last = sol.t, sol.y

    # After all intervals: compute derived on last interval
    if t_last is None or y_last is None:
        raise RuntimeError("Last interval trajectory was not captured (unexpected).")

    derived = model_module.derived(t_last, y_last, params)

    outputs = config.get("outputs", {})
    pk_key = outputs.get("pk_key", "Central_ugml")
    pd_key = outputs.get("pd_key", None)

    if pk_key not in derived:
        raise KeyError(f"Derived output '{pk_key}' not found. Available: {list(derived.keys())}")
    Central = np.asarray(derived[pk_key], dtype=float)

    C_trough = float(Central[-1])
    C_avg = float(np.trapz(Central, t_last) / Iota_days)

    summary = {
        "C_trough_ugml": C_trough,
        "C_avg_ugml": C_avg,
        "t_end_days": float(t_end),
        "dose_mgkg": float(dose_mgkg),
        "interval_weeks": int(interval_weeks),
    }

    if pd_key is not None:
        if pd_key not in derived:
            raise KeyError(f"Derived output '{pd_key}' not found. Available: {list(derived.keys())}")
        pd_series = np.asarray(derived[pd_key], dtype=float)
        summary[f"{pd_key}_trough"] = float(pd_series[-1])

    out_dir.mkdir(parents=True, exist_ok=True)

    # Write H5 (last interval only)
    if config["output"].get("write_run_h5", True):
        with h5py.File(out_dir / "run.h5", "w") as f:
            f.create_dataset("t_days", data=t_last)
            f.create_dataset("y", data=y_last)

            dgrp = f.create_group("derived")
            for k, v in derived.items():
                if v is None:
                    continue
                arr = np.asarray(v)
                if arr.shape == ():  # scalar
                    continue
                dgrp.create_dataset(k, data=arr)

            f.attrs["model_name"] = config["model"]["name"]
            f.attrs["dose_mgkg"] = float(dose_mgkg)
            f.attrs["interval_weeks"] = int(interval_weeks)
            f.attrs["interval_days"] = float(Iota_days)
            f.attrs["n_doses"] = int(n_doses)
            f.attrs["solver_method"] = str(method)
            f.attrs["rtol"] = float(rtol)
            f.attrs["atol"] = float(atol)
            f.attrs["dt_days"] = float(dt_days)
            f.attrs["pk_key"] = str(pk_key)
            if pd_key is not None:
                f.attrs["pd_key"] = str(pd_key)

    if config["output"].get("write_config_json", True):
        cfg_out = {
            "model": config["model"],
            "simulation": config["simulation"],
            "params": params,
            "outputs": outputs,
            "run_params": {"dose_mgkg": float(dose_mgkg), "interval_weeks": int(interval_weeks)},
        }
        (out_dir / "run_config.json").write_text(json.dumps(cfg_out, indent=2), encoding="utf-8")

    if config["output"].get("write_summary_json", True):
        (out_dir / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    return summary



import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed

def run_sweep(
    config_path,
    dose_values,
    interval_values,
    workers=1,
    overwrite=False,
):
    """
    Run a full sweep over dose_mgkg x interval_weeks.
    """
    cfg = load_config(_resolve_under_repo(config_path))

    # Resolve output root robustly (relative paths become repo-root relative)
    out_root = _resolve_under_repo(cfg["output"]["root_dir"])
    out_root.mkdir(parents=True, exist_ok=True)

    folder_template = cfg["output"]["folder_template"]

    combos = [(float(d), int(i)) for d in dose_values for i in interval_values]
    tasks = [(cfg, d, i, out_root, folder_template, overwrite) for d, i in combos]

    results = []

    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futures = [ex.submit(_run_one_worker, t) for t in tasks]
            for fut in as_completed(futures):
                results.append(fut.result())
    else:
        for t in tasks:
            results.append(_run_one_worker(t))

    return pd.DataFrame(results)



def run_reference(config_path: str | Path, dose_mgkg: float = 1.0, interval_weeks: int = 2) -> Dict[str, Any]:
    cfg = load_config(_resolve_under_repo(config_path))
    out_root = _resolve_under_repo(cfg["output"]["root_dir"])
    folder = format_run_folder(cfg["output"]["folder_template"], dose_mgkg, interval_weeks)
    out_dir = out_root / folder
    return run_one(cfg, dose_mgkg, interval_weeks, out_dir)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/Kapitanov_sweep.yaml")
    p.add_argument("--dose_mgkg", type=float, default=1.0)
    p.add_argument("--interval_weeks", type=int, default=2)
    args = p.parse_args()
    summary = run_reference(args.config, args.dose_mgkg, args.interval_weeks)
    print(json.dumps(summary, indent=2))