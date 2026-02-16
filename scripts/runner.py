"""
Runner for ODE workflow sweeps and timecourse runs.

Key ideas:
- Results live under:
    results/<ModelName>_sweep/params_<hash>/...
    results/<ModelName>_timecourse/params_<hash>/...
- Hash depends ONLY on effective model parameters:
    DEFAULTS + cfg.params + param_overrides
  (solver settings are NOT hashed, by design)
- Runs are stored as:
    <dose>mgkg_q<interval>w/              (sweep)
    <dose>mgkg_q<interval>w_n<n>/         (timecourse, to separate n=1 vs n=25)

Notebook usage:
  from scripts.runner import run_sweep, run_reference, run_timecourse_reference
"""

from __future__ import annotations

import sys
import json
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import importlib

import numpy as np
import pandas as pd
import yaml
import h5py
from scipy.integrate import solve_ivp
from concurrent.futures import ProcessPoolExecutor, as_completed


# -------------------------
# Repo path helpers
# -------------------------

def _repo_root() -> Path:
    """Repo root = folder containing configs/, scripts/, models/, notebooks/."""
    return Path(__file__).resolve().parents[1]

def _resolve_under_repo(p: str | Path) -> Path:
    """If p is relative, interpret it relative to repo root."""
    p = Path(p)
    return p if p.is_absolute() else (_repo_root() / p)

# Ensure repo root on sys.path so `models.*` imports work
if str(_repo_root()) not in sys.path:
    sys.path.insert(0, str(_repo_root()))


# -------------------------
# Config / params / hashing
# -------------------------

def load_config(path: str | Path) -> Dict[str, Any]:
    path = _resolve_under_repo(path)
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def _canonicalize(obj):
    """JSON-stable deterministic representation for hashing."""
    if isinstance(obj, dict):
        return {k: _canonicalize(obj[k]) for k in sorted(obj)}
    if isinstance(obj, (list, tuple)):
        return [_canonicalize(x) for x in obj]
    if hasattr(obj, "item") and callable(obj.item):
        try:
            return obj.item()
        except Exception:
            pass
    if isinstance(obj, float):
        return float(f"{obj:.12g}")
    return obj

def _params_fingerprint(params: dict) -> str:
    payload = json.dumps(_canonicalize(params), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]

def _effective_params(cfg: dict, model_module, param_overrides: Optional[dict]) -> dict:
    """Effective params = DEFAULTS + cfg.params + overrides. This is what gets hashed."""
    p = dict(getattr(model_module, "DEFAULTS", {}))
    p.update(cfg.get("params", {}))
    if param_overrides:
        p.update(param_overrides)
    return p

def _base_results_root(cfg: dict, kind: str) -> Path:
    """
    kind: "sweep" or "timecourse"
    Returns: <repo>/results/<ModelName>_<kind>
    """
    model_name = cfg["model"]["name"]
    return _repo_root() / "results" / f"{model_name}_{kind}"

def _param_root_dir(cfg: dict, model_module, param_overrides: Optional[dict], kind: str) -> Path:
    eff = _effective_params(cfg, model_module, param_overrides)
    pid = _params_fingerprint(eff)
    return _base_results_root(cfg, kind) / f"params_{pid}"

def _write_param_manifest(param_root: Path, cfg: dict, eff_params: dict, param_overrides: Optional[dict]) -> None:
    """Write param_set.json once per param folder."""
    param_root.mkdir(parents=True, exist_ok=True)
    manifest_path = param_root / "param_set.json"
    if manifest_path.exists():
        return

    manifest = {
        "model": cfg.get("model", {}),
        "outputs": cfg.get("outputs", {}),
        "param_overrides": param_overrides or {},
        "effective_params": eff_params,
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


# -------------------------
# Run folder + grid helpers
# -------------------------

def format_run_folder(template: str, dose_mgkg: float, interval_weeks: int, n_doses: Optional[int] = None) -> str:
    """
    Folder name for one (dose, interval) run.
    For timecourse, append _n{n_doses} to avoid mixing SD vs SS.
    """
    base = template.format(dose_mgkg=dose_mgkg, interval_weeks=interval_weeks)
    if n_doses is not None:
        base = f"{base}_n{int(n_doses)}"
    return base

def make_grid(t_start: float, t_end: float, dt: float) -> np.ndarray:
    """Fixed output grid for solve_ivp(t_eval=...)."""
    n = int(np.floor((t_end - t_start) / dt))
    t = t_start + dt * np.arange(n + 1)
    if t[-1] < t_end - 1e-12:
        t = np.append(t, t_end)
    else:
        t[-1] = t_end
    return t


# -------------------------
# Skip helper
# -------------------------

def _maybe_load_summary(out_dir: Path, overwrite: bool) -> Optional[Dict[str, Any]]:
    """Return cached summary if present and overwrite=False; else None."""
    summary_path = out_dir / "run_summary.json"
    if summary_path.exists() and not overwrite:
        with open(summary_path, "r", encoding="utf-8") as f:
            d = json.load(f)
        d["_run_dir"] = str(out_dir)
        return d
    return None


# -------------------------
# Simulation core
# -------------------------

def run_one(
    config: Dict[str, Any],
    dose_mgkg: float,
    interval_weeks: int,
    out_dir: Path,
    param_overrides: Optional[Dict[str, Any]] = None,
    store_timecourse: bool = False,
    n_doses_override: Optional[int] = None,
    pk_only: bool = False,
) -> Dict[str, Any]:

    model_module = importlib.import_module(config["model"]["module"])

    # Parameters
    params = dict(getattr(model_module, "DEFAULTS", {}))
    params.update(config.get("params", {}))
    if param_overrides:
        params.update(param_overrides)
    model_module.validate_params(params)

    sim = config["simulation"]
    n_doses_cfg = int(sim["n_doses"])
    start_days = float(sim.get("start_days", 0.0))

    # You want YAML fixed at 25; override from notebook allowed
    if n_doses_cfg != 25:
        raise ValueError("YAML n_doses must remain 25 (override from notebook instead).")

    n_doses = int(n_doses_override) if n_doses_override is not None else n_doses_cfg

    method = sim["solver"]["method"]
    rtol = float(sim["solver"]["rtol"])
    atol = float(sim["solver"]["atol"])
    dt_days = float(sim["dt_days"])

    Iota_days = int(interval_weeks) * 7.0

    if not hasattr(model_module, "apply_dose"):
        raise AttributeError(f"Model {config['model']['module']} must define apply_dose(y, dose_mgkg, params).")

    y = model_module.initial_conditions(params)

    dose_times = [start_days + k * Iota_days for k in range(n_doses)]
    t_end = start_days + n_doses * Iota_days
    t_last_start = t_end - Iota_days

    t_all_list, y_all_list = [], []
    t_last, y_last = None, None

    for k, t_dose in enumerate(dose_times):
        # Discrete dosing event
        y = model_module.apply_dose(y.copy(), dose_mgkg, params)

        t0 = t_dose
        t1 = min(t_dose + Iota_days, t_end)
        is_last = (t0 >= t_last_start - 1e-12)

        if store_timecourse:
            t_eval = make_grid(t0, t1, dt_days)
        else:
            t_eval = make_grid(t0, t1, dt_days) if is_last else None

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

        y = sol.y[:, -1]

        if store_timecourse:
            # drop duplicate boundary point
            if k > 0:
                t_all_list.append(sol.t[1:])
                y_all_list.append(sol.y[:, 1:])
            else:
                t_all_list.append(sol.t)
                y_all_list.append(sol.y)

        if is_last:
            t_last, y_last = sol.t, sol.y

    if t_last is None or y_last is None:
        raise RuntimeError("Last interval trajectory was not captured (unexpected).")

    # Derived: last interval
    derived = model_module.derived(t_last, y_last, params)
    outputs = config.get("outputs", {})
    pk_key = outputs.get("pk_key", "Central_ugml")
    pd_key = outputs.get("pd_key", None)

    if pk_key not in derived:
        raise KeyError(f"Derived output '{pk_key}' not found. Available: {list(derived.keys())}")

    Central_last = np.asarray(derived[pk_key], dtype=float)
    C_trough = float(Central_last[-1])
    C_avg = float(np.trapz(Central_last, t_last) / Iota_days)

    summary: Dict[str, Any] = {
        "C_trough_ugml": C_trough,
        "C_avg_ugml": C_avg,
        "t_end_days": float(t_end),
        "dose_mgkg": float(dose_mgkg),
        "interval_weeks": int(interval_weeks),
        "n_doses": int(n_doses),
    }

    # Full timecourse (optional): compute SD PK metrics from full trajectory when available
    t_full = y_full = derived_full = None
    if store_timecourse:
        t_full = np.concatenate(t_all_list)
        y_full = np.concatenate(y_all_list, axis=1)
        derived_full = model_module.derived(t_full, y_full, params)

        if pk_key in derived_full:
            Central_full = np.asarray(derived_full[pk_key], dtype=float)
            summary["Cmax_ugml"] = float(np.max(Central_full))
            summary["AUC_ugml_day"] = float(np.trapz(Central_full, t_full))

    # PD trough (only for SS mode / when pk_only=False)
    if (not pk_only) and (pd_key is not None):
        if pd_key not in derived:
            raise KeyError(f"Derived output '{pd_key}' not found. Available: {list(derived.keys())}")
        pd_series = np.asarray(derived[pd_key], dtype=float)
        summary[f"{pd_key}_trough"] = float(pd_series[-1])

    # Write outputs
    out_dir.mkdir(parents=True, exist_ok=True)

    if config["output"].get("write_run_h5", True):
        with h5py.File(out_dir / "run.h5", "w") as f:
            # always last interval
            f.create_dataset("t_days", data=t_last)
            f.create_dataset("y", data=y_last)

            dgrp = f.create_group("derived")
            for key, v in derived.items():
                if v is None:
                    continue
                arr = np.asarray(v)
                if arr.shape == ():
                    continue
                dgrp.create_dataset(key, data=arr)

            # optional full timecourse
            if store_timecourse and (t_full is not None) and (y_full is not None):
                f.create_dataset("t_days_full", data=t_full)
                f.create_dataset("y_full", data=y_full)

                dgrp_full = f.create_group("derived_full")
                for key, v in (derived_full or {}).items():
                    if v is None:
                        continue
                    arr = np.asarray(v)
                    if arr.shape == ():
                        continue
                    dgrp_full.create_dataset(key, data=arr)

                f.attrs["store_timecourse"] = True
            else:
                f.attrs["store_timecourse"] = False

            # metadata
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
            f.attrs["pk_only"] = bool(pk_only)

    if config["output"].get("write_config_json", True):
        cfg_out = {
            "model": config["model"],
            "simulation": config["simulation"],
            "params": params,
            "outputs": outputs,
            "run_params": {"dose_mgkg": float(dose_mgkg), "interval_weeks": int(interval_weeks), "n_doses": int(n_doses)},
            "param_overrides": param_overrides or {},
            "pk_only": bool(pk_only),
            "store_timecourse": bool(store_timecourse),
        }
        (out_dir / "run_config.json").write_text(json.dumps(cfg_out, indent=2), encoding="utf-8")

    if config["output"].get("write_summary_json", True):
        (out_dir / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    summary["_run_dir"] = str(out_dir)
    summary["_kind"] = "timecourse" if store_timecourse else "sweep"
    return summary


# -------------------------
# Multiprocessing worker (sweeps)
# -------------------------

def _run_one_worker(args: Tuple[dict, float, int, Path, str, bool, Optional[dict]]) -> Dict[str, Any]:
    cfg, dose, interval, out_root, folder_template, overwrite, param_overrides = args
    out_dir = out_root / folder_template.format(dose_mgkg=dose, interval_weeks=interval)

    cached = _maybe_load_summary(out_dir, overwrite=overwrite)
    if cached is not None:
        cached["_kind"] = "sweep"
        return cached

    return run_one(cfg, dose, interval, out_dir, param_overrides=param_overrides, store_timecourse=False)


# -------------------------
# Public API
# -------------------------

def run_sweep(
    config_path: str | Path,
    dose_values,
    interval_values,
    workers: int = 1,
    overwrite: bool = False,
    param_overrides: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
    """
    Run a sweep over dose_values x interval_values.
    Results: results/<Model>_sweep/params_<id>/<dose>mgkg_q<interval>w/
    """
    cfg = load_config(config_path)
    model_module = importlib.import_module(cfg["model"]["module"])
    eff_params = _effective_params(cfg, model_module, param_overrides)

    out_root = _param_root_dir(cfg, model_module, param_overrides, kind="sweep")
    out_root.mkdir(parents=True, exist_ok=True)
    _write_param_manifest(out_root, cfg, eff_params, param_overrides)

    folder_template = cfg["output"]["folder_template"]
    combos = [(float(d), int(i)) for d in dose_values for i in interval_values]
    tasks = [(cfg, d, i, out_root, folder_template, overwrite, param_overrides) for d, i in combos]

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

def run_reference(
    config_path: str | Path,
    dose_mgkg: float = 1.0,
    interval_weeks: int = 2,
    param_overrides: Optional[Dict[str, Any]] = None,
    overwrite: bool = False,
) -> Dict[str, Any]:
    """
    Single run (last interval stored), stored under the sweep tree.
    """
    cfg = load_config(config_path)
    model_module = importlib.import_module(cfg["model"]["module"])
    eff_params = _effective_params(cfg, model_module, param_overrides)

    out_root = _param_root_dir(cfg, model_module, param_overrides, kind="sweep")
    out_root.mkdir(parents=True, exist_ok=True)
    _write_param_manifest(out_root, cfg, eff_params, param_overrides)

    folder = cfg["output"]["folder_template"].format(dose_mgkg=float(dose_mgkg), interval_weeks=int(interval_weeks))
    out_dir = out_root / folder

    cached = _maybe_load_summary(out_dir, overwrite=overwrite)
    if cached is not None:
        cached["_kind"] = "sweep"
        return cached

    return run_one(cfg, dose_mgkg, interval_weeks, out_dir, param_overrides=param_overrides, store_timecourse=False)

def run_timecourse_reference(
    config_path: str | Path,
    dose_mgkg: float,
    interval_weeks: int,
    param_overrides: Dict[str, Any] | None = None,
    n_doses_override: int | None = None,
    pk_only: bool = False,
    overwrite: bool = False,   # <-- NEW
) -> Dict[str, Any]:
    cfg = load_config(_resolve_under_repo(config_path))
    model_module = importlib.import_module(cfg["model"]["module"])

    eff_params = _effective_params(cfg, model_module, param_overrides)

    out_root = _param_root_dir(cfg, model_module, param_overrides, kind="timecourse")
    out_root.mkdir(parents=True, exist_ok=True)
    _write_param_manifest(out_root, cfg, eff_params, param_overrides)

    n_doses = int(n_doses_override) if n_doses_override is not None else 25

    folder = format_run_folder(cfg["output"]["folder_template"], dose_mgkg, interval_weeks, n_doses=n_doses)
    out_dir = out_root / folder
    summary_path = out_dir / "run_summary.json"

    # If already exists and not overwriting: just load summary and return
    if summary_path.exists() and not overwrite:
        with open(summary_path, "r", encoding="utf-8") as f:
            s = json.load(f)
        s["_run_dir"] = str(out_dir)
        s["_kind"] = "timecourse"
        return s

    return run_one(
        cfg,
        dose_mgkg,
        interval_weeks,
        out_dir,
        param_overrides=param_overrides,
        store_timecourse=True,
        n_doses_override=n_doses,
        pk_only=pk_only,
    )

# -------------------------
# CLI
# -------------------------

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/Kapitanov_sweep.yaml")
    p.add_argument("--dose_mgkg", type=float, default=1.0)
    p.add_argument("--interval_weeks", type=int, default=2)
    p.add_argument("--timecourse", action="store_true")
    args = p.parse_args()

    if args.timecourse:
        summary = run_timecourse_reference(args.config, args.dose_mgkg, args.interval_weeks)
    else:
        summary = run_reference(args.config, args.dose_mgkg, args.interval_weeks)

    print(json.dumps(summary, indent=2))
