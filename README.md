# PK-PD-Automation

This repository provides an automated Python workflow to run PK/PD models, adjust configurations, and generate parameter plots. It includes a reusable sweep runner, two example models (Walz and Kapitanov), and notebooks that demonstrate end-to-end usage.

## Repository layout

- `configs/`: Model + solver configuration YAMLs (e.g., `Walz_sweep.yaml`, `Kapitanov_sweep.yaml`).
- `models/`: PK/PD model implementations with a common interface (`DEFAULTS`, `validate_params`, `initial_conditions`, `rhs`, `derived`, `apply_dose`).
- `scripts/run_sweep.py`: Main runner used by both scripts and notebooks.
- `notebooks/`: Interactive workflows for running sweeps and validating references.
- `results/`: Output root for generated runs (created on first run). The exact subfolder names are controlled by each config's `output.root_dir` and `output.folder_template`.

## Quickstart (local)

1. Create the conda environment:
   ```bash
   conda env create -f environment.yml
   conda activate ode-sim
   ```
2. Run a reference simulation:
   ```bash
   python scripts/run_sweep.py --config configs/Kapitanov_sweep.yaml --dose_mgkg 1.0 --interval_weeks 2
   ```
3. Open notebooks (optional):
   ```bash
   jupyter lab
   ```

## Quickstart (GitHub Codespaces)

This repository includes a devcontainer configuration to set up a Codespace with the required conda environment.

1. Open the repo on GitHub and click **Code → Codespaces → Create codespace**.
2. On first start, the Codespace will create the `ode-sim` conda environment from `environment.yml`.
3. Activate the environment in your terminal (if not already active):
   ```bash
   conda activate ode-sim
   ```
4. Run scripts or notebooks as in the local quickstart.

### Where outputs are stored

All outputs are written under the `results/` directory at the repo root. Each configuration sets the output root and folder template:

- `configs/Walz_sweep.yaml` writes under `results/Walz_sweep/{dose_mgkg}mgkg_q{interval_weeks}w/`.
- `configs/Kapitanov_sweep.yaml` writes under `results/Kapitanov_sweep/{dose_mgkg}mgkg_q{interval_weeks}w/`.

Each run folder includes:
- `run.h5` (HDF5 data),
- `run_config.json` (resolved configuration),
- `run_summary.json` (summary metrics).

## Configuration notes

The YAML config controls:
- `model.module`: Python import path for the model (e.g., `models.Walz`).
- `simulation.solver`: `solve_ivp` method and tolerances.
- `outputs.pk_key` / `outputs.pd_key`: Keys used when computing summary metrics.

See `configs/*.yaml` for examples.
