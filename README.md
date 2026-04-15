# PK/PD Workflow Automation

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/badge/license-MIT-lightgrey)

A lightweight Python workflow to run repeated-dose PK/PD ODE models,
perform dose/interval sweeps, and visualize exposure-response or full
timecourses.

## Quick start (2 minutes)

``` bash
conda env create -f environment.yml
conda activate ode-sim
jupyter lab
```

Or:

``` bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run a test simulation:

``` bash
python scripts/runner.py --config configs/Dupi\_sweep.yaml --dose\_mgkg 5 --interval\_weeks 4
```

## Core idea

The workflow separates:

* **Models** → biology + ODEs (`models/<NAME>.py`)
* **Configs** → parameters + solver (`configs/<NAME>.yaml`)
* **Runner** → execution + storage (`scripts/runner.py`)

Everything flows through the runner.

## Repository structure

```
PK-PD-Automation/
   ├ configs/     # YAML configs per model
   ├ models/      # PK/PD model implementations
   ├ scripts/     # runner.py (core engine)
   ├ notebooks/   # Sweep, Timecourse, Helper workflows
   └ results/     # auto-generated outputs
```

## Notebooks

Each notebook contains markdown cells and documentation. This is just to provide an overview of the basic possibilities.

### Sweep.ipynb

* Run dose × interval grids
* Generate exposure-response plots
* Compare regimens (q4w, q8w, etc.)

### Timecourse.ipynb

* Simulate full PK/PD trajectories
* Plot PK and optional PD over time
* Supports steady-state and single-dose

### Helper.ipynb

* Inspect existing result folders
* Track parameter sets
* Clean up outdated runs

## How it works

### 1\. Model (`models/<Model>.py`)

Defines: - ODE system (`rhs`) - dosing (`apply\_dose`) - derived outputs
(PK/PD metrics)

### 2\. Config (`configs/<Model>\_sweep.yaml`)

Defines: - model module - parameters - solver settings - output keys

### 3\. Runner (`scripts/runner.py`)

* loads config + model
* builds parameter set
* computes hash
* runs simulation
* saves outputs

## Results structure
```
results/<br>
├─ sweep/<br>
│   └─ params/<br>
│      └─ 5mgkg_q4w/<br>
└─ timecourse/<br>
   └─ params/<br>
      └─ 5mgkg_q4w_n25/<br>
```

Each run contains: - `run.h5` - `run_config.json` - `run_summary.json`


## Add a new model

1. Create `models/<NewModel>.py`
2. Implement required API:

   * DEFAULTS
   * validate_params
   * initial_conditions
   * apply_dose
   * rhs
   * derived
3. Add config `configs/<NewModel>_sweep.yaml`
4. Register in notebooks
5. Run a test simulation

## Notes

* Results are cached by parameter hash
* YAML controls PK/PD outputs
* Timecourse and sweep outputs are separated

## Tip

If something breaks, it's almost always: - wrong `pk\_key` / `pd\_key` -
mismatch between model `derived()` and YAML - or stale cached results

