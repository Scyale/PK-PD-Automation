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
python scripts/runner.py --config configs/Dupi_sweep.yaml --dose_mgkg 5 --interval_weeks 4
```
## Repository structure

```
PK-PD-Automation/
   ├ configs/     # YAML configuration files
   ├ models/      # PK/PD model implementations
   ├ scripts/     # runner.py (core engine)
   ├ notebooks/   # User interface
   └ results/     # auto-generated output folders
```

## Core idea

The workflow uses these four units:

* **Notebooks** → simulation + management of outputs (`notebooks/...ipynb`)
* **Models** → biology + ODEs (`models/...py`)
* **Configurations** → parameters + solver (`configs/...yaml`)
* **Runner** → execution + storage (`scripts/runner.py`)

## Notebooks

Each notebook contains explicit markdown cells and documentation. This is just to provide an overview of the basic possibilities.

### Sweep.ipynb

* Generate exposure-response plots
* Run large dose × interval grids
* Compare E-R of differnet regimens (q4w, q8w, etc.)

### Timecourse.ipynb

* Simulate full PK/PD trajectories
* Plot multiple doses 
* Supports steady-state analysis (n=25)
* As well as single-dose analysis (n=1)
  
### Helper.ipynb

* Manage result folder
* Inspect existing results
* Clean up outdated runs

### Reference.ipynb

* Assists integration of novel models
* Check model notebbok compatibility
* Help debugging 

## How it works

### 1. Models

Define: ODE system (`rhs`), dosing function (`apply_dose`), derived outputs
(PK/PD metrics)

### 2. Configurations

Define: model module, parameters, solver settings, output keys

### 3. Runner

* loads config + model
* builds parameter set
* computes hash
* runs simulation
* saves outputs

## Results structure
```
results/
├─ <ModelName>_sweep/
│  └─ params_<hash>/
│     └─ 5mgkg_q4w/
└─ <ModelName>_timecourse/
   └─ params_<hash>/
      └─ 5mgkg_q4w_n25/
```

Each run contains: `run.h5` - `run_config.json` - `run_summary.json`


## Add a new model

The repo already contains 6 different models. But it is set up in a way that should allow the addition of any new models:

1. Create `models/<NewModel>.py` containing:
   * DEFAULTS
   * validate_params
   * initial_conditions
   * apply_dose
   * rhs
   * derived
2. Create `configs/<NewModel>_sweep.yaml` containing:
   * params
   * simulation & solver settings
   * outputs
3. Register in notebooks
4. Run a test simulation in Reference.ipynb

## Tip

Most LLMs can reliably adapt existing files to the desired new settings.
The easiest way is to simply download one of the existing `.py` and `.yaml` files.
Submit these along with complete documentation of the variables and ODEs for the new model you wish to add.\
**Check for hallucinations**
