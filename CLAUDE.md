# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Research code for the paper "Fairness in infectious disease modeling": a framework for evaluating the fairness of computational epidemic models (COVID-19 in NYC/London, Zika in Colombia, Santiago), and for designing fairness-aware vaccination strategies. It is a collection of Python scripts, not a package: there is no build system, test suite, linter, or requirements file.

Dependencies inferred from imports: `numpy`, `pandas`, `pyyaml`, `matplotlib`, `scipy`, `pyabc` (SEIR calibration), `covasim` + `optuna` (agent-based calibration), `geopandas` (figure 3).

## Running things

All scripts use **paths relative to their own directory** (e.g. `../data/regions/...`), so run them from the directory the script lives in.

- **Compartmental model calibration (ABC-SMC)** — there is no entry script. Write one in `code/calibration/` calling `abcsmc_general.run_calibration(config_dir, contact_matrix_dir, ifr_dir, filename)`. It needs a YAML config (not in the repo) with top-level sections that get flattened into one dict; `simulation` must hold `start_org_date`, `start_simu_date`, `end_date`, and the others must provide `basin`, `eps`, `mu`, `daily_steps`, `R0_min/max`, `Delta_min/max`, `i0_min/max`, `p_stay`, `population_size`, `max_nr_populations`, `max_walltime_hours`, `minimum_epsilon`.
- **Agent-based (Covasim) borough calibration** — run from `code/calibration/`:
  ```
  python agent_based_model_borough.py --borough BX --n_trials 10000 --n_jobs 8 --top_k 1000
  python agent_based_model_borough.py --borough NYC --prog_by_age 0 --sus_age_specific 0 --pop_type random
  ```
  Boroughs: `NYC, BX, BK, MN, QN, SI`. The 4 model variants are the combinations of `prog_by_age`/`sus_age_specific` (0/0 or 1/1) × `pop_type` (`random`/`hybrid`). Outputs go to `covasim_posteriors_borough/{borough}_prog{p}_sus{s}_{pop_type}_{params,loss,trajectories,trajectories_by_age,summary}.*`.
- **Vaccination intervention scenarios** — `code/simulation/run_simu_interventions.py` calls `run_cf(...)` at module level (edit the call at the bottom to change strategy: `'pop' | 'theil' | 'combine'`). Note it currently won't run as-is: it imports a nonexistent `load_config` module, passes an unsupported `gamma` kwarg, expects `./nyc_params0.yaml` and `./posteriors/...` next to it, and writes to `./simulations/cf_intervention/` rather than `results/`.
- **Figures** — `python code/reproduce_figures/figureN_script.py` from any directory: input paths are resolved from the repo root via `ROOT = Path(__file__).resolve().parents[2]`; output images (`figureN.png`; figure 5 writes `figure5_assort-hom_one_wave.png` and `figure5_assort-hom_two_waves.png`) are written to the current directory. Figure 3 reads the Santiago shapefile from `data/shp/shp/santiago.shp`. Figure 4 silently skips (prints "Skipping") any model file it cannot find rather than failing.

## Architecture

### Stochastic SEIR model (`SEIR_model_general.py`)
Metapopulation SEIR over *social groups* (age groups in London; race/ethnicity groups in NYC — last group is "others"), chain-binomial with `daily_steps` sub-steps per day. Key mechanics that span files:
- `functions_general.import_country(basin)` loads `data/regions/{basin}/demographic/pop.csv` (group sizes `Nk`), `restriction/all_reductions.csv` (daily mobility multiplier `all_red`), and `epidemic/weekly_deaths.csv` (`date`, `total`).
- The contact matrix `C` is scaled each day by `all_red` (`imply_reductions`); `beta` is set from `R0` via the spectral radius of the population-weighted initial contact matrix (`get_beta`).
- Deaths = binomial(new recoveries, per-group `ifr`), shifted forward by delay `Delta` days.
- Simulation runs from `start_org_date`; weekly outputs are truncated to start at `start_simu_date`. Returns `[weekly_deaths, weekly_infections, weekly_deaths_by_group, weekly_infections_by_group]`.
- Calibrated parameters are `R0` (continuous), `Delta` and `i0` (discrete); distance is weighted MAPE on total weekly deaths.

**Model variants** are not code branches — they are chosen by input files: contact matrix (London: `contacts_all_locations` = hetero vs `contacts_homo`; NYC: `contact_assortative`/`proportionate`/`suscept`/`uniform`) × IFR (`IFR_hetero`/`IFR_age_adjusted` vs `IFR_homo`). The `filename` label (e.g. `assortContact_homoIFR`) encodes the variant.

### Calibration outputs → downstream consumers
`save_results` writes (with a uuid suffix on every filename):
- `posteriors/pos/pos_{basin}_{label}_{uuid}.csv` — posterior parameter samples (columns read positionally as `Delta, R0, i0` in the simulation script)
- `posteriors/deaths/[age_]weekly_deaths_{basin}_{label}_{uuid}.npz`, `posteriors/infections/...` — trajectories under key `arr_0`; `age_` prefix = by social group
- `calibration_runs/{basin}/{dbs,abc_history}/` — pyabc DB and pickled history (not committed)

Figure scripts and the intervention script hard-code these uuid filenames; regenerating a calibration produces new uuids that must be updated there.

### Interventions (`code/simulation/`)
`SEIR_model_interventions.py` is a copy of the general model plus `remove_fractions`, `remove_day`, `total_num_removal`: on `remove_day`, `total_num_removal * remove_fractions[g]` susceptibles are removed from each group (vaccination). Strategies differ only in the hard-coded `remove_fractions` vector (population-proportional, Theil-index optimized, or combined with weight `gamma`). It also calls `np.random.seed(0)` internally. Keep the two SEIR files in sync when changing shared dynamics.

Result files in `code/simulation/results/` come in two naming schemes: `{strategy}_{org|intervention}_{overall|races}_weekly_deaths_frac{f}_it{n}_seed{s}.npz` (written by the current simulation script), and `higher11_{pop|death|combine}_gamma{None|10|05}_{overall|races}_weekly_deaths_frac{f}_it1000_seed1_modelassort-hom.npz` plus a `no_intervention_gammaNone` frac0.0 baseline, 69 weeks from 2020-03-15 (two waves). `figure5_script.py` reads the `higher11_*` files; these were produced by a version of the simulation code not in this repo (it has a `death` strategy and a second-wave mobility increase).

### Agent-based model (`agent_based_model_borough.py`)
Covasim sim per borough with custom age data injected into `cv.data.country_age_data`, mobility applied by scaling `beta` daily (`MobilityIntervention`), and `FlatPrognoses`/`FlatSusceptibility` interventions to remove age heterogeneity. Calibrated with Optuna (MSE on weekly deaths, params `beta`, `pop_infected`, `rel_death`); the top-k trials are re-run to form an approximate posterior.

### Fairness metrics
Implemented inside figure scripts, not a shared module: Theil index decomposition across groups (`calculate_contribution_theil_index`, figures 2–4) for unordered groups, and concentration curves / relative concentration index (`compute_rci`, `concentration_curve`, figures 1 and 4) for ordered groups (age/income). Duplicated implementations exist across figure scripts.
