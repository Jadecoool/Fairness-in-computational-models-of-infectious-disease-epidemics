# Fairness in infectious disease modeling
Code for the paper "Fairness in infectious disease modeling"

# Abstract
Although the concept of fairness has been extensively examined within the domains of machine learning and artificial
intelligence, it remains largely underexplored in the field of computational epidemic modeling. Nevertheless, such models exert
substantial influence on public health policy, particularly in the context of outbreak preparedness and response. Here, we
propose a mathematical framework for evaluating the fairness of computational epidemic models, grounded in core principles
from social epidemiology. We begin by applying our framework to a range of epidemic modeling approaches and simulation
scenarios, including the spread of COVID-19 in New York and the 2016 Zika virus outbreak in Colombia, demonstrating its
consistent capacity to assess model fairness across diverse disease dynamics. Subsequently, we illustrate how our definition
of fairness can be incorporated into the design of immunization strategies to enhance health equity while simultaneously
improving overall effectiveness. Together, our results offer a systematic methodology for quantifying fairness in computational
epidemiology.

# Repository structure
We provide the input data, the code for the models and their calibration, the simulations of vaccine intervention scenarios, and the scripts to reproduce the figures of the paper.

```
├── data
│   ├── regions
│   │   ├── London          # age groups (ordered social groups)
│   │   ├── NYC             # race/ethnicity groups (unordered social groups)
│   │   ├── NYC_BX, NYC_BK, NYC_MN, NYC_QN, NYC_SI   # NYC boroughs
│   │   ├── Santiago        # comunas
│   │   └── Colombia        # 2016 Zika outbreak
│   └── shp                 # Santiago shapefile
└── code
    ├── calibration
    │   ├── posteriors
    │   └── covasim_posteriors_borough
    ├── simulation
    │   └── results
    └── reproduce_figures
```

## Data
The `data/regions` folder contains the model inputs for each region. Depending on the region, it includes:
- `demographic`: population size of each social group (`pop.csv`) and population by age (`pop_age.csv`)
- `contacts_matrix`: contact matrices between social groups (e.g. heterogeneous vs. homogeneous contacts in London; assortative, proportionate, uniform and susceptibility-based contacts in NYC)
- `epidemic`: reported weekly deaths (in total, by race and by age) and infection fatality rates (`IFR_*.csv`)
- `restriction`: daily reductions of contacts derived from mobility data

The NYC borough folders contain population by age and weekly deaths for each borough. The `Santiago` folder contains real and simulated cases by comuna together with the Human Development Index, and `data/shp` contains the corresponding shapefile. The `Colombia` folder contains model outcomes for the 2016 Zika outbreak.

## Code
### Calibration
The `code/calibration` folder contains:
- `SEIR_model_general.py` and `functions_general.py`: the stochastic SLIRD compartmental model with social groups, used for both the ordered social groups in London and the unordered social groups in NYC
- `abcsmc_general.py`: the ABC-SMC calibration of the compartmental model (entry point `run_calibration`, see `code/calibration/readme.md`)
- `agent_based_model_borough.py`: the Covasim agent-based model of NYC and its boroughs, calibrated with Optuna, e.g. `python agent_based_model_borough.py --borough BX --n_trials 10000 --n_jobs 8 --top_k 1000`

The `posteriors` folder contains the posterior distributions of the compartmental models' parameters (`pos`) and the calibrated weekly deaths (`deaths`). The `covasim_posteriors_borough` folder contains the calibrated parameters, losses and death trajectories of the agent-based models.

### Simulation
The `code/simulation` folder contains the code for running scenarios of vaccine interventions (`SEIR_model_interventions.py`, `run_simu_interventions.py`). The `results` folder contains the simulated weekly deaths with and without intervention under different allocation strategies.

### Reproducing the figures
The `code/reproduce_figures` folder contains one script for each figure of the paper. All input paths are resolved relative to the repository, so the scripts can be run from any directory, and the figures are saved to the current directory:
```
python code/reproduce_figures/figure1_script.py
```
- Figure 1: fairness framework and age-related fairness of the models in London
- Figure 2: race/ethnicity-related fairness of the models in NYC (Theil index)
- Figure 3: spatial fairness of the models in Santiago
- Figure 4: fairness of the agent-based models across age groups and NYC boroughs
- Figure 5: vaccine intervention strategies in NYC; the printed relative reduction in deaths is computed over the second wave only

### Libraries
The code requires `numpy`, `pandas`, `scipy`, `matplotlib` and `pyyaml`. In addition, `pyabc` is needed for the ABC-SMC calibration, `covasim` and `optuna` for the agent-based model, and `geopandas` for Figure 3.
