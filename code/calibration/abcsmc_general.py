import numpy as np
import pyabc
import os
import uuid
import pickle as pkl
from pyabc import RV, Distribution
from typing import Callable, List
from datetime import datetime, timedelta
from functions_general import import_country
from SEIR_model_general import SEIR
import pandas as pd
import yaml

def load_config(config_dir):
    """loading config
    Args:
        config_dir (str): file path
    Returns:
        dict: dictionary including all config
    """
    config_path = os.path.join(config_dir)
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # dealing with date string

    for date_field in ['start_org_date', 'start_simu_date', 'end_date']:
        if date_field in config['simulation']:
            config['simulation'][date_field] = pd.to_datetime(config['simulation'][date_field])

    # flatten
    flat_config = {}
    for category, params in config.items():
        if isinstance(params, dict):
            flat_config.update(params)
        else:
            flat_config[category] = params

    return flat_config

def wmape_pyabc(sim_data: dict, actual_data: dict) -> float:
    return np.sum(np.abs(actual_data['data'] - sim_data['data'])) / np.sum(np.abs(actual_data['data']))

def create_folder(country):
    """make sure folders exist"""
    base_path = f"./calibration_runs/{country}"
    if not os.path.exists(base_path):
        os.makedirs(os.path.join(base_path, 'abc_history'), exist_ok=True)
        os.makedirs(os.path.join(base_path, 'dbs'), exist_ok=True)

    posteriors_dirs = ['pos', 'deaths', 'infections']
    for dir_name in posteriors_dirs:
        os.makedirs(f"./posteriors/{dir_name}", exist_ok=True)


def run_fullmodel(start_org_date, start_simu_date, end_date, i0,
                  R0, eps, mu, ifr, C, Delta, daily_steps, basin):
    results = SEIR(start_org_date=start_org_date,
                   start_simu_date=start_simu_date,
                   end_date=end_date,
                   i0=i0, R0=R0, eps=eps, mu=mu, ifr=ifr,
                   C=C, Delta=Delta, daily_steps=daily_steps,
                   basin=basin)

    return results[0], results[1], results[2], results[3]


def calibration(epimodel: Callable,
                prior: pyabc.Distribution,
                params: dict,
                distance: Callable,
                observations: List[float],
                basin_name: str,
                transition: pyabc.AggregatedTransition,
                max_walltime: timedelta = None,
                population_size: int = 1000,
                minimum_epsilon: float = 0.15,
                max_nr_populations: int = 10,
                filename: str = '',
                run_id=None,
                db=None):
    """ABC-SMC calibration"""
    filename = filename+'_'+str(uuid.uuid4())
    print(f"Using filename: {filename}")

    def model(p):
        weekly_deaths, weekly_infections, deaths_by_group, infections_by_group = epimodel(**p, **params)
        return {
            'data': weekly_deaths,
            'weekly_infections': weekly_infections,
            'weekly_deaths_by_group': deaths_by_group,
            'weekly_infections_by_group': infections_by_group
        }

    # Initialize ABC-SMC
    abc = pyabc.ABCSMC(model, prior, distance,
                       transitions=transition,
                       population_size=population_size,
                       sampler=pyabc.sampler.SingleCoreSampler())

    # Settting database
    if db is None:
        db_path = os.path.join(f'./calibration_runs/{basin_name}/dbs/', f"{filename}.db")
        abc.new("sqlite:///" + db_path, {"data": observations})
    else:
        abc.load(db, run_id)

    # run ABC
    history = abc.run(minimum_epsilon=minimum_epsilon,
                      max_nr_populations=max_nr_populations,
                      max_walltime=max_walltime)

    # save results
    save_results(history, basin_name, filename)

    return history


def save_results(history, basin_name, filename):
    with open(os.path.join(f'./calibration_runs/{basin_name}/abc_history/', f"{basin_name}_{filename}.pkl"), 'wb') as file:
        pkl.dump(history, file)

    history.get_distribution()[0].to_csv(f"./posteriors/pos/pos_{basin_name}_{filename}.csv")

    np.savez_compressed(f"./posteriors/{basin_name}_{filename}.npz",
                        np.array([d["data"] for d in history.get_weighted_sum_stats()[1]]))

    np.savez_compressed(f"./posteriors/deaths/weekly_deaths_{basin_name}_{filename}.npz",
                        np.array([d["data"] for d in history.get_weighted_sum_stats()[1]]))

    np.savez_compressed(f"./posteriors/infections/weekly_infections_{basin_name}_{filename}.npz",
                        np.array([d["weekly_infections"] for d in history.get_weighted_sum_stats()[1]]))

    # age-specific deaths
    np.savez_compressed(f"./posteriors/deaths/age_weekly_deaths_{basin_name}_{filename}.npz",
                        np.array([d["weekly_deaths_by_group"] for d in history.get_weighted_sum_stats()[1]]))

    # age-specific infections
    np.savez_compressed(f"./posteriors/infections/age_weekly_infections_{basin_name}_{filename}.npz",
                        np.array([d["weekly_infections_by_group"] for d in history.get_weighted_sum_stats()[1]]))


def run_calibration(config_dir: str, contact_matrix_dir: str, ifr_dir: str, filename: str):
    """运行校准流程
    Args:
        config_dir (str)
    Returns:
        pyabc.History: ABC calibration history
    """
    # load config
    config = load_config(config_dir)
    # load contact matrix
    C = np.loadtxt(contact_matrix_dir, delimiter=',')
    ifr = np.loadtxt(ifr_dir, delimiter=',')

    create_folder(config['basin'])

    # load country data
    country_dict = import_country(config['basin'])

    #  load real deaths
    deaths_real = country_dict["deaths"].loc[
        (country_dict["deaths"]["date"] >= config['start_simu_date'].strftime('%Y-%m-%d')) &
        (country_dict["deaths"]["date"] <= config['end_date'].strftime('%Y-%m-%d'))
        ]["total"].values

    # Prior
    prior = Distribution(
        R0=RV("uniform", config['R0_min'], config['R0_max'] - config['R0_min']),
        Delta=RV('rv_discrete', values=(
            np.arange(config['Delta_min'], config['Delta_max']),
            [1. / (config['Delta_max'] - config['Delta_min'])] * (config['Delta_max'] - config['Delta_min'])
        )),
        i0=RV('rv_discrete', values=(
            np.arange(config['i0_min'], config['i0_max']),
            [1. / (config['i0_max'] - config['i0_min'])] * (config['i0_max'] - config['i0_min'])
        ))
    )

    transition = pyabc.AggregatedTransition(
        mapping={
            'R0': pyabc.MultivariateNormalTransition(),
            'Delta': pyabc.DiscreteJumpTransition(
                domain=np.arange(config['Delta_min'], config['Delta_max']),
                p_stay=config['p_stay']
            ),
            'i0': pyabc.DiscreteJumpTransition(
                domain=np.arange(config['i0_min'], config['i0_max']),
                p_stay=config['p_stay']
            ),
        }
    )

    # run calibration
    history = calibration(
        run_fullmodel,
        prior=prior,
        params={
            'end_date': config['end_date'],
            'start_org_date': config['start_org_date'],
            'start_simu_date': config['start_simu_date'],
            'eps': config['eps'],
            'mu': config['mu'],
            'ifr': ifr,
            'C': C,
            'daily_steps': config['daily_steps'],
            'basin': config['basin'],
        },
        distance=wmape_pyabc,
        basin_name=config['basin'],
        observations=deaths_real,
        transition=transition,
        max_nr_populations=config['max_nr_populations'],
        population_size=config['population_size'],
        max_walltime=timedelta(hours=config['max_walltime_hours']),
        minimum_epsilon=config['minimum_epsilon'],
        filename=filename
    )

    return history
