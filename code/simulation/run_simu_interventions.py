import pandas as pd
import numpy as np
from SEIR_model_interventions import SEIR
import yaml
import os
from load_config import load_config
import matplotlib.pyplot as plt

def load_config(config_dir):
    config_path = os.path.join(config_dir)
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    for date_field in ['start_org_date', 'start_simu_date', 'end_date', 'end_date_2ndwave']:
        if date_field in config['simulation']:
            config['simulation'][date_field] = pd.to_datetime(config['simulation'][date_field])

    flat_config = {}
    for category, params in config.items():
        if isinstance(params, dict):
            flat_config.update(params)
        else:
            flat_config[category] = params
    return flat_config

def run_cf(strategy: str,
           iterations: int,
           remove_day: int,
           total_remove_frac: float):
    config_dir = './nyc_params0.yaml'
    contact_matrix_dir = '../data/regions/NYC/contacts_matrix/contact_assortative.csv'
    ifr_dir = '../data/regions/NYC/epidemic/IFR_homo.csv'
    config = load_config(config_dir)
    # 加载接触矩阵
    C = np.loadtxt(contact_matrix_dir, delimiter=',')
    ifr = np.loadtxt(ifr_dir, delimiter=',')

    end_date = config['end_date']
    start_org_date = config['start_org_date']
    start_simu_date = config['start_simu_date']
    eps = config['eps']
    mu = config['mu']
    daily_steps = config['daily_steps']
    basin = config['basin']
    seed=1
    np.random.seed(seed)
    posteriors = pd.read_csv("./posteriors/pos/pos_NYC_assortContact_homoIFR_adfffde9-1132-43d4-9cff-80804be1ff3b.csv")
    num_trajectories = iterations
    sample_pos = posteriors.sample(n=num_trajectories, replace=True)


    path_to_data = "../data"
    Nk = pd.read_csv(path_to_data + "/regions/" + basin + "/demographic/pop.csv")['population'].values


    if strategy == 'pop':
        removal_fractions = np.array([0.3268992 , 0.29727174, 0.22684271, 0.14898634, 0])
    elif strategy == 'theil':
        removal_fractions = np.array([0.19134425, 0.35795191, 0.28247547, 0.16822836, 0])
    elif strategy == 'combine':
        removal_fractions = np.array([0.22675884, 0.42130029, 0.23249578, 0.11944509, 0]) # gamma =1
        #removal_fractions = np.array([0.27545968, 0.36098201, 0.22974655, 0.13381176, 0]) # gamma =0.5

    print('target number', int(np.sum(Nk[:-1]) * 0.2))


    results_intervention_overall=[] #2-dimension
    results_org_overall=[]

    results_intervention_races=[] #3-dimension
    results_org_races=[]

    for idx, data in sample_pos.iterrows():
        Delta = data[1]
        R0 = data[2]
        i0 = data[3]

        sample_inter = SEIR(start_org_date=start_org_date,
                       start_simu_date=start_simu_date,
                       end_date=end_date,
                       i0=i0, R0=R0, eps=eps, mu=mu, ifr=ifr,
                       C=C, Delta=Delta, daily_steps=daily_steps,
                       basin=basin, remove_fractions=removal_fractions, remove_day=remove_day, total_num_removal=int(np.sum(Nk[:-1])*total_remove_frac) )

        results_intervention_overall.append(sample_inter[0])
        results_intervention_races.append(sample_inter[2])

        sample_org = SEIR(start_org_date=start_org_date,
                       start_simu_date=start_simu_date,
                       end_date=end_date,
                       i0=i0, R0=R0, eps=eps, mu=mu, ifr=ifr,
                       C=C, Delta=Delta, daily_steps=daily_steps,
                       basin=basin, remove_fractions=np.array([0,0,0,0,0]), remove_day=remove_day, total_num_removal=0 )

        results_org_overall.append(sample_org[0])
        results_org_races.append(sample_org[2])

    np.savez_compressed(f"./simulations/cf_intervention/{strategy}_gamma05_org_overall_weekly_deaths_frac{total_remove_frac}_it{iterations}_seed{seed}.npz", results_org_overall)
    np.savez_compressed(f"./simulations/cf_intervention/{strategy}_gamma05_org_races_weekly_deaths_frac{total_remove_frac}_it{iterations}_seed{seed}.npz", results_org_races)
    np.savez_compressed(f"./simulations/cf_intervention/{strategy}_gamma05_intervention_overall_weekly_deaths_frac{total_remove_frac}_it{iterations}_seed{seed}.npz", results_intervention_overall)
    np.savez_compressed(f"./simulations/cf_intervention/{strategy}_gamma05_intervention_races_weekly_deaths_frac{total_remove_frac}_it{iterations}_seed{seed}.npz", results_intervention_races)


# example for running
# the input for 'strategy' should be 'pop' or 'theil' or 'combine'
run_cf(strategy='combine', iterations=10000, remove_day=1, total_remove_frac = 0.2, gamma = 1)


