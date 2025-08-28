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
    """加载所有配置参数
    Args:
        config_dir (str): 配置文件目录
    Returns:
        dict: 包含所有配置参数的字典
    """
    config_path = os.path.join(config_dir)
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # 处理日期字符串

    for date_field in ['start_org_date', 'start_simu_date', 'end_date']:
        if date_field in config['simulation']:
            config['simulation'][date_field] = pd.to_datetime(config['simulation'][date_field])

    # 展平配置字典
    flat_config = {}
    for category, params in config.items():
        if isinstance(params, dict):
            flat_config.update(params)
        else:
            flat_config[category] = params

    return flat_config

def wmape_pyabc(sim_data: dict, actual_data: dict) -> float:
    """计算WMAPE"""
    return np.sum(np.abs(actual_data['data'] - sim_data['data'])) / np.sum(np.abs(actual_data['data']))

def create_folder(country):
    """创建必要的文件夹"""
    base_path = f"./calibration_runs/{country}"
    if not os.path.exists(base_path):
        os.makedirs(os.path.join(base_path, 'abc_history'), exist_ok=True)
        os.makedirs(os.path.join(base_path, 'dbs'), exist_ok=True)

    # 创建posteriors相关文件夹
    posteriors_dirs = ['pos', 'deaths', 'infections']
    for dir_name in posteriors_dirs:
        os.makedirs(f"./posteriors/{dir_name}", exist_ok=True)


def run_fullmodel(start_org_date, start_simu_date, end_date, i0,
                  R0, eps, mu, ifr, C, Delta, daily_steps, basin):
    """运行完整SEIR模型"""
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
    """运行ABC校准"""
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

    # 初始化ABC
    abc = pyabc.ABCSMC(model, prior, distance,
                       transitions=transition,
                       population_size=population_size,
                       sampler=pyabc.sampler.SingleCoreSampler())

    # 设置数据库
    if db is None:
        db_path = os.path.join(f'./calibration_runs/{basin_name}/dbs/', f"{filename}.db")
        abc.new("sqlite:///" + db_path, {"data": observations})
    else:
        abc.load(db, run_id)

    # 运行ABC
    history = abc.run(minimum_epsilon=minimum_epsilon,
                      max_nr_populations=max_nr_populations,
                      max_walltime=max_walltime)

    # 保存结果
    save_results(history, basin_name, filename)

    return history


def save_results(history, basin_name, filename):
    """保存结果到文件"""
    # 保存历史记录
    # history_path = os.path.join(f'./calibration_runs/{basin_name}/abc_history/',
    #                             f"{filename}.pkl")
    # with open(history_path, 'wb') as file:
    #     pkl.dump(history, file)
    #
    # # 保存各类结果
    # result_types = {
    #     'pos': [lambda h: h.get_distribution()[0], 'csv'],
    #     'deaths/weekly_deaths': [lambda h: np.array([d["data"] for d in h.get_weighted_sum_stats()[1]]), 'npz'],
    #     'infections/weekly_infections': [
    #         lambda h: np.array([d["weekly_infections"] for d in h.get_weighted_sum_stats()[1]]), 'npz'],
    #     'deaths/group_weekly_deaths': [
    #         lambda h: np.array([d["weekly_deaths_by_group"] for d in h.get_weighted_sum_stats()[1]]), 'npz'],
    #     'infections/groop_weekly_infections': [
    #         lambda h: np.array([d["weekly_infections_by_group"] for d in h.get_weighted_sum_stats()[1]]), 'npz']
    # }
    #
    # for key, (data_func, format_type) in result_types.items():
    #     path = f"./posteriors/{key}_{basin_name}_{filename}"
    #     if format_type == 'csv':
    #         data_func(history).to_csv(f"{path}.csv")
    #     else:
    #         np.savez_compressed(f"{path}.npz", data_func(history))
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
        config_dir (str): 配置文件目录
    Returns:
        pyabc.History: ABC校准历史记录
    """
    # 加载所有配置
    config = load_config(config_dir)
    # 加载接触矩阵
    C = np.loadtxt(contact_matrix_dir, delimiter=',')
    ifr = np.loadtxt(ifr_dir, delimiter=',')

    create_folder(config['basin'])

    # 导入国家数据
    country_dict = import_country(config['basin'])

    # 获取真实死亡数据
    deaths_real = country_dict["deaths"].loc[
        (country_dict["deaths"]["date"] >= config['start_simu_date'].strftime('%Y-%m-%d')) &
        (country_dict["deaths"]["date"] <= config['end_date'].strftime('%Y-%m-%d'))
        ]["total"].values

    # 创建prior分布
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

    # 创建转换对象
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

    # 运行校准
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