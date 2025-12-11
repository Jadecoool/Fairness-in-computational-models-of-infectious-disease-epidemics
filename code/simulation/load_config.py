import os
import yaml
import pandas as pd
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
    for date_field in ['start_org_date', 'start_simu_date', 'end_date', 'end_date_2ndwave']:
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