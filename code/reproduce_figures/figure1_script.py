import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def generate_synthetic_data(total_people=1000000, base_income=20000, n_groups=10,
                            income_growth=1.2, alpha=1, a=0.4):
    group_sizes = np.array([total_people // n_groups] * n_groups)
    incomes = np.array([base_income * (income_growth ** i) for i in range(n_groups)])
    reference_income = incomes[-1]
    death_rates = a * (reference_income / incomes) ** alpha
    death_numbers = (group_sizes * death_rates / 1000).astype(int)
    actual_death_rate = death_numbers / group_sizes

    print("group id   population  average income")
    for i in range(n_groups):
        print(f"group {i + 1:2d}   {group_sizes[i]:5d}   {incomes[i]:>12,.0f}")
    print('pop', group_sizes)
    print('death rate per 1000', death_rates)
    print('death number', death_numbers)
    print('actual death rate', actual_death_rate)

    return group_sizes, incomes, death_rates, death_numbers, actual_death_rate


def calculate_concentration_curve(group_sizes, death_numbers):
    cumulative_population = np.cumsum(group_sizes) / np.sum(group_sizes)
    health_concentration = np.cumsum(death_numbers) / np.sum(death_numbers)
    cumulative_population = np.concatenate([[0], cumulative_population])
    health_concentration = np.concatenate([[0], health_concentration])
    return cumulative_population, health_concentration


def get_health_age(file_path):
    death_sim = np.load(file_path)["arr_0"]
    total_deaths_of_median_byage_model = []
    for k in range(16):
        age_group_data = death_sim[:, k, :]
        median_values = np.median(age_group_data, axis=0)
        total_median = np.sum(median_values)
        total_deaths_of_median_byage_model.append(total_median)
    return total_deaths_of_median_byage_model


def plot_figure1(output_path='./figure1.png',
                 data_dir=ROOT / 'data',
                 posteriors_dir=ROOT / 'code' / 'calibration' / 'posteriors'):
    group_sizes, incomes, death_rates, death_numbers, actual_death_rate = generate_synthetic_data()
    cumulative_population, health_concentration = calculate_concentration_curve(group_sizes, death_numbers)

    fig = plt.figure(figsize=(13.5, 13))
    gs = GridSpec(2, 2, height_ratios=[1, 1], width_ratios=[1, 1],
                  hspace=0.45, wspace=0.3, left=0.08, right=0.95,
                  bottom=0.05, top=0.95, figure=fig)

    # Panel A - Left
    ax_a1 = fig.add_subplot(gs[0, 0])
    ax_a1.bar(range(len(group_sizes)), actual_death_rate, color='#58b69e', alpha=0.3)
    ax_a1.set_ylabel('Death rate', color='k', fontsize=18)
    ax_a1.set_xticks(range(len(group_sizes)))
    ax_a1.set_xticklabels([f'Decile {i + 1}' for i in range(len(group_sizes))], rotation=45, fontsize=14)
    ax_a1.set_ylim(0, max(actual_death_rate) * 1.2)
    ax_a1.set_xlabel('Equity-related feature', fontsize=18)
    ax_a1.tick_params(axis='y', labelsize=14)
    ax_a1.spines[['right', 'top']].set_visible(False)
    ax_a1.grid(True, alpha=0.3)

    # Panel A - Right
    ax_a2 = fig.add_subplot(gs[0, 1])
    ax_a2.plot([0, 1], [0, 1], '--k', linewidth=1)
    ax_a2.plot(cumulative_population, health_concentration, marker='o', color='k',
               markersize=8, markeredgecolor='k', markerfacecolor='white')

    vertices = [(0, 0)]
    vertices.extend(zip(cumulative_population, health_concentration))
    vertices.append((1, 1))
    vertices.append((0, 0))
    poly = Polygon(vertices, facecolor='#58b69e', alpha=0.3)
    ax_a2.add_patch(poly)

    vertices2 = [(0, 0), (1, 1), (1, 0), (0, 0)]
    poly2 = Polygon(vertices2, facecolor='grey', alpha=0.1)
    ax_a2.add_patch(poly2)

    ax_a2.annotate('Line of equality', xy=(0.5, 0.5), xytext=(0.5, 0.23),
                   arrowprops=dict(arrowstyle='->', lw=1.5, color='k'), fontsize=18, color='k')
    ax_a2.annotate('Relative concentration curve', xy=(0.46, 0.67), xytext=(0.108, 0.9),
                   arrowprops=dict(arrowstyle='->', lw=1.5, color='k'), fontsize=18, color='k')
    ax_a2.text(0.48, 0.59, 'A', transform=ax_a2.transAxes, fontsize=18)
    ax_a2.text(0.63, 0.48, 'B', transform=ax_a2.transAxes, fontsize=18)
    ax_a2.text(0.028, 0.6, r'RCI = $\frac{\mathrm{B-(A+B)}}{\mathrm{B}}$',
               transform=ax_a2.transAxes, fontsize=18)
    ax_a2.set_xlim(0, 1)
    ax_a2.set_ylim(0, 1)
    ax_a2.set_xlabel('Cumulative population proportion', fontsize=18, labelpad=20)
    ax_a2.set_ylabel('Cumulative health outcome proportion', fontsize=18)
    ax_a2.spines[['right', 'top']].set_visible(False)
    ax_a2.tick_params(axis='both', labelsize=14)
    ax_a2.grid(True, alpha=0.3)

    # Panel B
    ax_b = fig.add_subplot(gs[1, 0])
    region = 'London'
    population = pd.read_csv(f"{data_dir}/regions/{region}/demographic/pop.csv")['population'].values
    start_simu_date = '2020-03-15'
    end_date = '2020-07-05'
    df_deaths = pd.read_csv(f"{data_dir}/regions/{region}/epidemic/weekly_deaths.csv")
    death_real = df_deaths.loc[(df_deaths["date"] >= start_simu_date) & (df_deaths["date"] <= end_date)]
    real_death_each_age = death_real.drop(columns=['date', 'total']).sum().values

    age_groups = ['0-4', '5-9', '10-14', '15-19', '20-24', '25-29',
                  '30-34', '35-39', '40-44', '45-49', '50-54', '55-59',
                  '60-64', '65-69', '70-74', '75+']
    x_positions = np.arange(len(age_groups))
    death_rate = np.array(real_death_each_age) / np.array(population) * 1000

    ax_b.bar(x_positions, death_rate, width=0.7, color='dimgrey', alpha=0.6)
    ax_b.set_yscale('log')
    ax_b.set_ylabel('Death rate per 1,000', fontsize=18)
    ax_b.set_xlabel('Age groups', fontsize=18)
    ax_b.set_xticks(x_positions)
    ax_b.set_xticklabels(age_groups, rotation=45)
    ax_b.grid(True, alpha=0.3)
    ax_b.spines[['right', 'top']].set_visible(False)
    ax_b.tick_params(labelsize=14)

    # Panel C
    ax_c = fig.add_subplot(gs[1, 1])
    path = 'deaths'

    hetero_hetero = get_health_age(
        f'{posteriors_dir}/{path}/age_weekly_deaths_London_heteroContact_heteroIFR_a55cc592-5235-401c-851b-e29e34db9ef2.npz')
    hetero_homo = get_health_age(
        f'{posteriors_dir}/{path}/age_weekly_deaths_London_heteroContact_homoIFR_4a8b04ca-999e-427d-acf0-0d0ea6c2edcb.npz')
    homo_hetero = get_health_age(
        f'{posteriors_dir}/{path}/age_weekly_deaths_London_homoContact_heteroIFR_e8caac57-2a67-4412-b604-905fb6d63930.npz')
    homo_homo = get_health_age(
        f'{posteriors_dir}/{path}/age_weekly_deaths_London_homoContact_homoIFR_08682062-de57-4fb0-9064-442c64a4cacd.npz')

    death_simu_models = [homo_homo, homo_hetero, hetero_homo, hetero_hetero]
    labels_model = ['HomC, HomIFR', 'HomC, HetIFR', 'HetC, HomIFR', 'HetC, HetIFR']
    tab20c = plt.cm.tab20c
    colors = [tab20c(1), '#4ecdc4', '#ff6b6b', tab20c(13)]

    df = pd.DataFrame({
        'age_group': [f'{i * 5}-{i * 5 + 4}' if i < 15 else '75+' for i in range(len(population))],
        'population': population,
        'death_counts': real_death_each_age
    })
    df['cum_death_count'] = df['death_counts'].cumsum()
    df['cum_population'] = df['population'].cumsum()
    df['cum_population_pct'] = df['cum_population'] / df['population'].sum()
    df['cum_death_pct'] = df['cum_death_count'] / df['death_counts'].sum()

    ax_c.scatter(df['cum_population_pct'], df['cum_death_pct'], color='k', s=49, facecolor='w', label='Real data')

    for i in range(len(death_simu_models)):
        df_model = pd.DataFrame({
            'age_group': [f'{j * 5}-{j * 5 + 4}' if j < 15 else '75+' for j in range(len(population))],
            'population': population,
            'death_counts': death_simu_models[i]
        })
        df_model['cum_death_count'] = df_model['death_counts'].cumsum()
        df_model['cum_population'] = df_model['population'].cumsum()
        df_model['cum_population_pct'] = df_model['cum_population'] / df_model['population'].sum()
        df_model['cum_death_pct'] = df_model['cum_death_count'] / df_model['death_counts'].sum()
        ax_c.plot(df_model['cum_population_pct'], df_model['cum_death_pct'], marker='o',
                  label=labels_model[i], markersize=7, color=colors[i], linewidth=1.5)

    ax_c.plot([0, 1], [0, 1], '--', color='dimgrey', label='Line of equality', linewidth=1.5)
    ax_c.set_xlabel('Cumulative population proportion', fontsize=18, labelpad=23)
    ax_c.set_ylabel('Cumulative death proportion', fontsize=18)
    ax_c.legend(frameon=False, fontsize=14)
    ax_c.grid(True, alpha=0.3)
    ax_c.spines[['right', 'top']].set_visible(False)
    ax_c.tick_params(labelsize=14)

    fig.text(0.02, 0.97, 'a', va='top', weight='bold', fontsize=26)
    fig.text(0.02, 0.45, 'b', va='top', weight='bold', fontsize=26)
    fig.text(0.51, 0.45, 'c', va='top', weight='bold', fontsize=26)

    separator_y = 0.477
    line = Line2D([0.02, 0.95], [separator_y, separator_y],
                  transform=fig.transFigure, color='grey', linewidth=1.5, linestyle=':')
    fig.add_artist(line)

    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print('Figure 1 finished!')
    return output_path


if __name__ == "__main__":
    plot_figure1()