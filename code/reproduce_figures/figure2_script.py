import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


def get_sample_total_deaths_by_race(death_sim):
    total_deaths = death_sim.sum(axis=2)
    return total_deaths[:, :4]


def calculate_contribution_theil_index(samples_total_deaths_by_race, populations):
    deaths = np.array(samples_total_deaths_by_race, dtype=float)
    if deaths.ndim == 1:
        deaths = deaths.reshape(1, -1)

    populations = np.array(populations, dtype=float)
    population_shares = populations / populations.sum()
    death_shares = deaths / deaths.sum(axis=1)[:, np.newaxis]
    ratios = death_shares / population_shares
    contributions = np.where(death_shares > 0, death_shares * np.log(ratios), 0)
    return contributions


def plot_figure2(output_path='./figure2.png',
                 data_dir='/data',
                 posteriors_dir='/code/calibration/posteriors'):
    data = pd.read_csv(f'{data_dir}/regions/NYC/epidemic/weekly_death_byrace.csv', index_col=False)
    data3_date = data.loc[(data['date'] >= '2020-03-14') & (data['date'] <= '2020-07-05')]
    data3_date = data3_date.copy()
    data3_date['Other'] = (data3_date['all_race_ethnicity'] - data3_date['White'] -
                           data3_date['Hispanic_Latino'] - data3_date['Black_African_American'] -
                           data3_date['Asian_Pacific_Islander'])

    real_deaths_byrace = data3_date[['all_race_ethnicity', 'White', 'Hispanic_Latino',
                                     'Black_African_American', 'Asian_Pacific_Islander', 'Other']].sum()
    print('Deaths:')
    print(real_deaths_byrace)
    real_deaths_byrace = real_deaths_byrace[1:-1]
    print(real_deaths_byrace)

    baseline_deaths = np.load(
        f"{posteriors_dir}/deaths/age_weekly_deaths_NYC_uniformCandSuscept_homoIFR_d9d1621f-35c1-403f-83cc-8245ee013dab.npz")[
        "arr_0"]
    baseline_totals = get_sample_total_deaths_by_race(baseline_deaths)

    pop = pd.read_csv(f'{data_dir}/regions/NYC/demographic/pop.csv')['population'].values
    print(pop)
    print('Population:')
    for i, groupname in enumerate(
            ['White', 'Hispanic_Latino', 'Black_African_American', 'Asian_Pacific_Islander', 'Other']):
        print(groupname, pop[i])
    pop = pop[:-1]

    contribution_real_data = calculate_contribution_theil_index(real_deaths_byrace.values, pop)
    race_names = ['White', 'Hispanic', 'Black', 'Asian']
    model_names = ['Baseline', 'Variable susceptibility', 'Assortative contacts']

    baseline_contribution = calculate_contribution_theil_index(baseline_totals, pop)

    model4_deaths = np.load(
        f"{posteriors_dir}/deaths/age_weekly_deaths_NYC_assortContact_homoIFR_adfffde9-1132-43d4-9cff-80804be1ff3b.npz")[
        "arr_0"]
    model3_deaths = \
    np.load(f"{posteriors_dir}/deaths/age_weekly_deaths_NYC_Suscept_homoIFR_f21366a4-dae6-49a0-baf9-cb20187199fa.npz")[
        "arr_0"]

    model3_totals = get_sample_total_deaths_by_race(model3_deaths)
    model4_totals = get_sample_total_deaths_by_race(model4_deaths)

    models_contributions = [baseline_contribution]
    for output in [model3_totals, model4_totals]:
        models_contributions.append(calculate_contribution_theil_index(output, pop))

    n_races = len(race_names)
    n_rows, n_cols = 1, 3
    positions = np.arange(n_races)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 5))

    for idx, model_contrib in enumerate(models_contributions):
        col = idx % n_cols
        ax = axes[col]

        box_data = []
        for j in range(n_races):
            q1, median, q2 = np.percentile(model_contrib[:, j], [5, 50, 95])
            min_val = np.min(model_contrib[:, j])
            max_val = np.max(model_contrib[:, j])
            stats = {
                'med': median, 'q1': q1, 'q3': q2,
                'whislo': min_val, 'whishi': max_val, 'fliers': []
            }
            box_data.append(stats)

        F4_2 = np.exp(-sum(np.absolute(np.percentile(model_contrib, 50, axis=0) - contribution_real_data[0])))
        print('--------')
        print('Fairness score:', round(F4_2, 3))

        bplot = ax.bxp([box_data[j] for j in range(n_races)],
                       positions=positions, widths=0.6, patch_artist=True,
                       medianprops=dict(color='black'),
                       boxprops=dict(facecolor='#58b69e', alpha=0.6),
                       showfliers=False, zorder=1)

        p1 = ax.scatter(positions, contribution_real_data[0], color='darkgrey', marker='o', s=90, zorder=2)

        ax.set_xticks(positions)
        ax.set_xticklabels(race_names, rotation=45, fontsize=16)
        ax.grid(True, linestyle='-', alpha=0.3)
        ax.tick_params(axis='y', labelsize=18)
        ax.set_ylim(-0.1, 0.18)
        ax.spines[['right', 'top']].set_visible(False)
        ax.text(0.86, 0.94, f"F={round(F4_2, 3)}",
                ha='center', va='center', fontsize=18,
                bbox=dict(boxstyle="square", facecolor='lightgrey', alpha=0.2, edgecolor='grey'),
                transform=ax.transAxes)
        ax.set_title(model_names[idx], fontsize=20, pad=15)

    box_patch = Patch(facecolor='#58b69e', alpha=0.6)
    fig.legend([box_patch, p1], ["Model outcome", "Real data"],
               loc='center', bbox_to_anchor=(0.5, -0.14), ncol=3, frameon=False, fontsize=18)
    axes[0].set_ylabel('Components of Theil index', fontsize=20)
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.1, hspace=0.4, wspace=0.35)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print('Figure 2 finished!')
    return output_path


if __name__ == "__main__":
    plot_figure2()