import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def plot_org_vs_intervention_all_methods2(results_org_overall, results_org_races, results_intervention_overall,
                                          results_intervention_races,
                                          start_date, end_date, title, model, diff_start_week=0):
    Nk = pd.read_csv(ROOT / 'data' / 'regions' / region / 'demographic' / 'pop.csv')['population'].values
    df = pd.read_csv(ROOT / 'data' / 'regions' / region / 'epidemic' / 'weekly_deaths.csv')
    date_real = df.loc[(df["date"] >= start_date) & (df["date"] <= end_date)]["date"].values
    results_org_races = np.array(results_org_races)
    results_intervention_races = np.array(results_intervention_races)

    method_names = ['Population-based', 'Death-rate-based', 'Theil-index-based',
                    r'Combined ($\gamma$=0.5)']  # r'Combined ($\gamma$=1)',
    group_names = ['Overall', 'White', 'Hispanic', 'Black', 'Asian']

    differences = {method: {} for method in method_names}

    for method_idx, method_name in enumerate(method_names):
        # Overall difference
        # diff_overall = (
        #     (sum(np.quantile(results_org_overall[method_idx] / np.sum(Nk) * 100000, axis=0, q=0.5)) -
        #      sum(np.quantile(results_intervention_overall[method_idx] / np.sum(Nk) * 100000, axis=0, q=0.5))) /
        #     sum(np.quantile(results_org_overall[method_idx] / np.sum(Nk) * 100000, axis=0, q=0.5))
        # )

        # diff_overall = ( sum( np.quantile(results_org_overall[method_idx], q=0.5, axis=0) )
        # - sum( np.quantile(results_intervention_overall[method_idx], q=0.5, axis=0) ) ) / sum( np.quantile(results_org_overall[method_idx], axis=0, q=0.5) )

        # relative reduction is computed only from week diff_start_week onwards
        org_overall = results_org_overall[method_idx][:, diff_start_week:]
        intervention_overall = results_intervention_overall[method_idx][:, diff_start_week:]
        diff_overall = (np.sum(org_overall, axis=1) - np.sum(intervention_overall, axis=1)) / np.sum(org_overall, axis=1)
        print('diff_overall shape:', diff_overall.shape)
        differences[method_name]['Overall'] = diff_overall
        print(
            f"{method_name} - Overall difference: {np.quantile(diff_overall, q=0.5):.4f}[{np.quantile(diff_overall, q=0.025):.4f}, {np.quantile(diff_overall, q=0.975):.4f}]")

        # Races differences
        for race_idx, race_name in enumerate(['White', 'Hispanic', 'Black', 'Asian']):
            # diff_race = ( sum( np.quantile(results_org_races[method_idx][:, race_idx, :], q=0.5, axis=0) )
            #  - sum( np.quantile(results_intervention_races[method_idx][:, race_idx, :], q=0.5, axis=0) ) ) / sum( np.quantile(results_org_races[method_idx][:, race_idx, :], axis=0, q=0.5) )

            org_race = results_org_races[method_idx][:, race_idx, diff_start_week:]
            intervention_race = results_intervention_races[method_idx][:, race_idx, diff_start_week:]
            diff_race = (np.sum(org_race, axis=1) - np.sum(intervention_race, axis=1)) / np.sum(org_race, axis=1)
            print('diff_race shape:', diff_race.shape)

            # diff_race = (
            #     (sum(np.quantile(results_org_races[method_idx][:, race_idx, :] / np.sum(Nk) * 100000, axis=0, q=0.5)) -
            #      sum(np.quantile(results_intervention_races[method_idx][:, race_idx, :] / np.sum(Nk) * 100000, axis=0, q=0.5))) /
            #     sum(np.quantile(results_org_races[method_idx][:, race_idx, :] / np.sum(Nk) * 100000, axis=0, q=0.5))
            # )
            differences[method_name][race_name] = diff_race
            print(
                f"{method_name} - {race_name} difference: {np.quantile(diff_race, q=0.5):.4f}[{np.quantile(diff_race, q=0.025):.4f}, {np.quantile(diff_race, q=0.975):.4f}]")
            # print(f"{method_name} - {race_name} difference: {diff_race:.4f}")
        print()


    diff_data = {}
    for method_name in method_names:
        diff_data[method_name] = {}
        for group_name in group_names:
            diff_array = differences[method_name][group_name]
            median = np.quantile(diff_array, q=0.5)
            ci_lower = np.quantile(diff_array, q=0.025)
            ci_upper = np.quantile(diff_array, q=0.975)

            diff_data[method_name][group_name] = f"{median * 100:.2f}% [{ci_lower * 100:.2f}%, {ci_upper * 100:.2f}%]"

    diff_df = pd.DataFrame(diff_data).T
    diff_df = diff_df[group_names]  # 确保列顺序

    print("\n=== Death Rate Reduction by Method ===")
    print(diff_df.to_string())
    print(
        f"\nNote: The relative deaths computed depend on the time variant. \n The vaccine only began from the second wave! If drawing the two waves together, the relative reduction in deaths are considered the both waves, the denominated is larger!")

    fig, axes = plt.subplots(1, 5, figsize=(25, 5))

    colors_intervention = ["#66C2A5", "#FC8D62", "#8DA0CB", "#E78AC3"]  # 干预方法颜色

    method_names = ['Population-based', 'Death-rate-based', 'Theil-index-based', r'Combined ($\gamma$=0.5)']
    group_names = ['Overall', 'White', 'Hispanic', 'Black', 'Asian']

    for group_idx in range(5):
        ax = axes[group_idx]

        if group_idx == 0:

            max_overall = 85

            ax.fill_between(date_real,
                            np.quantile(results_org_overall[0] / np.sum(Nk) * 100000, axis=0, q=0.025),
                            np.quantile(results_org_overall[0] / np.sum(Nk) * 100000, axis=0, q=0.975),
                            alpha=0.1, color='gray')

            ax.plot(date_real,
                    np.quantile(results_org_overall[0] / np.sum(Nk) * 100000, axis=0, q=0.5),
                    linewidth=2, color='gray', linestyle='-',
                    label='Without intervention')

            for method_idx in range(len(method_names)):
                ax.fill_between(date_real,
                                np.quantile(results_intervention_overall[method_idx] / np.sum(Nk) * 100000, axis=0,
                                            q=0.025),
                                np.quantile(results_intervention_overall[method_idx] / np.sum(Nk) * 100000, axis=0,
                                            q=0.975),
                                alpha=0.1, color=colors_intervention[method_idx])

                ax.plot(date_real,
                        np.quantile(results_intervention_overall[method_idx] / np.sum(Nk) * 100000, axis=0, q=0.5),
                        linewidth=2, color=colors_intervention[method_idx], linestyle='-',
                        label=f'{method_names[method_idx]}')
            ax.set_ylim(0, max_overall)

        else:
            race_idx = group_idx - 1
            max_race = 35

            ax.fill_between(date_real,
                            np.quantile(results_org_races[0][:, race_idx, :] / np.sum(Nk) * 100000, axis=0, q=0.025),
                            np.quantile(results_org_races[0][:, race_idx, :] / np.sum(Nk) * 100000, axis=0, q=0.975),
                            alpha=0.1, color='gray')

            ax.plot(date_real,
                    np.quantile(results_org_races[0][:, race_idx, :] / np.sum(Nk) * 100000, axis=0, q=0.5),
                    linewidth=2, color='gray', linestyle='-',
                    label='Without intervention')

            for method_idx in range(len(method_names)):
                ax.fill_between(date_real,
                                np.quantile(
                                    results_intervention_races[method_idx][:, race_idx, :] / np.sum(Nk) * 100000,
                                    axis=0, q=0.025),
                                np.quantile(
                                    results_intervention_races[method_idx][:, race_idx, :] / np.sum(Nk) * 100000,
                                    axis=0, q=0.975),
                                alpha=0.1, color=colors_intervention[method_idx])

                ax.plot(date_real,
                        np.quantile(results_intervention_races[method_idx][:, race_idx, :] / np.sum(Nk) * 100000,
                                    axis=0, q=0.5),
                        linewidth=2, color=colors_intervention[method_idx], linestyle='-',
                        label=f'{method_names[method_idx]}')
            ax.set_ylim(0, max_race)

        ax.set_title(group_names[group_idx], fontsize=22)
        # ax.set_ylim(0, max_overall + 5)
        ax.grid(True, alpha=0.3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(axis='y', labelsize=20)
        ax.set_xticks(date_real[::(int(len(date_real) / 5))])
        ax.set_xticklabels(date_real[::(int(len(date_real) / 5))], rotation=30, fontsize=16)
        if group_idx == 0:

            from matplotlib.patches import Patch
            from matplotlib.lines import Line2D

            legend_elements = []
            labels = ['Without intervention'] + method_names
            colors = ['gray'] + colors_intervention

            for i, (color, label) in enumerate(zip(colors, labels)):

                element = (Patch(facecolor=color, alpha=0.1, edgecolor=None),
                           Line2D([0], [0], color=color, linewidth=2))
                legend_elements.append(element)

            ax.legend(legend_elements, labels, loc='upper center', fontsize=22,
                      ncol=5, bbox_to_anchor=(2.8, -0.25), frameon=False)


    fig.text(0.06, 0.5, 'Death rate per 100,000', va='center', rotation='vertical', fontsize=20)
    # fig.suptitle(title, fontsize=24, y=1.1)
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.1, left=0.1)
    plt.savefig('./figure5.png', dpi=300, bbox_inches='tight')
    plt.show()
    
region = 'NYC'
results_intervention_overall_list = []
results_intervention_races_list = []
results_org_overall_list = []
results_org_races_list = []

base_path = ROOT / 'code' / 'simulation' / 'results'

higher = '11'
frac_intervention = 0.2
frac_baseline = 0.0
iterations = 1000
seed = 1
slice_start = 0
model = 'assort-hom'
interventions = ['pop_gammaNone', 'death_gammaNone', 'combine_gamma10', 'combine_gamma05']


def load(method, dtype, frac):
    filename = f'higher{higher}_{method}_{dtype}_weekly_deaths_frac{frac}_it{iterations}_seed{seed}_model{model}.npz'
    print(filename)
    data = np.load(base_path / filename)['arr_0']

    if dtype == 'overall':
        return data[:, slice_start:]
    else:  # races
        return data[:, :, slice_start:]


results_intervention_overall_list = [load(m, 'overall', frac_intervention) for m in interventions]
results_intervention_races_list = [load(m, 'races', frac_intervention) for m in interventions]
results_org_overall_list = [load('no_intervention_gammaNone', 'overall', frac_baseline) for _ in interventions]
results_org_races_list = [load('no_intervention_gammaNone', 'races', frac_baseline) for _ in interventions]

plot_org_vs_intervention_all_methods2(results_org_overall_list, results_org_races_list,
                                      results_intervention_overall_list, results_intervention_races_list,
                                      '2020-03-15', '2021-07-05', 'Two waves, mobility increased by 40%',
                                      model + '_two_waves',
                                      diff_start_week=17)  # skip the first wave (17 weeks, 2020-03-15 to 2020-07-05)