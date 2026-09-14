import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from pathlib import Path


def plot_org_vs_intervention(results_org_overall, results_org_races,
                             results_intervention_overall, results_intervention_races,
                             start_date, end_date, data_dir='/data', region='NYC'):
    Nk = pd.read_csv(f'{data_dir}/regions/{region}/demographic/pop.csv')['population'].values
    df = pd.read_csv(f"{data_dir}/regions/{region}/epidemic/weekly_deaths.csv")
    date_real = df.loc[(df["date"] >= start_date) & (df["date"] <= end_date)]["date"].values

    results_org_races = np.array(results_org_races)
    results_intervention_races = np.array(results_intervention_races)

    method_names = ['Population based', 'Theil-index based', r'Combined ($\gamma$=0.5)']
    group_names = ['Overall', 'White', 'Hispanic', 'Black', 'Asian']
    colors_intervention = ["#66C2A5", "#FC8D62", "#8DA0CB", "#E78AC3"]

    fig, axes = plt.subplots(1, 5, figsize=(25, 5))

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
                    linewidth=2, color='gray', linestyle='-', label='Without intervention')

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
                    linewidth=2, color='gray', linestyle='-', label='Without intervention')

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
        ax.grid(True, alpha=0.3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(axis='y', labelsize=20)
        ax.set_xticks(date_real[::(int(len(date_real) / 5))])
        ax.set_xticklabels(date_real[::(int(len(date_real) / 5))], rotation=30, fontsize=16)

        if group_idx == 0:
            legend_elements = []
            labels = ['Without intervention'] + method_names
            legend_colors = ['gray'] + colors_intervention
            for color, label in zip(legend_colors, labels):
                element = (Patch(facecolor=color, alpha=0.1, edgecolor=None),
                           Line2D([0], [0], color=color, linewidth=2))
                legend_elements.append(element)
            ax.legend(legend_elements, labels, loc='upper center', fontsize=22,
                      ncol=5, bbox_to_anchor=(2.8, -0.25), frameon=False)

    fig.text(0.06, 0.5, 'Death rate per 100,000', va='center', rotation='vertical', fontsize=20)
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.1, left=0.1)
    return fig


def plot_figure4(output_path='./figure4.png',
                 data_dir='/data',
                 results_dir='/code/simulation/results'):
    base_path = Path(results_dir)
    frac_intervention = 0.2
    frac_baseline = 0.0
    iterations = 1000
    seed = 1
    slice_start = 0
    interventions = ['pop_gammaNone', 'combine_gamma10', 'combine_gamma05']

    def load(method, dtype, frac):
        filename = f'{method}_{dtype}_weekly_deaths_frac{frac}_it{iterations}_seed{seed}.npz'
        data = np.load(base_path / filename)['arr_0']
        if dtype == 'overall':
            return data[:, slice_start:]
        else:
            return data[:, :, slice_start:]

    results_intervention_overall_list = [load(m, 'overall', frac_intervention) for m in interventions]
    results_intervention_races_list = [load(m, 'races', frac_intervention) for m in interventions]
    results_org_overall_list = [load('no_intervention_gammaNone', 'overall', frac_baseline) for _ in interventions]
    results_org_races_list = [load('no_intervention_gammaNone', 'races', frac_baseline) for _ in interventions]

    fig = plot_org_vs_intervention(
        results_org_overall_list, results_org_races_list,
        results_intervention_overall_list, results_intervention_races_list,
        '2020-03-15', '2021-07-05', data_dir=data_dir
    )

    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print('Figure 4 finished!')
    return output_path


if __name__ == "__main__":
    plot_figure4()