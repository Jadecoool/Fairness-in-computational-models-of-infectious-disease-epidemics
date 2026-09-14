import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


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


def plot_figure3(output_path='./figure3.png',
                 data_dir=ROOT / 'data'):
    st = gpd.read_file(f"{data_dir}/shp/shp/santiago.shp")
    df = pd.read_csv(f'{data_dir}/regions/Santiago/simulated_deaths_cases_new.csv')
    df['cases_rate'] = df['real_cases'] / df['popul'] * 1000
    rate_map = dict(zip(df['comuna'], df['cases_rate']))
    st['cases_rate'] = st['Comuna'].map(rate_map)
    st = st[~st['Comuna'].isin(['Lampa', 'Colina'])]

    df_sort = df.sort_values('HDI')
    Nk = df_sort['popul'].values
    case_real = df_sort['real_cases'].values
    case_sim_main = df_sort['simulated_cases'].values
    case_sim_nomob = df_sort['simulated_cases_no_mob'].values

    fig = plt.figure(figsize=(22, 14))
    gs = GridSpec(2, 3, figure=fig, height_ratios=[1, 0.8], width_ratios=[1.1, 1.1, 1.05], hspace=0., wspace=0.45)

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])
    ax4 = fig.add_subplot(gs[1, :])

    # Panel A - HDI map
    norm_hdi = colors.Normalize(vmin=st['HDI'].min(), vmax=st['HDI'].max())
    st.plot(ax=ax1, column='HDI', cmap=plt.cm.GnBu, norm=norm_hdi, edgecolor='black', linewidth=0.5, legend=True,
            legend_kwds={'label': "HDI", 'orientation': 'vertical', 'shrink': 0.7, 'pad': 0.035, 'fraction': 0.08})
    ax1.axis('off')
    legend1 = ax1.get_figure().get_axes()[-1]
    legend1.set_ylabel('HDI', fontsize=20, labelpad=10)
    legend1.tick_params(labelsize=20)
    ax1_pos = ax1.get_position()
    fig.text(ax1_pos.x0, ax1_pos.y0 + ax1_pos.height + 0.017, 'a', fontsize=26, fontweight='bold', ha='left', va='top')

    # Panel B - Cases rate map
    norm_cases = colors.Normalize(vmin=st['cases_rate'].min(), vmax=st['cases_rate'].max())
    st.plot(ax=ax2, column='cases_rate', cmap=plt.cm.Reds, norm=norm_cases, edgecolor='black', linewidth=0.5,
            legend=True,
            legend_kwds={'label': "Infection rate", 'orientation': 'vertical', 'shrink': 0.7, 'pad': 0.000,
                         'fraction': 0.08})
    ax2_pos = ax2.get_position()
    fig.text(ax2_pos.x0 - 0.03, ax2_pos.y0 + ax2_pos.height + 0.013, 'b', fontsize=26, fontweight='bold', ha='left',
             va='top')
    ax2.axis('off')
    legend2 = ax2.get_figure().get_axes()[-1]
    legend2.set_ylabel('Infection rate per 1,000', fontsize=20, labelpad=10)
    legend2.tick_params(labelsize=20)

    # Panel C - Concentration curve
    colors_cc = ['#F88455', '#76CBB4']
    df_cc = pd.DataFrame({
        'age_group': [f'{i * 10}-{(i + 1) * 10}' for i in range(len(Nk))],
        'population': Nk,
        'case_counts': case_real
    })
    df_cc['cum_case_count'] = df_cc['case_counts'].cumsum()
    df_cc['cum_population'] = df_cc['population'].cumsum()
    df_cc['cum_population_pct'] = df_cc['cum_population'] / df_cc['population'].sum()
    df_cc['cum_case_pct'] = df_cc['cum_case_count'] / df_cc['case_counts'].sum()

    area = np.trapz(np.insert(df_cc['cum_case_pct'], 0, 0), np.insert(df_cc['cum_population_pct'], 0, 0))
    rci_real = 1 - 2 * area
    print('Real data RCI:', rci_real)

    ax3.scatter(df_cc['cum_population_pct'], df_cc['cum_case_pct'], color='k', facecolor='w', label='Real data')

    labels_model = ['SLIR with mobility', 'SLIR without mobility']
    model_data = [case_sim_main, case_sim_nomob]

    for i, (model, label) in enumerate(zip(model_data, labels_model)):
        df_m = pd.DataFrame({
            'age_group': [f'{j * 10}-{(j + 1) * 10}' for j in range(len(Nk))],
            'population': Nk,
            'case_counts': model
        })
        df_m['cum_case_count'] = df_m['case_counts'].cumsum()
        df_m['cum_population'] = df_m['population'].cumsum()
        df_m['cum_population_pct'] = df_m['cum_population'] / df_m['population'].sum()
        df_m['cum_case_pct'] = df_m['cum_case_count'] / df_m['case_counts'].sum()

        area = np.trapz(np.insert(df_m['cum_case_pct'].values, 0, 0),
                        np.insert(df_m['cum_population_pct'].values, 0, 0))
        rci = 1 - 2 * area
        fairness_score = 1 - np.abs(rci_real - rci) / 2
        print(f'{labels_model[i]} RCI: {rci}, F score: {fairness_score}')

        ax3.plot(df_m['cum_population_pct'], df_m['cum_case_pct'], marker='o', label=label, color=colors_cc[i],
                 linewidth=1.5)

    ax3.plot([0, 1], [0, 1], '--', color='k', label='Line of equality', linewidth=1.5)
    ax3.set_xlabel('Cumulative population proportion', fontsize=20)
    ax3.set_ylabel('Cumulative case proportion', fontsize=20)
    ax3.legend(frameon=False, fontsize=16, loc='lower right', bbox_to_anchor=(1.05, 0))
    ax3.grid(True, alpha=0.3)
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    ax3.tick_params(labelsize=16)
    ax3_pos = ax3.get_position()
    fig.text(ax3_pos.x0 - 0.07, ax3_pos.y0 + ax3_pos.height - 0.056, 'c', fontsize=26, fontweight='bold', ha='left',
             va='top')
    ax3.set_aspect('equal')

    ax3_original_pos = ax3.get_position()
    ax3.set_position([ax3_original_pos.x0 - 0.02, ax3_original_pos.y0 + 0.01,
                      ax3_original_pos.width * 0.9, ax3_original_pos.height * 0.9])

    ax2_original_pos = ax2.get_position()
    legend2_original_pos = legend2.get_position()
    ax2.set_position([ax2_original_pos.x0 - 0.03, ax2_original_pos.y0, ax2_original_pos.width, ax2_original_pos.height])
    legend2.set_position([legend2_original_pos.x0 - 0.02, legend2_original_pos.y0,
                          legend2_original_pos.width, legend2_original_pos.height])

    # Panel D - Theil index components
    df_sort = df.sort_values('popul')
    Nk = df_sort['popul'].values
    case_real = df_sort['real_cases'].values
    case_sim_main = df_sort['simulated_cases'].values
    case_sim_nomob = df_sort['simulated_cases_no_mob'].values

    n_groups = len(case_real)
    positions = np.arange(n_groups)
    contribution_real_data = calculate_contribution_theil_index(case_real, Nk)

    ax4.scatter(positions, contribution_real_data[0], color='darkgrey', marker='o', s=90, alpha=0.8, label='Real data',
                zorder=3)

    model_names = ['SLIR with mobility', 'SLIR without mobility']
    colors_theil = ['#F88455', '#76CBB4']
    simulation_deaths_list = [case_sim_main, case_sim_nomob]

    for idx, simulation_deaths in enumerate(simulation_deaths_list):
        simulation_deaths = np.array(simulation_deaths)
        simulation_contribution = calculate_contribution_theil_index(simulation_deaths, Nk)
        F4_2 = np.exp(-sum(np.absolute(simulation_contribution[0] - contribution_real_data[0])))
        print(f'F score (based on Theil index) {model_names[idx]}:', round(F4_2, 3))
        ax4.scatter(positions, simulation_contribution[0], color=colors_theil[idx], marker='o', s=90, alpha=0.8,
                    label=f'{model_names[idx]}', zorder=2)

    ax4.set_xticks(positions)
    ax4.set_xticklabels(df_sort['comuna'].values, rotation=70, fontsize=14)
    ax4.set_xlabel('Comunas', fontsize=20)
    ax4.set_ylabel('Components of Theil index', fontsize=20)
    ax4.grid(True, linestyle='-', alpha=0.3)
    ax4.tick_params(axis='y', labelsize=18)
    ax4.set_ylim((-0.05, 0.08))
    ax4.spines[['right', 'top']].set_visible(False)
    ax4.legend(loc='upper left', frameon=False, fontsize=18)

    ax4_pos = ax4.get_position()
    fig.text(ax4_pos.x0, ax4_pos.y0 + ax4_pos.height + 0.04, 'd', fontsize=26, fontweight='bold', ha='left', va='top')

    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print('Figure 3 finished!')
    return output_path


if __name__ == "__main__":
    plot_figure3()