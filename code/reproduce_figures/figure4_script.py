"""
Combined figure:
  Left  – RCI concentration curves (from Script 1)
  Right – 2×2 Theil index box plots (from Script 2)
  Both panels have equal size.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.gridspec import GridSpec
from itertools import combinations
from scipy import stats
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


AGE_LABELS = ['0-17', '18-24', '25-34', '35-44',
              '45-54', '55-64', '65-74', '75+']

AGE_COLS_REAL = [
    'age_0_17', 'age_18_24', 'age_25_34', 'age_35_44',
    'age_45_54', 'age_55_64', 'age_65_74', 'age_75up'
]

AGE_KEYS_SIM = [
    'deaths_age_0_17', 'deaths_age_18_24', 'deaths_age_25_34', 'deaths_age_35_44',
    'deaths_age_45_54', 'deaths_age_55_64', 'deaths_age_65_74', 'deaths_age_75up'
]

SIM_START = '2020-03-15'
SIM_END   = '2020-07-05'
FIT_START = '2020-03-15'
N_DRAWS   = 100

BASE = ROOT / 'code' / 'calibration' / 'covasim_posteriors_borough'
DATA = ROOT / 'data' / 'regions'

POP_FILE       = DATA / 'NYC' / 'demographic' / 'pop_age.csv'
REAL_DATA_PATH = DATA / 'NYC' / 'epidemic' / 'weekly_death_byage.csv'

MODELS_RCI = [
    (BASE / 'BORO_AGG_prog0_sus0_random_trajectories_by_age_weekly.npz',
     'HomProbs, RN', True),
    (BASE / 'BORO_AGG_prog1_sus1_random_trajectories_by_age_weekly.npz',
     'AgeProbs, RN', True),
    (BASE / 'BORO_AGG_prog0_sus0_hybrid_trajectories_by_age_weekly.npz',
     'HomProbs, HN', True),
    (BASE / 'BORO_AGG_prog1_sus1_hybrid_trajectories_by_age_weekly.npz',
     'AgeProbs, HN', True),
]

BOROUGHS      = ['BX', 'BK', 'MN', 'QN', 'SI']
BOROUGH_NAMES = ['Bronx', 'Brooklyn', 'Manhattan', 'Queens', 'Staten Island']
CONFIGS_THEIL = [
    ('prog0_sus0_random', 'HomProbs, RN'),
    ('prog1_sus1_random', 'AgeProbs, RN'),
    ('prog0_sus0_hybrid', 'HomProbs, HN'),
    ('prog1_sus1_hybrid', 'AgeProbs, HN'),
]

def load_population_8groups(pop_file):
    pop = pd.read_csv(pop_file, encoding='utf-8-sig')
    p = pop['population'].values
    population = np.array([
        int(p[0] + p[1] + p[2] + int(p[3] * 8 / 10)),
        int(int(p[3] * 2 / 10) + int(p[4] * 5 / 10)),
        int(int(p[4] * 5 / 10) + p[5]),
        int(p[6] + p[7]),
        int(p[8] + p[9]),
        int(p[10] + p[11]),
        int(p[12] + p[13]),
        int(p[14]) + int(p[15]),
    ], dtype=float)
    return population


def compute_rci(population, death_counts):
    cum_pop_pct   = np.cumsum(population) / population.sum()
    cum_death_pct = np.cumsum(death_counts) / death_counts.sum()
    x = np.insert(cum_pop_pct, 0, 0.0)
    y = np.insert(cum_death_pct, 0, 0.0)
    area = np.trapz(y, x)
    return 1.0 - 2.0 * area


def get_real_deaths(csv_path):
    df = pd.read_csv(csv_path)
    df['date'] = pd.to_datetime(df['date'])
    mask = (df['date'] >= '2020-03-14') & (df['date'] <= SIM_END)
    return df.loc[mask, AGE_COLS_REAL].sum().values


def get_model_median_deaths(npz_path, is_weekly=False):
    data = np.load(npz_path, allow_pickle=True)
    n_ages = len(AGE_KEYS_SIM)
    median_deaths = np.zeros(n_ages, dtype=float)
    if is_weekly:
        for j, key in enumerate(AGE_KEYS_SIM):
            arr = data[key]
            median_weekly = np.median(arr, axis=0)
            median_deaths[j] = median_weekly.sum()
    else:
        dates = pd.to_datetime(data['dates'])
        for j, key in enumerate(AGE_KEYS_SIM):
            arr = data[key]
            median_daily = np.median(arr, axis=0)
            ts = pd.Series(median_daily, index=dates)
            weekly = ts.resample('W-SAT').sum()
            weekly_cut = weekly[(weekly.index >= SIM_START) & (weekly.index <= SIM_END)]
            median_deaths[j] = weekly_cut.sum()
    return median_deaths


def load_model_totals(npz_path, is_weekly=False):
    data = np.load(npz_path, allow_pickle=True)
    n_ages = len(AGE_KEYS_SIM)
    if is_weekly:
        n_samples = data[AGE_KEYS_SIM[0]].shape[0]
        totals = np.zeros((n_samples, n_ages), dtype=float)
        for j, key in enumerate(AGE_KEYS_SIM):
            totals[:, j] = data[key].sum(axis=1)
    else:
        dates = pd.to_datetime(data['dates'])
        n_samples = data[AGE_KEYS_SIM[0]].shape[0]
        totals = np.zeros((n_samples, n_ages), dtype=float)
        for j, key in enumerate(AGE_KEYS_SIM):
            arr = data[key]
            for i in range(n_samples):
                ts = pd.Series(arr[i], index=dates)
                weekly = ts.resample('W-SAT').sum()
                weekly_cut = weekly[(weekly.index >= SIM_START) & (weekly.index <= SIM_END)]
                totals[i, j] = weekly_cut.sum()
    return totals


def compute_model_scores(population, totals, rci_real):
    n_samples = totals.shape[0]
    rci_samples = np.array([compute_rci(population, totals[i]) for i in range(n_samples)])
    fscore_samples = 1.0 - np.abs(rci_real - rci_samples) / 2.0
    return rci_samples, fscore_samples


def concentration_curve(population, deaths):
    cum_pop_pct   = np.cumsum(population) / population.sum()
    cum_death_pct = np.cumsum(deaths) / deaths.sum()
    x = np.insert(cum_pop_pct, 0, 0.0)
    y = np.insert(cum_death_pct, 0, 0.0)
    return x, y


def summarise(values):
    return np.median(values), np.percentile(values, 2.5), np.percentile(values, 97.5)


def calculate_contribution_theil_index(deaths, populations):
    deaths = np.array(deaths, dtype=float)
    if deaths.ndim == 1:
        deaths = deaths.reshape(1, -1)
    population_shares = populations / populations.sum()
    death_shares = deaths / deaths.sum(axis=1)[:, np.newaxis]
    ratios = death_shares / population_shares
    return death_shares * np.log(ratios)


def daily_to_weekly_total(daily_traj, dates):
    n_samples = daily_traj.shape[0]
    totals = np.zeros(n_samples)
    df_dates = pd.to_datetime(dates)
    for i in range(n_samples):
        s = pd.Series(daily_traj[i], index=df_dates)
        w = s.resample('W').sum()
        w = w[FIT_START:SIM_END]
        totals[i] = w.sum()
    return totals


def load_borough_deaths_sampled(config):
    borough_total = {}
    for b in BOROUGHS:
        fpath = f'{BASE}/{b}_{config}_trajectories.npz'
        data = np.load(fpath, allow_pickle=True)
        traj = data['trajectories']
        dates = data['dates']
        borough_total[b] = daily_to_weekly_total(traj, dates)

    n_samples = len(borough_total[BOROUGHS[0]])
    rng = np.random.default_rng(seed=42)
    idx = {b: rng.integers(0, n_samples, size=N_DRAWS) for b in BOROUGHS}

    result = np.zeros((N_DRAWS, len(BOROUGHS)))
    for j, b in enumerate(BOROUGHS):
        result[:, j] = borough_total[b][idx[b]]
    return result

if __name__ == '__main__':

    # ── Load data for RCI ──
    population = load_population_8groups(POP_FILE)
    real_deaths = get_real_deaths(REAL_DATA_PATH)
    rci_real = compute_rci(population, real_deaths)
    print(f"Real data RCI: {rci_real:.4f}")

    # ── Load data for Theil ──
    borough_pop = np.array([
        pd.read_csv(f'{DATA}/NYC_{b}/demographic/pop_age.csv')['population'].sum()
        for b in BOROUGHS
    ], dtype=float)

    real_deaths_by_borough = np.zeros(len(BOROUGHS))
    for i, b in enumerate(BOROUGHS):
        df = pd.read_csv(f'{DATA}/NYC_{b}/epidemic/weekly_deaths.csv')
        mask = (df['date'] >= '2020-03-14') & (df['date'] <= SIM_END)
        real_deaths_by_borough[i] = df.loc[mask, 'total'].sum()

    contribution_real = calculate_contribution_theil_index(real_deaths_by_borough, borough_pop)
    theil_real = contribution_real.sum()
    print(f"Real data Theil Index: {theil_real:.4f}")


    fig = plt.figure(figsize=(20, 9))
    gs = GridSpec(2, 4, figure=fig, wspace=0.45, hspace=0.45)

    # Left panel spans rows 0-1, cols 0-1 (i.e. half the width)
    ax_left = fig.add_subplot(gs[:, 0:2])

    # Right 2×2 panels: each occupies one cell in rows 0-1, cols 2-3
    ax_r = [
        fig.add_subplot(gs[0, 2]),  # top-left
        fig.add_subplot(gs[0, 3]),  # top-right
        fig.add_subplot(gs[1, 2]),  # bottom-left
        fig.add_subplot(gs[1, 3]),  # bottom-right
    ]

    tab20c = plt.cm.tab20c
    colors_rci = [tab20c(1), '#4ecdc4', '#ff6b6b', tab20c(13)]
    linestyles_rci = ['-', '-', '-', '-']
    markers_rci = ['o', 'o', 'o', 'o']

    x_real, y_real = concentration_curve(population, real_deaths)
    ax_left.scatter(x_real[1:], y_real[1:], color='k', s=100, facecolors='w',
                    zorder=5, label='Real data')

    all_results_rci = {}
    for idx_m, (npz_path, label, is_weekly) in enumerate(MODELS_RCI):
        try:
            median_deaths = get_model_median_deaths(npz_path, is_weekly=is_weekly)
            totals = load_model_totals(npz_path, is_weekly=is_weekly)
        except FileNotFoundError:
            print(f'Skipping (not found): {label}')
            continue

        rci_samples, fscore_samples = compute_model_scores(population, totals, rci_real)
        all_results_rci[label] = {'rci': rci_samples, 'fscore': fscore_samples}

        rci_med, rci_lo, rci_hi = summarise(rci_samples)
        fs_med, fs_lo, fs_hi = summarise(fscore_samples)
        print(f"\n{label}")
        print(f"  RCI:     {rci_med:.4f}  [95% CI: {rci_lo:.4f}, {rci_hi:.4f}]")
        print(f"  F-score: {fs_med:.4f}  [95% CI: {fs_lo:.4f}, {fs_hi:.4f}]")

        x_m, y_m = concentration_curve(population, median_deaths)
        ax_left.plot(x_m, y_m, marker=markers_rci[idx_m], markersize=11, linewidth=2,
                     color=colors_rci[idx_m], alpha=1, label=label,
                     linestyle=linestyles_rci[idx_m])

    ax_left.plot([0, 1], [0, 1], '--', color='dimgrey', linewidth=1.5,
                 label='Line of equality')

    ax_left.set_xlabel('Cumulative population proportion', fontsize=18)
    ax_left.set_ylabel('Cumulative death proportion', fontsize=18)
    ax_left.legend(frameon=False, fontsize=18, loc='upper left')
    ax_left.grid(color='grey', linestyle='--', linewidth=0.35, alpha=0.6)
    ax_left.tick_params(labelsize=18)
    #ax_left.tick_params(axis='y', labelsize=18)
    ax_left.spines[['right', 'top']].set_visible(False)
    #ax_left.set_title('RCI Concentration Curves', fontsize=16, fontweight='bold', pad=12)
    ax_left.text(-0.08, 1.06, 'a', transform=ax_left.transAxes,
             fontsize=24, fontweight='bold', va='top', ha='right')

    n_boroughs = len(BOROUGHS)
    positions = np.arange(n_boroughs)
    all_fscores_theil = {}
    panel_labels = ['', '', '', '']

    for idx_c, (config, config_name) in enumerate(CONFIGS_THEIL):
        ax = ax_r[idx_c]

        try:
            sampled_deaths = load_borough_deaths_sampled(config)
        except FileNotFoundError as e:
            print(f'Skipping {config_name}: {e}')
            continue

        model_contrib = calculate_contribution_theil_index(sampled_deaths, borough_pop)
        theil_samples = model_contrib.sum(axis=1)
        theil_med = np.median(theil_samples)
        theil_lo, theil_hi = np.percentile(theil_samples, [2.5, 97.5])
        print(f'\n{config_name}')
        print(f'  Theil Index: {theil_med:.4f} [{theil_lo:.4f}, {theil_hi:.4f}]')

        F = np.exp(-np.abs(model_contrib - contribution_real).sum(axis=1))
        all_fscores_theil[config_name] = F
        fs_med = np.median(F)
        fs_lo, fs_hi = np.percentile(F, [2.5, 97.5])
        print(f'  F-score: {fs_med:.3f} [{fs_lo:.3f}, {fs_hi:.3f}]')

        # Box plot
        box_data = []
        for j in range(n_boroughs):
            q1, median, q2 = np.percentile(model_contrib[:, j], [5, 50, 95])
            stats_dict = {
                'med': median,
                'q1': q1,
                'q3': q2,
                'whislo': np.min(model_contrib[:, j]),
                'whishi': np.max(model_contrib[:, j]),
                'fliers': []
            }
            box_data.append(stats_dict)

        bplot = ax.bxp(box_data, positions=positions, widths=0.6,
                       patch_artist=True,
                       medianprops=dict(color='black'),
                       boxprops=dict(facecolor='#58b69e', alpha=0.6),
                       showfliers=False, zorder=1)

        p1 = ax.scatter(positions, contribution_real[0], color='darkgrey',
                        marker='o', s=70, zorder=2)

        ax.set_xticks(positions)
        if idx_c < 2:
            ax.set_xticklabels([])
        else:
            ax.set_xticklabels(BOROUGH_NAMES, rotation=35, fontsize=14, ha='center')
        ax.grid(True, linestyle='-', alpha=0.3)
        ax.tick_params(axis='y', labelsize=14)
        ax.spines[['right', 'top']].set_visible(False)
        ax.text(0.91, 0.96, f"F={fs_med:.3f}",
                ha='center', va='center', fontsize=16,
                bbox=dict(boxstyle="square", facecolor='lightgrey', alpha=0.2, edgecolor='grey'),
                transform=ax.transAxes)
        ax.set_title(config_name, fontsize=16, pad=8)
        ax_r[0].text(-0.35, 1.15, 'b', transform=ax_r[0].transAxes, fontsize=24, fontweight='bold', va='top', ha='right')

    # Y-axis labels only on the left column of the 2×2 grid
    ax_r[0].set_ylabel('Components of Theil index', fontsize=16)
    ax_r[2].set_ylabel('Components of Theil index', fontsize=16)

    # Shared legend for right panels
    box_patch = Patch(facecolor='#58b69e', alpha=0.6)
    fig.legend([box_patch, p1], ["Model outcome", "Real data"],
               loc='lower center', bbox_to_anchor=(0.72, 0.47), #bbox_to_anchor=(0.72, -0.02),
               ncol=2, frameon=False, fontsize=16)

    plt.savefig('figure4.png', dpi=300, bbox_inches='tight')
    plt.show()

    print(f"\n{'='*80}")
    print("RCI Summary (median [2.5%, 97.5%])")
    print(f"{'='*80}")
    print(f"{'Model':<40} {'RCI':>25} {'F-score':>25}")
    print("-" * 90)
    for label, res in all_results_rci.items():
        r = res['rci']
        f = res['fscore']
        rci_str = f"{np.median(r):.4f} [{np.percentile(r,2.5):.4f}, {np.percentile(r,97.5):.4f}]"
        f_str   = f"{np.median(f):.4f} [{np.percentile(f,2.5):.4f}, {np.percentile(f,97.5):.4f}]"
        print(f"{label:<40} {rci_str:>25} {f_str:>25}")
    print(f"\nReal data RCI: {rci_real:.4f}")

    print(f"\n{'='*80}")
    print("Theil Summary (median [2.5%, 97.5%])")
    print(f"{'='*80}")
    print(f"{'Model':<45} {'Theil Index':>25} {'F-score':>25}")
    print("-" * 95)
    for config, config_name in CONFIGS_THEIL:
        if config_name in all_fscores_theil:
            F = all_fscores_theil[config_name]
            sampled = load_borough_deaths_sampled(config)
            mc = calculate_contribution_theil_index(sampled, borough_pop)
            ts = mc.sum(axis=1)
            t_str = f"{np.median(ts):.4f} [{np.percentile(ts,2.5):.4f}, {np.percentile(ts,97.5):.4f}]"
            f_str = f"{np.median(F):.3f} [{np.percentile(F,2.5):.3f}, {np.percentile(F,97.5):.3f}]"
            print(f"{config_name:<45} {t_str:>25} {f_str:>25}")
    print(f"\nReal data Theil Index: {theil_real:.4f}")

    # Pairwise t-tests
    print(f"\n{'='*80}")
    print("Pairwise t-tests for Theil F-scores")
    print(f"{'='*80}")
    model_names_list = [name for _, name in CONFIGS_THEIL if name in all_fscores_theil]
    for m1, m2 in combinations(model_names_list, 2):
        t_stat, p_val = stats.ttest_ind(all_fscores_theil[m1], all_fscores_theil[m2], equal_var=False)
        sig = '***' if p_val < 0.001 else ('**' if p_val < 0.01 else ('*' if p_val < 0.05 else 'ns'))
        print(f'{m1} vs {m2}: t={t_stat:.4f}, p={p_val:.4e} {sig}')

    print("\nSaved: figure4.png")