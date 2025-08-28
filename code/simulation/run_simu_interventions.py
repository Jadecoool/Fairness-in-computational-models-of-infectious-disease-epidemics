import pandas as pd
import numpy as np
from SEIR_model_interventions import SEIR
import yaml
import os
from load_config import load_config
import matplotlib.pyplot as plt


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
        #removal_fractions = np.array([0.22675884, 0.42130029, 0.23249578, 0.11944509, 0]) # gamma =1
        removal_fractions = np.array([0.27545968, 0.36098201, 0.22974655, 0.13381176, 0]) # gamma =0.5

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



run_cf(strategy='combine', iterations=1000, remove_day=1, total_remove_frac = 0.2)



# plt.plot(range(len(results_org[0])), results_org[0]/np.sum(Nk)*100000, label='without interventions')
# plt.plot(range(len(results_intervention[0])), results_intervention[0]/np.sum(Nk)*100000, label='with interventions')
# plt.ylabel('Total death rate per 100,000')
# plt.legend(frameon=False)
# plt.show()
#
#
# for ig in range(0,4):
#     print((sum(results_org[2][ig])-sum(results_intervention[2][ig]))/Nk[ig])
#     plt.plot(range(len(results_org[2][ig])), results_org[2][ig]/np.sum(Nk)*100000, label='without interventions')
#     plt.plot(range(len(results_intervention[2][ig])), results_intervention[2][ig]/np.sum(Nk)*100000)
#     plt.show()


'''
results_org_races= np.array(results_org_races)
results_intervention_races= np.array(results_intervention_races)

fig, axes = plt.subplots(2, 3, figsize=(18, 12))
# fig.suptitle('Death Rate Comparison: With vs Without Interventions', fontsize=16)

# 第一个子图：总体死亡率比较
ax = axes[0, 0]
ax.fill_between(range(len(results_org_overall[0])), np.quantile(results_org_overall / np.sum(Nk) * 100000, axis=0, q=0.025),
                                                np.quantile(results_org_overall / np.sum(Nk) * 100000, axis=0, q=0.975),
                                                linewidth=2, alpha=0.1)
ax.fill_between(range(len(results_intervention_overall[0])), np.quantile(results_intervention_overall / np.sum(Nk) * 100000, axis=0, q=0.025),
                                                np.quantile(results_intervention_overall / np.sum(Nk) * 100000, axis=0, q=0.975),
                                                linewidth=2, alpha=0.1)

ax.plot(range(len(results_org_overall[0])), np.quantile(results_org_overall / np.sum(Nk) * 100000, axis=0, q=0.5),
                                                label='without interventions', linewidth=2)
ax.plot(range(len(results_org_overall[0])), np.quantile(results_intervention_overall / np.sum(Nk) * 100000, axis=0, q=0.5),
                                                label='with interventions', linewidth=2)



difference = ((sum(np.quantile(results_org_overall, axis=0, q=0.5)) - sum(np.quantile(results_intervention_overall, axis=0, q=0.5)))
              / sum(np.quantile(results_org_overall, axis=0, q=0.5)))
print(f"Overal difference: {difference}")

ax.set_ylabel('Total death rate per 100,000')
ax.set_xlabel('t')
ax.set_title('Overall')
max_overall = np.max(results_org_overall / np.sum(Nk) * 100000)
ax.set_ylim(0,max_overall+5)
ax.legend(frameon=False)
ax.grid(True, alpha=0.3)

# 接下来的4个子图：各年龄组比较
group_names = ['White', 'Hispanic', 'Black', 'Asian']
plot_positions = [(0, 1), (0, 2), (1, 0), (1, 1)]


for ig in range(0, 4):
    # 计算差异并打印
    difference = ( sum( np.quantile(results_org_races[:, ig, :], q=0.5, axis=0) )
                   - sum( np.quantile(results_intervention_races[:, ig, :], q=0.5, axis=0) ) ) / sum( np.quantile(results_org_races[:, ig, :], axis=0, q=0.5) )

    print(f"{group_names[ig]} difference: {difference}")

    # 获取子图位置
    row, col = plot_positions[ig]
    ax = axes[row, col]

    # 绘制对比图
    # ax.plot(range(len(results_org[2][ig])), results_org[2][ig] / np.sum(Nk) * 100000,
    #         label='without interventions', linewidth=2)
    # ax.plot(range(len(results_intervention[2][ig])), results_intervention[2][ig] / np.sum(Nk) * 100000,
    #         label='with interventions', linewidth=2)

    ax.fill_between(range(len(results_org_races[0][ig])),
                    np.quantile(results_org_races[:, ig, :] / np.sum(Nk) * 100000, axis=0, q=0.025),
                    np.quantile(results_org_races[:, ig, :] / np.sum(Nk) * 100000, axis=0, q=0.975),
                    linewidth=2, alpha=0.1)
    ax.fill_between(range(len(results_intervention_races[0][ig])),
                    np.quantile(results_intervention_races[:, ig, :] / np.sum(Nk) * 100000, axis=0, q=0.025),
                    np.quantile(results_intervention_races[:, ig, :] / np.sum(Nk) * 100000, axis=0, q=0.975),
                    linewidth=2, alpha=0.1)

    ax.plot(range(len(results_org_races[0][ig])), np.quantile(results_org_races[:, ig, :] / np.sum(Nk) * 100000, axis=0, q=0.5),
            label='without interventions', linewidth=2)
    ax.plot(range(len(results_org_races[0][ig])),
            np.quantile(results_intervention_races[:, ig, :] / np.sum(Nk) * 100000, axis=0, q=0.5),
            label='with interventions', linewidth=2)

    ax.set_ylabel('Death rate per 100,000')
    ax.set_xlabel('t')
    ax.set_title(group_names[ig])
    ax.legend(frameon=False)
    ax.set_ylim(0,max_overall+5)
    ax.grid(True, alpha=0.3)

# 移除最后一个空的子图
fig.delaxes(axes[1, 2])

# 调整子图间距
plt.tight_layout()
plt.subplots_adjust(
    top=0.93,       # 顶部边距
    bottom=0.08,    # 底部边距
    hspace=0.3,     # 子图之间的垂直间距（上下间距）
    wspace=0.2      # 子图之间的水平间距（左右间距）
)
# 显示图表
plt.show()
'''