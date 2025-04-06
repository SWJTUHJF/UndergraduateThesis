import pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.core.pylabtools import figsize

from NPUE import Network, lower_problem
from UE import FW
from matplotlib.patches import FancyArrowPatch
import seaborn as sns


def first_insight():
    # Basic data
    ue_tstt, so_tstt = 93525, 88866
    five_PUE, five_s = 93556, [5, 5, 5, 5]
    three_PUE, three_s = 93556, [3, 3, 3, 3]
    two_PUE, two_s = 100071, [2, 2, 2, 2]
    optimal_PUE, optimal_s = 90219, [3, 1, 2, 2]
    fig, ax = plt.subplots(figsize=(10, 6))
    x = [1, 2, 3, 4]
    height = [two_PUE, three_PUE, five_PUE, optimal_PUE]
    # plot the bar
    ax.bar(x=x, height=height, width=0.6, alpha=0.8, color="#87CEEB")
    # set x and y ticks and labels
    ax.set_xlim(0, 5)
    ax.set_xticks(range(1, 5))
    ax.set_xticklabels([f"{two_s}", f"{three_s}", f"{five_s}", f"{optimal_s}(Optimal)"], fontsize=12)
    ax.set_ylim(85000, 102000)
    ax.set_yticks([85000, 88866, 90000, 93525, 95000, 100000])
    ax.set_xlabel("Information provision strategy", fontsize=15)
    ax.set_ylabel("Total system travel time", fontsize=15)
    # remove the axis line
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # add the horizontal gird
    ax.yaxis.grid(True, which='major', color='gray', linestyle='-', linewidth=1.2, alpha=0.5)
    ax.set_axisbelow(True)  # 确保网格线在条形图下方
    # remove the ticks
    ax.yaxis.set_tick_params(length=0)  # 设置 y 轴刻度长度为 0
    # plot the UE and SO
    ax.hlines(ue_tstt, xmax=5, xmin=0, color="#FFA500")
    ax.text(4.35, ue_tstt+200, "TSTT under UE", color='orange', fontsize=12)
    ax.hlines(so_tstt, xmax=5, xmin=0, color="#FFA500")
    ax.text(4.35, so_tstt-600, "TSTT under SO", color='orange', fontsize=12)
    # show the improvement
    arrow = FancyArrowPatch((4, ue_tstt - 1800), (4, optimal_PUE),
                            arrowstyle='-|>', color='g', mutation_scale=10,
                            linestyle="--")
    ax.add_patch(arrow)
    arrow = FancyArrowPatch((4, optimal_PUE + 1800), (4, ue_tstt),
                            arrowstyle='-|>', color='g', mutation_scale=10,
                            linestyle="--")
    ax.add_patch(arrow)
    ax.text(3.48, 91750, "A 70% improvement towards SO", color="g", fontsize=12)
    # add annotations
    ax.text(2.25, 98500, "Users from OD pair 1, 2, 3, 4 are provided with a, b, c, d"
                         "\npaths when information provision strategy is [a, b, c, d].", fontsize=12,
            bbox=dict(boxstyle='round,pad=0.3', edgecolor='black', facecolor='gray', alpha=0.3))
    # adjust the margin space
    plt.subplots_adjust(left=0.1, right=0.95, top=0.95, bottom=0.1)
    ax.set_title("TSTT of several information provision strategies", fontsize=20, pad=0)
    plt.show()


def generate_data():
    net_ue = FW(name='Nguyen-Dupuis', algorithm='LS', BC=0.0001, FWC=0.0001, model="UE", sst=1)
    net_ue.conduct_FW()
    link_flow_ue = [link.flow for link in net_ue.LINKS[1:]]
    link_cost_ue = [link.cost for link in net_ue.LINKS[1:]]
    net_so = FW(name='Nguyen-Dupuis', algorithm='LS', BC=0.0001, FWC=0.0001, model="SO", sst=1)
    net_so.conduct_FW()
    link_flow_so = [link.flow for link in net_so.LINKS[1:]]
    link_cost_so = [link.get_specific_cost(link.flow) for link in net_so.LINKS[1:]]
    nd = Network("Nguyen-Dupuis", sst=1)
    lower_problem(nd, ppn=[3, 1, 2, 2])
    link_flow_npue =[link.flow for link in nd.LINKS[1:]]
    link_cost_npue = [link.cost for link in nd.LINKS[1:]]
    with open("nd_optimal\\flow_ue.pkl", "wb") as file:
        pickle.dump(link_flow_ue, file)
    with open("nd_optimal\\cost_ue.pkl", "wb") as file:
        pickle.dump(link_cost_ue, file)
    with open("nd_optimal\\flow_so.pkl", "wb") as file:
        pickle.dump(link_flow_so, file)
    with open("nd_optimal\\cost_so.pkl", "wb") as file:
        pickle.dump(link_cost_so, file)
    with open("nd_optimal\\flow_npue.pkl", "wb") as file:
        pickle.dump(link_flow_npue, file)
    with open("nd_optimal\\cost_npue.pkl", "wb") as file:
        pickle.dump(link_cost_npue, file)


def cost_heatmap():
    net_ue = FW(name='Nguyen-Dupuis', algorithm='LS', BC=0.0001, FWC=0.0001, model="UE", sst=1)
    net_ue.conduct_FW()
    net_so = FW(name='Nguyen-Dupuis', algorithm='LS', BC=0.0001, FWC=0.0001, model="SO", sst=1)
    net_so.conduct_FW()
    nd = Network("Nguyen-Dupuis", sst=1)
    lower_problem(nd, ppn=[3, 1, 2, 2])
    npue_ue = np.zeros(shape=(len(nd.NODES[1:]), len(nd.NODES[1:])))
    npue_so = np.zeros(shape=(len(nd.NODES[1:]), len(nd.NODES[1:])))
    for i in range(1, len(nd.LINKS)):
        cost_ue = net_ue.LINKS[i].cost
        cost_so = net_so.LINKS[i].get_specific_cost(net_so.LINKS[i].flow)
        cost_npue = nd.LINKS[i].cost
        tail, head = nd.LINKS[i].tail.node_id, nd.LINKS[i].head.node_id
        npue_ue[tail-1][head-1] = (cost_npue-cost_ue) / cost_ue
        npue_so[tail-1][head-1] = (cost_npue-cost_so) / cost_so
    # 设置图形
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6), gridspec_kw={'width_ratios': [1, 1]})
    # 绘制热图
    heatmap1 = sns.heatmap(npue_ue, ax=ax1, cmap='RdBu', annot=False, cbar=True,
                cbar_kws={'label': 'Relative changes in link cost comparing DIPS and UE'},
                vmin=-0.3, vmax=0.3, linewidths=0.8, linecolor='white', alpha=0.9)
    ax1.set_title('Cost changes compared to UE')
    ax1.set_xlabel('Node (From)')
    ax1.set_xticklabels(np.arange(1, 14))
    ax1.set_ylabel('Node (To)')
    ax1.set_yticklabels(np.arange(1, 14))
    ax1.set_aspect('equal')

    sns.heatmap(npue_so, ax=ax2, cmap='RdBu', annot=False, cbar=True,
                cbar_kws={'label': 'Relative changes in link cost comparing DIPS and SO'},
                vmin=-0.3, vmax=0.3, linewidths=0.8, linecolor='white', alpha=0.9)
    ax2.set_title('Cost changes compared to SO')
    ax2.set_xlabel('Node (From)')
    ax2.set_xticklabels(np.arange(1, 14))
    ax2.set_ylabel('Node (to)')
    ax2.set_yticklabels(np.arange(1, 14))
    ax2.set_aspect('equal')


    # 显示图形
    plt.tight_layout()
    plt.show()


def flow_heatmap():
    net_ue = FW(name='Nguyen-Dupuis', algorithm='LS', BC=0.0001, FWC=0.0001, model="UE", sst=1)
    net_ue.conduct_FW()
    net_so = FW(name='Nguyen-Dupuis', algorithm='LS', BC=0.0001, FWC=0.0001, model="SO", sst=1)
    net_so.conduct_FW()
    nd = Network("Nguyen-Dupuis", sst=1)
    lower_problem(nd, ppn=[3, 1, 2, 2])
    npue_ue = np.zeros(shape=(len(nd.NODES[1:]), len(nd.NODES[1:])))
    npue_so = np.zeros(shape=(len(nd.NODES[1:]), len(nd.NODES[1:])))
    for i in range(1, len(nd.LINKS)):
        flow_ue = net_ue.LINKS[i].flow
        flow_so = net_so.LINKS[i].flow
        flow_npue = nd.LINKS[i].flow
        tail, head = nd.LINKS[i].tail.node_id, nd.LINKS[i].head.node_id
        npue_ue[tail-1][head-1] = (flow_npue-flow_ue) / flow_ue
        npue_so[tail-1][head-1] = (flow_npue-flow_so) / flow_so

    # 设置图形
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6), gridspec_kw={'width_ratios': [1, 1]})
    # 绘制热图
    sns.heatmap(npue_ue, ax=ax1, cmap='RdBu', annot=False, cbar=True,
                cbar_kws={'label': 'Relative changes in link flow comparing DIPS and UE'},
                vmin=-0.3, vmax=0.3, linewidths=0.8, linecolor='white', alpha=0.9)
    ax1.set_title('Flow changes compared to UE')
    ax1.set_xlabel('Node (From)')
    ax1.set_xticklabels(np.arange(1, 14))
    ax1.set_ylabel('Node (To)')
    ax1.set_yticklabels(np.arange(1, 14))
    ax1.set_aspect('equal')

    sns.heatmap(npue_so, ax=ax2, cmap='RdBu', annot=False, cbar=True,
                cbar_kws={'label': 'Relative changes in link flow comparing DIPS and SO'},
                vmin=-0.3, vmax=0.3, linewidths=0.8, linecolor='white', alpha=0.9)
    ax2.set_title('Flow changes compared to SO')
    ax2.set_xlabel('Node (From)')
    ax2.set_xticklabels(np.arange(1, 14))
    ax2.set_ylabel('Node (to)')
    ax2.set_yticklabels(np.arange(1, 14))
    ax2.set_aspect('equal')

    # 显示图形
    plt.tight_layout()
    plt.show()


def final():
    time_list_ue = list()
    time_list_npue = list()
    net_ue = FW(name='Nguyen-Dupuis', algorithm='LS', BC=0.0001, FWC=0.0001, model="UE", sst=1)
    net_ue.conduct_FW()
    nd = Network("Nguyen-Dupuis", sst=1)
    lower_problem(nd, ppn=[3, 1, 2, 2])
    for od in net_ue.ODPAIRS:
        o_id = od.origin.node_id
        d_id = od.destination.node_id
        net_ue.LS(o_id, d_id)
        min_path = net_ue.obtain_shortest_path(d_id)
        time_list_ue.append(sum([link.cost for link in min_path]))
    for od in nd.OD:
        min_dist, min_path = od.find_shortest_path_in_PPS()
        time_list_npue.append(min_dist)
    reduction = [(time_list_ue[i]-time_list_npue[i])/time_list_ue[i] for i in range(4)]
    reduction_new = [ele+0.04 for ele in reduction]
    fig, ax = plt.subplots()
    # Example data
    od = ['OD1(1, 2)', 'OD2(1, 3)', 'OD3(4, 2)', 'OD4(4, 3)']
    ax.bar(x=[1, 2, 3, 4], height=reduction_new, width=0.6, alpha=0.8,
           color=['skyblue', 'green', 'skyblue', 'skyblue'])
    ax.set_yticks([0.00, 0.02, 0.04, 0.06, 0.08, 0.1])
    ax.set_yticklabels(["-4%", "-2%", "0", "2%", "4%", "6%"])
    ax.set_xticks([1, 2, 3, 4])
    ax.set_xticklabels(od)
    ax.set_xlabel("Users from different OD pairs", fontsize=12)
    ax.set_ylabel("Relative improvement of travel time", fontsize=12)
    ax.set_title("How many drivers are benefited?", fontsize=15)
    ax.hlines(0.04, xmax=4.5, xmin=0.5, color="gray", linestyles="--", alpha=0.5)
    plt.show()


def ch2_convergence():
    nd = Network(name="Nguyen-Dupuis", sst=1, model="UE", from_file=False)
    nd.generate_total_path_set()
    total_gap_list = list()
    for num in range(2, 6):
        ppn = [num] * len(nd.OD)
        _, gap_list = lower_problem(nd, ppn)
        total_gap_list.append(gap_list)
    fig, axes = plt.subplots(2, 2, figsize=(15, 9), sharey=True)
    for i in range(2):
        for j in range(2):
            axes[i][j].set_xlabel("Number of iteration", fontsize=12)
            axes[i][j].set_ylabel("Relative gap", fontsize=12)
            num = 2 * i + j
            axes[i][j].set_title(f"Provide {num+2} paths to all users", fontsize=15)
            axes[i][j].plot(range(len(total_gap_list[num])), total_gap_list[num])
            axes[i][j].set_yscale("log")
            axes[i][j].hlines(1e-7, xmax=len(total_gap_list[num])-1, xmin=0, color="orange", alpha=0.5)
    fig.subplots_adjust(wspace=0.08, hspace=0.3)
    plt.show()


def ch2_tstt():
    nd = Network(name="Nguyen-Dupuis", sst=1, model="UE", from_file=False)
    nd.generate_total_path_set()
    # Basic data
    ue_tstt, so_tstt = 93525, 88866
    two_s = [2, 2, 2, 2]
    two_tstt, _ = lower_problem(nd, two_s)
    three_s = [3, 3, 3, 3]
    three_tstt, _ = lower_problem(nd, three_s)
    four_s = [4, 4, 4, 4]
    four_tstt, _ = lower_problem(nd, four_s)
    five_s = [5, 5, 5, 5]
    five_tstt, _ = lower_problem(nd, five_s)
    fig, ax = plt.subplots(figsize=(10, 6))
    x = [1, 2, 3, 4]
    height = [two_tstt, three_tstt, four_tstt, five_tstt]
    # plot the bar
    ax.bar(x=x, height=height, width=0.6, alpha=0.8, color="#87CEEB")
    # set x and y ticks and labels
    ax.set_xlim(0, 5)
    ax.set_xticks(range(1, 5))
    ax.set_xticklabels([f"{two_s}", f"{three_s}", f"{four_s}", f"{five_s}"], fontsize=12)
    ax.set_ylim(85000, 102000)
    ax.set_yticks([85000, 88866, 90000, 93525, 95000, 100000])
    ax.set_xlabel("Information provision strategy", fontsize=15)
    ax.set_ylabel("Total system travel time", fontsize=15)
    # remove the axis line
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # add the horizontal gird
    ax.yaxis.grid(True, which='major', color='gray', linestyle='-', linewidth=1.2, alpha=0.5)
    ax.set_axisbelow(True)  # 确保网格线在条形图下方
    # remove the ticks
    ax.yaxis.set_tick_params(length=0)  # 设置 y 轴刻度长度为 0
    # plot the UE and SO
    ax.hlines(ue_tstt, xmax=5, xmin=0, color="#FFA500")
    ax.text(4.35, ue_tstt+200, "TSTT under UE", color='orange', fontsize=12)
    ax.hlines(so_tstt, xmax=5, xmin=0, color="#FFA500")
    ax.text(4.35, so_tstt-600, "TSTT under SO", color='orange', fontsize=12)
    # show the improvement
    # add annotations
    ax.text(2.25, 98500, "Users from OD pair 1, 2, 3, 4 are provided with a, b, c, d"
                         "\npaths when information provision strategy is [a, b, c, d].", fontsize=12,
            bbox=dict(boxstyle='round,pad=0.3', edgecolor='black', facecolor='gray', alpha=0.3))
    # adjust the margin space
    plt.subplots_adjust(left=0.1, right=0.95, top=0.95, bottom=0.1)
    ax.set_title("TSTT of several information provision strategies", fontsize=20, pad=0)
    plt.show()


if __name__ == '__main__':
    ch2_tstt()
