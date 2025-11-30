import os
import json
import numpy as np
import matplotlib.pyplot as plt

real_problem_exp_paths = [
    "exp-11-30_121844-LLaMEA-gpt-4o-meta_surface",
    "exp-11-30_130916-LLaMEA-gpt-4o-meta_surface",
    "exp-11-30_135533-LLaMEA-gpt-4o-meta_surface",
    "exp-11-30_144728-LLaMEA-gpt-4o-meta_surface",
    "exp-11-30_153834-LLaMEA-gpt-4o-meta_surface",
]
with_gp_exp_paths = [
    "exp-11-29_175156-LLaMEA-gpt-4o-meta_surface_gp",
    "exp-11-30_134927-LLaMEA-gpt-4o-meta_surface_gp",
    "exp-11-30_160156-LLaMEA-gpt-4o-meta_surface_gp",
]


def parse_json(exp_path, with_gp=False):
    y = []
    f = open(f"{exp_path}/log.jsonl", "r")
    lines = f.readlines()
    counter = 0
    for line in lines:
        counter += 1
        content = json.loads(line)
        fit = content["fitness"] if not np.isinf(content["fitness"]) else 0.
        if not with_gp:
            y += [fit]
        else:
            if counter % 10 == 0:
                y += [fit]
    return y


colors = ['b', 'r', 'g',
          # 'c',
          'b', 'r', 'g', 'b', 'r', 'g']
linestyles = ['solid', 'solid', 'solid',
              # 'solid',
              'dotted', 'dotted', 'dotted', 'dashed', 'dashed', 'dashed']
labels = ["LLaMEA without GP functions", "LLaMEA with GP functions"]

y_real = []
y_gp = []
for exp_path in real_problem_exp_paths:
    y = parse_json(exp_path)
    y_real += [y]
for exp_path in with_gp_exp_paths:
    y = parse_json(exp_path, True)
    y_gp += [y]
y_real = np.array(y_real)
y_gp = np.array(y_gp)

mean_aucs = np.array([np.mean(y_real, axis=0), np.mean(y_gp, axis=0)])
std_aucs = np.array([np.std(y_real, axis=0), np.std(y_gp, axis=0)])
x = np.arange(mean_aucs.shape[1])
for i in range(mean_aucs.shape[0]):
    plt.plot(x, mean_aucs[i], color=colors[i], linestyle=linestyles[i],
             label=labels[i])
    plt.fill_between(x, mean_aucs[i] - std_aucs[i], mean_aucs[i] +
                     std_aucs[i], color=colors[i], alpha=0.05)
    # plt.fill_between(x, 0, 1, where=error_bars, color='r', alpha=0.2)
    plt.ylim(0.0, 1.)
    # plt.xlim(0, 100)
    # plt.plot(x, mean_2, 'r-', label='mean_2')
    # plt.fill_between(x, mean_2 - std_2, mean_2 + std_2, color='r', alpha=0.2)
plt.legend()
plt.tight_layout()
plt.savefig(f"results/comparison_gp_plot.png")
plt.close()
