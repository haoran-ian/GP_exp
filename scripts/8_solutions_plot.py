# fmt: off
import os
import sys
import math
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
sys.path.insert(0, os.getcwd())
import problems.meta_surface_solver.Meta_Generator as Gnr
from matplotlib import rcParams
# fmt: on
rcParams.update({'font.size': 14})

################################################################################
# meta_surface
def mat_inpso(b, n):
    a = np.zeros((2 * n, 2 * n))
    jj = 0
    for i in range(n):
        for j in range(i + 1):
            c = b[jj]
            a[i, j] = c
            a[j, i] = c
            a[j, 2 * n - 1 - i] = c
            a[i, 2 * n - 1 - j] = c
            a[2 * n - 1 - i, 2 * n - 1 - j] = c
            a[2 * n - 1 - j, 2 * n - 1 - i] = c
            a[2 * n - 1 - j, i] = c
            a[2 * n - 1 - i, j] = c
            jj += 1
    return a


meta_surface_solution = np.load("data/solutions/meta_surface_solution.npy")
meta_surface_respond = np.load("data/solutions/meta_surface_respond.npy")
meta_surface_solution = np.where(meta_surface_solution <= 0.5, 0., 1.)
n = 9
mat_in = mat_inpso(meta_surface_solution, n)
x = np.linspace(0, 1, 100)
y1 = 1 - 2 * (x - 0.5) ** 2
y2 = np.sqrt(meta_surface_respond[0]**2 + meta_surface_respond[1]**2)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
im = ax1.imshow(mat_in, cmap='gray')
ax1.axis('off')
ax1.set_title("Solution")
ax2.plot(x, y1, label='target', color='red', linewidth=2)
ax2.plot(x, y2, label='magnitude', color='blue', linewidth=2)
ax2.set_xlim(0, 1)
ax2.set_xticks([])
ax2.set_yticks([])
ax2.set_title("Comparison of the magnitude of \nour solution with the target")
ax2.legend(loc='upper right')
ax2.grid(True, alpha=0.3)
pos1 = ax1.get_position()
pos2 = ax2.get_position()
new_height = min(pos1.height, pos2.height)
new_y = min(pos1.y0, pos2.y0)
ax1.set_position([pos1.x0, new_y, pos1.width, new_height])
ax2.set_position([pos2.x0, new_y, pos2.width, new_height])
ax1.set_aspect('equal', adjustable='box')
# plt.tight_layout()
plt.savefig("results/meta_surface_solution.png")

################################################################################
# bragg