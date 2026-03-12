# fmt: off
import os
import sys
import cv2
import math
import numpy as np
import seaborn as sns
import PyMoosh as pm
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
sys.path.insert(0, os.getcwd())
import problems.meta_surface_solver.Meta_Generator as Gnr
from matplotlib import rcParams
from matplotlib.gridspec import GridSpec
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from problems.photovotaic_problems.brag_mirror import brag_mirror
from problems.photovotaic_problems.sophisticated_antireflection_design import sophisticated_antireflection_design
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
ax2.plot(x, y1, label='target', color='C1', linewidth=2)
ax2.plot(x, y2, label='our solution', color='C0', linewidth=2)
ax2.set_xlim(0, 1)
ax2.set_xticks([])
ax2.set_yticks([])
ax2.set_title("Comparison of the magnitude of \nour solution with the target")
ax2.legend(loc='lower center')
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
plt.close()

################################################################################
# mini-Bragg


def Bragg_solution_plot(nb_layers, x):
    target_wl = 600.0  # nm
    mat_env = 1.0      # materials: ref. index
    mat1 = 1.4
    mat2 = 1.8
    prob = brag_mirror(nb_layers, target_wl, mat_env, mat1, mat2)
    struct = prob.setup_structure(x)
    # struct.plot_stack()

    wls = np.linspace(400, 1000, 121)
    R = np.zeros_like(wls)
    for i, wl in enumerate(wls):
        _, _, R[i], _ = pm.coefficient(struct, wl, incidence=0, polarization=0)

    fig = plt.figure(figsize=(12, 6))
    gs = GridSpec(1, 2, width_ratios=[1, 1.2], wspace=0.3)
    ax1 = fig.add_subplot(gs[0])
    img = cv2.imread(f"results/Bragg_struct_{nb_layers}.png")
    ax1.imshow(img)
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.set_ylabel('D (nm)', fontsize=12)
    ax1.set_title('Structure', fontsize=14, pad=15)
    ax1.set_aspect('auto')
    ax2 = fig.add_subplot(gs[1])
    ax2.plot(wls, R)
    ax2.axvline(600, color='k', dashes=[2, 2])
    # ax2.set_xlim(0, 1)
    # ax2.set_ylim(-1.2, 1.2)
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.set_xlabel('wavelength (nm)', fontsize=12)
    ax2.set_ylabel('reflectivity', fontsize=12)
    ax2.set_title('Maximize reflectivity at 600 nm wavelength',
                  fontsize=14, pad=15)
    # ax2.set_xticks([])
    # ax2.set_yticks([])
    plt.tight_layout()
    plt.savefig(f"results/Bragg_solution_{nb_layers}.png")
    plt.close()


nb_layers = 10
x = [85.19483689362964, 99.79911419199105, 91.34051173236992, 97.82313377774571,
     83.90537947772964, 114.30619104668908, 79.2949047602944, 108.6909038265911,
     81.45774019766088, 111.35922342316027]
Bragg_solution_plot(nb_layers, x)

nb_layers = 20
x = [88.58027445576352, 93.34220990169761, 82.8843247981396, 106.91889345235505,
     85.0511695682233, 104.12473116555176, 97.00123240234731, 90.5318452080175,
     94.49036976381628, 100.35093656917297, 85.50450982919119, 101.00823222093676,
     92.26352553487027, 94.32205495979517, 88.0347552662337, 108.1119642666936,
     79.73467323714122, 121.11432375602617, 81.58705269771626, 99.57663579576867]
Bragg_solution_plot(nb_layers, x)

################################################################################
# photovotaic
nb_layers = 10
min_thick = 30
max_thick = 250
wl_min = 375
wl_max = 750
prob = sophisticated_antireflection_design(nb_layers, min_thick, max_thick,
                                           wl_min, wl_max)
x = [75.78507551727988, 115.96961820193481, 146.46929161165156, 123.08495754805755,
     146.49350057099974, 127.76339155776908, 129.5708409243326, 134.7108088449702,
     92.65769119379769, 72.04720592428609]
struct = prob.setup_structure(x)
_, _, _, wl, _, spec_A = pm.photo(
    struct, incidence=0, polarization=0,
    wl_min=wl_min, wl_max=wl_max,
    active_layers=len(x)+1, number_points=100)
struct_zeros = prob.setup_structure(np.zeros_like(x))
_, _, _, wl, _, spec_A_zero = pm.photo(
    struct_zeros, incidence=0, polarization=0,
    wl_min=wl_min, wl_max=wl_max,
    active_layers=len(x)+1, number_points=100)
prob_for_plot = sophisticated_antireflection_design(nb_layers, min_thick, max_thick,
                                                    wl_min, wl_max, thick_aSi=0)
struct_for_plot = prob_for_plot.setup_structure(x)
# struct_for_plot.plot_stack()

fig = plt.figure(figsize=(12, 6))
gs = GridSpec(1, 2, width_ratios=[1, 1.2], wspace=0.3)
ax1 = fig.add_subplot(gs[0])
img = cv2.imread(f"results/photovoltaic_struct.png")
ax1.imshow(img)
ax1.set_xticks([])
ax1.set_yticks([])
ax1.set_ylabel('D (nm)', fontsize=12)
ax1.set_title('Structure', fontsize=14, pad=15)
ax1.set_aspect('auto')
ax2 = fig.add_subplot(gs[1])
ax2.plot(wl, spec_A, color='C0', label='our solution')
ax2.plot(wl, spec_A_zero, color='C1', label='no AR coating')
ax2.grid(True, alpha=0.3, linestyle='--')
ax2.set_xlabel('wavelength (nm)', fontsize=12)
ax2.set_ylabel('absorption', fontsize=12)
ax2.set_title('Maximize absorption within desired wavelength range',
              fontsize=14, pad=15)
ax2.legend()
# ax2.set_xticks([])
# ax2.set_yticks([])
plt.tight_layout()
plt.savefig(f"results/photovoltaic_solution.png")
plt.close()
