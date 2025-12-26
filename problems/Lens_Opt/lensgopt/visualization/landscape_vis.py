import json
import os
from pathlib import Path
import subprocess

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import CloughTocher2DInterpolator
from palettable.colorbrewer.diverging import RdYlBu_11 as colorblind_safe_cmap


def sample_plane(x1, x2, x3, npoints, K=20):
    d12_bg = np.linalg.norm(x1 - x2)
    d13_bg = np.linalg.norm(x1 - x3)
    d23_bg = np.linalg.norm(x2 - x3)
    pos1, pos2, pos3 = 0, 1, 2
    if d13_bg > d12_bg and d13_bg > d23_bg:
        print("x2 is top")
        x2, x3 = x3, x2
        pos2, pos3 = pos3, pos2
    elif d23_bg > d12_bg and d23_bg > d13_bg:
        print("x1 is top")
        x1, x2, x3 = x2, x3, x1
        pos1, pos2, pos3 = pos2, pos3, pos1
    else:
        print("x3 is top")
    c12 = x2 - x1
    c13 = x3 - x1
    c = c13 - np.inner(c12, c13) / np.linalg.norm(c12) ** 2 * c12
    e1 = c12 / np.linalg.norm(c12)
    e2 = c / np.linalg.norm(c)

    step_e1 = np.linalg.norm(x2 - x1) / npoints
    step_e2 = np.inner(c13, e2) / npoints

    plane = np.zeros((2 * K + npoints, 2 * K + npoints, len(x1)))
    for i in range(-K, npoints + K):
        for j in range(-K, npoints + K):
            plane[i + K][j + K] = x1 + e1 * j * step_e1 + e2 * i * step_e2
    x_coords = np.zeros((3, 2), dtype=int)
    x_coords[pos1][0] = K
    x_coords[pos1][1] = K
    x_coords[pos2][0] = K
    x_coords[pos2][1] = npoints + K
    x_coords[pos3][0] = npoints + K
    x_coords[pos3][1] = K + round(np.inner(c13, e1) / step_e1)
    return plane, x_coords


def calc_top_coord(x1, x2, x3, npoints):
    d12_bg = np.linalg.norm(x1 - x2)
    d13_bg = np.linalg.norm(x1 - x3)
    d23_bg = np.linalg.norm(x2 - x3)
    pos1, pos2, pos3 = 0, 1, 2
    if d13_bg > d12_bg and d13_bg > d23_bg:
        x2, x3 = x3, x2
        pos2, pos3 = pos3, pos2
    elif d23_bg > d12_bg and d23_bg > d13_bg:
        x1, x2, x3 = x2, x3, x1
        pos1, pos2, pos3 = pos2, pos3, pos1
    c12 = x2 - x1
    c13 = x3 - x1
    c = c13 - np.inner(c12, c13) / np.linalg.norm(c12) ** 2 * c12
    e1 = c12 / np.linalg.norm(c12)
    e2 = c / np.linalg.norm(c)
    step_e1 = np.linalg.norm(x2 - x1) / npoints
    step_e2 = np.inner(c13, e2) / npoints
    i = npoints
    j = round(np.inner(c13, e1) / step_e1)
    return x1 + e1 * j * step_e1 + e2 * i * step_e2, x3


def log_plane_values(p1, p2, p3, npoints, K, coords, Z, C, filename):
    with open(filename, "w") as file:
        print("p1", *p1, sep=",", file=file)
        print("p2", *p2, sep=",", file=file)
        print("p3", *p3, sep=",", file=file)
        print("coord_p1", *coords[0], sep=",", file=file)
        print("coord_p2", *coords[1], sep=",", file=file)
        print("coord_p3", *coords[2], sep=",", file=file)
        print("N", npoints, sep=",", file=file)
        print("K", K, sep=",", file=file)
        print("Nlines", (npoints + 2 * K) ** 2, file=file)
        for i in range(len(Z)):
            for j in range(len(Z[i])):
                print(Z[i][j], file=file)


def log_json(p1, p2, p3, npoints, K, coords, Z, C, filename):
    tp = lambda a: a.tolist() if type(a) is np.ndarray else a
    data = {
        "p1": tp(p1),
        "p2": tp(p2),
        "p3": tp(p3),
        # 'coord_p1': tp(coords[0]), 'coord_p2': tp(coords[1]), 'coord_p3': tp(coords[2]),
        "coords": tp(coords),
        "npoints": int(npoints),
        "K": int(K),
        "sz": int((npoints + 2 * K) ** 2),
        "Z": tp(Z),
        "C": tp(C),
    }
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def parse_plane_values(filename):
    with open(filename, "r") as file:
        cnt = 1
        coords = np.zeros((3, 2), dtype=int)
        while cnt < 10:
            line = file.readline()
            if cnt == 1:
                p1 = np.array(line.split(",")[1:], dtype=float)
            if cnt == 2:
                p2 = np.array(line.split(",")[1:], dtype=float)
            if cnt == 3:
                p3 = np.array(line.split(",")[1:], dtype=float)
            if cnt == 4:
                coords[0] = np.array(line.split(",")[1:], dtype=int)
            if cnt == 5:
                coords[1] = np.array(line.split(",")[1:], dtype=int)
            if cnt == 6:
                coords[2] = np.array(line.split(",")[1:], dtype=int)
            if cnt == 7:
                npoints = int(line.split(",")[1])
            if cnt == 8:
                K = int(line.split(",")[1])
            cnt += 1
        plane, _ = sample_plane(p1, p2, p3, npoints, K)
        Z = np.zeros((npoints + 2 * K, npoints + 2 * K))
        for i in range(npoints + 2 * K):
            for j in range(npoints + 2 * K):
                Z[i][j] = float(file.readline())
    return plane, Z, coords


def parse_json(filename):
    npa = lambda a: np.array(a)
    with open(filename, "r") as f:
        data = json.load(f)
        plane, _ = sample_plane(
            npa(data["p1"]),
            npa(data["p2"]),
            npa(data["p3"]),
            data["npoints"],
            data["K"],
        )
    return plane, np.array(data["Z"]), np.array(data["coords"])


def parse_json_c(filename):
    npa = lambda a: np.array(a)
    with open(filename, "r") as f:
        data = json.load(f)
        plane, _ = sample_plane(
            npa(data["p1"]),
            npa(data["p2"]),
            npa(data["p3"]),
            data["npoints"],
            data["K"],
        )
        C = data["C"]
    return plane, np.array(data["Z"]), np.array(data["coords"]), C


def decomp_constr_matrix(C, Z):
    decomp = []
    for k in range(len(C[0][0])):
        M = np.zeros((len(C), len(C[0])))
        for i in range(len(C)):
            for j in range(len(C[0])):
                M[i][j] = C[i][j][k]
        decomp.append(M)
    er = np.copy(Z)
    er = er**2 - sum(decomp)
    return er, decomp


def plt_config():
    plt.style.use("default")
    plt.style.use("seaborn-v0_8-poster")
    current_file = Path(__file__).resolve()
    current_folder = current_file.parent
    with open(current_folder.parent / "utils" / "latex-preambula.tex", "r") as f:
        latex_preambula = f.read()
    plt.rcParams["text.usetex"] = True
    plt.rc("text.latex", preamble=latex_preambula)
    mpl.rcParams["pdf.fonttype"] = 42
    mpl.rcParams["ps.fonttype"] = 42
    mpl.rcParams["font.size"] = 15
    plt.rcParams["axes.grid"] = True
    # plt.rcParams['grid.linestyle'] = '--'
    plt.rcParams["grid.linestyle"] = (0, (5, 5))
    plt.rcParams["grid.linewidth"] = 0.5
    # print(plt.rcParams.keys())


def crop_pdf(pdfname):
    subprocess.run(["pdfcrop", pdfname])


class LensSurface:
    def __init__(self, c, th, material):
        self.c = c
        self.th = th
        self.material = material


def build_lens_design(lenses):
    for i, lens in enumerate(lenses):
        r = 1 / lens.c
        if i % 2 == 0:
            x = np.linspace(-lens.c, 0, 100)


def interpalate(X, Y, Z, N_inter, coords):
    n = len(X)
    m = len(X[0])
    segm = [0, 1]
    bx1 = segm[0] + (segm[1] - segm[0]) / n / 2
    bx2 = segm[0] + (segm[1] - segm[0]) / m / 2
    to_x1 = lambda num: num / n + bx1
    to_x2 = lambda num: num / m + bx2
    p, v = [], []
    for i in range(len(X)):
        for j in range(len(X[0])):
            p.append([to_x1(X[i][j]), to_x2(Y[i][j])])
            v.append(Z[i][j])
    interp = CloughTocher2DInterpolator(p, v)
    print("Interpolation trained", flush=True)
    s = (segm[1] - segm[0]) / N_inter / 2
    x1i = np.linspace(to_x1(0) + s, to_x1(n - 1) - s, N_inter).tolist()
    x2i = np.linspace(to_x2(0) + s, to_x2(m - 1) - s, N_inter).tolist()
    for i in range(n):
        x1i.append(to_x1(X[i][0]))
    for i in range(m):
        x2i.append(to_x2(Y[0][i]))
    x1i.sort()
    x2i.sort()
    if not coords is None:
        coordsi = np.copy(coords)
        eps = (segm[1] - segm[0]) / (N_inter + n + m) / 10
        for i in range(len(coordsi)):
            for j in range(len(coords[i])):
                if j == 0:
                    ci_xj = to_x1(coords[i][0])
                    xi = x1i
                else:
                    ci_xj = to_x2(coords[i][1])
                    xi = x2i
                for k in range(len(xi)):
                    if abs(ci_xj - xi[k]) < eps:
                        coordsi[i][j] = k
    else:
        coordsi = None
    X1i, X2i = np.meshgrid(x1i, x2i)
    X1i, X2i = X1i.T, X2i.T
    Zi = interp(X1i, X2i)
    print("Interpolation done", flush=True)
    print("New matrix size", Zi.shape)
    X1i, X2i = np.meshgrid([i for i in range(len(x1i))], [i for i in range(len(x2i))])
    X1i, X2i = X1i.T, X2i.T
    return X1i, X2i, Zi, coordsi, interp


def build_surface(
    X,
    Y,
    Z,
    coords,
    subfig,
    filename,
    title=None,
    show=False,
    is_lines=True,
    is_inverse=False,
    is_connect=False,
    is_logscale=False,
    is_colorbar=False,
    is_interpalate=False,
    func_name="F",
    N_inter=300,
):
    plt_config()
    # mycmap = mpl.cm.binary_r
    # mycmap = mpl.cm.jet
    mycmap = colorblind_safe_cmap.mpl_colormap
    print("Matrix size", Z.shape)
    if is_interpalate:
        X, Y, Z, coords, f_inter = interpalate(X, Y, Z, N_inter, coords)
    else:
        f_inter = None
    fig, ax = plt.subplots(subplot_kw={"projection": "3d", "computed_zorder": False})
    fig.set_size_inches(18.5, 10.5)
    fig.canvas.manager.set_window_title(filename)
    # surf = ax.plot_surface(X, Y, Z, cmap=cmap, alpha=0.6)
    if is_logscale:
        with np.errstate(divide="ignore"):
            Z1 = np.log(np.abs(Z))
            if Z1[Z1 <= float("-inf")].any():
                eps = 1e-15
                Z1 = Z + eps
                Z1 = np.log(np.abs(Z1))
                print(
                    f"log(0) encountered. Shifting all values by eps={eps:.2E}, ln(eps)={np.log(eps):.5f}",
                    flush=True,
                )
            Z = Z1
    if is_lines:
        # ls = mpl.colors.LightSource(270, 45)
        # rgb = ls.shade(Z, cmap=mycmap, vert_exag=0.1, blend_mode='soft')
        # ax.plot_surface(X, Y, Z, facecolors=rgb, shade=False, antialiased=False, linewidth=0, edgecolor='none',
        #     rcount=len(X), ccount=len(X[0]), alpha=1, zorder=3)
        ax.plot_surface(
            X,
            Y,
            Z,
            cmap=mycmap,
            antialiased=True,
            linewidth=10 / len(X),
            edgecolor="k",
            rcount=len(X),
            ccount=len(X[0]),
            alpha=0.5,
            zorder=3,
        )
    else:
        ax.plot_surface(
            X,
            Y,
            Z,
            cmap=mycmap,
            antialiased=True,
            linewidth=0.0,
            edgecolor=None,
            rcount=len(X),
            ccount=len(X[0]),
            alpha=0.5,
            zorder=3,
        )
    zax_min, zax_max = ax.get_zlim()
    h = zax_max - zax_min
    if is_inverse:
        zax_min, zax_max = zax_max + 2.5 * h, zax_min
    else:
        zax_min, zax_max = zax_min - 2.5 * h, zax_max
    ax.set_zlim(zax_min, zax_max)
    levels = 50
    ax.contourf(
        X,
        Y,
        Z,
        zdir="z",
        offset=zax_min,
        cmap=mycmap,
        extend="both",
        levels=levels,
        alpha=0.5,
        zorder=2,
    )
    ax.contour(
        X,
        Y,
        Z,
        zdir="z",
        offset=zax_min,
        cmap=mpl.cm.binary_r,
        linewidths=1,
        extend="both",
        levels=levels,
        alpha=0.5,
        zorder=2,
    )
    # sns.heatmap(Z, ax=ax, cmap=mycmap, )
    tick_values = ax.get_xticks()
    labels = [item.get_text() for item in ax.get_xticklabels()]
    for i in range(len(labels)):
        labels[i] = ""
    ax.set_xticks(tick_values)
    ax.set_xticklabels(labels)
    tick_values = ax.get_yticks()
    labels = [item.get_text() for item in ax.get_yticklabels()]
    for i in range(len(labels)):
        labels[i] = ""
    ax.set_yticks(tick_values)
    ax.set_yticklabels(labels)
    # ax.view_init(45, 0)
    FS = 15
    xax_min, xax_max = ax.get_xlim()
    yax_min, yax_max = ax.get_ylim()
    zax_min, zax_max = ax.get_zlim()
    text_shift = np.array(
        [
            (-xax_min + xax_max) * 0.01,
            (-yax_min + yax_max) * 0.01,
            (-zax_min + zax_max) * 0.01,
        ]
    )
    if not coords is None:
        for i in range(len(coords)):
            ax.scatter3D(
                [coords[i][0]],
                [coords[i][1]],
                [Z[coords[i][0]][coords[i][1]]],
                c="red",
                marker="*",
                linewidth=2,
                s=100,
                zorder=4,
            )
            sp = r"$\mathbf{" + f"{subfig}" + "}" + f"_{i + 1}$"
            if is_logscale:
                s = (
                    r"$\ln\!\br{"
                    + func_name
                    + r"(\mathbf{"
                    + f"{subfig}"
                    + "}"
                    + f"_{i + 1})"
                    + "}$"
                )
                print(
                    f"ln({func_name}({subfig}{i+1})) = {Z[coords[i][0]][coords[i][1]]:.5f}"
                )
            else:
                s = r"$" + func_name + r"(\mathbf{" + f"{subfig}" + "}" + f"_{i + 1})$"
                print(
                    f"{func_name}({subfig}{i+1}) = {Z[coords[i][0]][coords[i][1]]:.5f}"
                )
            # sp = r'$\Pi\!\br{\mathbf{' + f'{subfig}' + ' }' + f'_{i + 1}' + '}$'
            ax.text(
                coords[i][0] + text_shift[0],
                coords[i][1] + text_shift[1],
                Z[coords[i][0]][coords[i][1]] + text_shift[2],
                s,
                color="red",
                fontsize=FS,
                zorder=4,
            )
            ax.scatter(
                [coords[i][0]],
                [coords[i][1]],
                zax_min,
                c="red",
                marker=".",
                linewidth=2,
                s=100,
                zorder=4,
            )
            ax.text(
                coords[i][0] + text_shift[0],
                coords[i][1] + text_shift[1],
                zax_min,
                sp,
                color="red",
                fontsize=FS,
                zorder=4,
            )
    if is_connect and not coords is None:
        for i, j in [(0, 1), (1, 2), (0, 2)]:
            ax.plot(
                [coords[i][0], coords[j][0]],
                [coords[i][1], coords[j][1]],
                zax_min,
                c="red",
                linewidth=2,
                zorder=4,
            )
    ax.set_xlabel("$\mathbf{e}_" + f"{subfig}" + "^{(1)}$", fontsize=FS, labelpad=-10)
    ax.set_ylabel("$\mathbf{e}_" + f"{subfig}" + "^{(2)}$", fontsize=FS, labelpad=-10)
    if title:
        ax.set_title(title)
    if is_logscale:
        # ax.set_zlabel(r'$\ln(F(\mathbf{z}))$', fontsize=FS, labelpad=-10)
        ax.set_zlabel(r"$\ln(" + func_name + r"(\mathbf{z}))$", fontsize=FS, labelpad=5)
        # ax.tick_params(axis='z', which='major', pad=-20)
    else:
        ax.set_zlabel(f"${func_name}" + r"(\mathbf{z})$", fontsize=FS, labelpad=5)
    if is_colorbar:
        zmin, zmax = Z.min(), Z.max()
        sm = plt.cm.ScalarMappable(cmap=mycmap, norm=plt.Normalize(zmin, zmax))
        fig.colorbar(sm, ax=ax, ticks=np.linspace(zmin, zmax, 6), shrink=0.7)
    if show:
        plt.show()
    else:
        plt.show()
        pdffilename = os.path.splitext(filename)[0] + ".pdf"
        if is_interpalate:
            pdffilename = os.path.splitext(filename)[0] + "_inter.pdf"
        if is_logscale:
            pdffilename = os.path.splitext(filename)[0] + "_log.pdf"
        fig.savefig(pdffilename)
        crop_pdf(pdffilename)
    plt.close()
    return f_inter


def vis_plane_vals(plane, Z, coords, subfig, filename, **kwargs):
    idx = [i for i in range(len(plane))]
    idy = [i for i in range(len(plane[0]))]
    X, Y = np.meshgrid(idx, idy)
    X, Y = X.T, Y.T
    return build_surface(X, Y, Z, coords, subfig, filename, **kwargs)


class Lvis:
    def __init__(self, lb=None, ub=None, F=None, constraint_names=None) -> None:
        self.lb = lb
        self.ub = ub
        self.F = F
        self.constraint_names = constraint_names

    def to_bounds(self, p):
        for k in range(len(p)):
            p[k] = max(self.lb[k], min(self.ub[k], p[k]))
        return p

    def sample_line(self, x1, x2, npoints):
        samples = []
        for i in range(-2, npoints + 2):
            samples.append(x2 * i / npoints + x1 * (1 - i / npoints))
        values = [self.F(self.to_bounds(x)) for x in samples]
        return samples, values

    def build_chart_line(self, line, values):
        x = [i for i in range(len(line))]
        fig = plt.figure()
        plt.plot(x, values, c="red")
        plt.show()
        # fig.savefig(f'line.png')
        # plt.close()

    def do1D(self, p1, p2):
        line, values = self.sample_line(p1, p2, 100)
        self.build_chart_line(line, values)

    def find_Npoints(self, x1, x2, x3, K, MAXSAMPLE=10**6, MAXDIST=2e-4):
        d12_bg = np.linalg.norm(x1 - x2)
        d13_bg = np.linalg.norm(x1 - x3)
        d23_bg = np.linalg.norm(x2 - x3)
        pos1, pos2, pos3 = 0, 1, 2
        if d13_bg > d12_bg and d13_bg > d23_bg:
            x2, x3 = x3, x2
            pos2, pos3 = pos3, pos2
        elif d23_bg > d12_bg and d23_bg > d13_bg:
            x1, x2, x3 = x2, x3, x1
            pos1, pos2, pos3 = pos2, pos3, pos1
        c12 = x2 - x1
        c13 = x3 - x1
        c = c13 - np.inner(c12, c13) / np.linalg.norm(c12) ** 2 * c12
        e1 = c12 / np.linalg.norm(c12)
        e2 = c / np.linalg.norm(c)
        r = int(np.sqrt(MAXSAMPLE) - 2 * K) + 1
        print(f"Max npoints {r}")
        right = self.F(x3)
        min_x_diff, npoints_min_x_diff, approx_min_x_diff = float("inf"), None, None
        for npoints in range(100, r + 1):
            step_e1 = np.linalg.norm(x2 - x1) / npoints
            step_e2 = np.inner(c13, e2) / npoints
            i = npoints
            j = round(np.inner(c13, e1) / step_e1)
            top_approx = x1 + e1 * j * step_e1 + e2 * i * step_e2
            x_diff = np.linalg.norm(top_approx - x3)
            if x_diff < MAXDIST:
                print(
                    f"Npoints for MAXDIST={MAXDIST:.5f} found!\nNpoints {npoints}\nTop approx x error {x_diff:.5f}\nTop approx y error {abs(self.F(top_approx) - right):.5f}"
                )
                return npoints
            if x_diff < min_x_diff:
                min_x_diff, npoints_min_x_diff, approx_min_x_diff = (
                    x_diff,
                    npoints,
                    top_approx,
                )
        print(
            f"Npoints for MAXDIST={MAXDIST:.5f} NOT found =(\nNpoints {npoints_min_x_diff}\nTop approx x error {min_x_diff:.5f}\nTop approx y error {abs(self.F(approx_min_x_diff) - right):.5f}"
        )
        return npoints_min_x_diff

    def eval_plane(self, plane):
        Z = np.zeros((len(plane), len(plane[0])))
        C = [[None for i in range(len(plane[0]))] for j in range(len(plane))]
        total = len(plane)
        for i, line in enumerate(plane):
            for j, p in enumerate(line):
                # p = self.to_bounds(p)
                Z[i][j] = self.F(p)
                C[i][j] = self.F.constraints()
            print(
                f"row {i+1}/{total}: min={Z[i].min():.5f}, max={Z[i].max():.5f}",
                flush=True,
            )
        return Z, C

    def precomp2D(self, p1, p2, p3, precomp_file, max_dist, K):
        npoints = self.find_Npoints(p1, p2, p3, K=K, MAXDIST=max_dist)
        plane, coords = sample_plane(p1, p2, p3, npoints, K=K)
        Z, C = self.eval_plane(plane)
        log_json(p1, p2, p3, npoints, K, coords, Z, C, precomp_file)

    def precomp2D_ortogonal(
        self,
        reference: np.ndarray,
        v1: np.ndarray,
        v2: np.ndarray,
        dist_one_step: float,
        precomp_file: str,
        n_steps: int = 75,
    ):
        plane = np.zeros((2 * n_steps + 1, 2 * n_steps + 1, len(v1)))
        for i in range(-n_steps, n_steps + 1):
            for j in range(-n_steps, n_steps + 1):
                plane[n_steps + i][n_steps + j] = (
                    reference + i * dist_one_step * v1 + j * dist_one_step * v2
                )
        coords = np.zeros((3, 2), dtype=int)
        coords[0] = np.array([n_steps, n_steps])
        coords[1] = np.array([2 * n_steps, n_steps])
        coords[2] = np.array([n_steps, 2 * n_steps])
        Z, C = self.eval_plane(plane)
        log_json(
            reference,
            reference + n_steps * dist_one_step * v1,
            reference + n_steps * dist_one_step * v2,
            2 * n_steps + 1,
            0,
            coords,
            Z,
            C,
            precomp_file,
        )

    def do2D(self, precomp_file, **kwargs):
        print(f"Processing {precomp_file} ...", flush=True)
        print("Config:", kwargs.items())
        plane, Z, coords = parse_json(precomp_file)
        return vis_plane_vals(plane, Z, coords, "z", precomp_file, **kwargs)

    def do2D_c(self, precomp_file, **kwargs):
        print(f"Processing constraints {precomp_file} ...", flush=True)
        print("Config:", kwargs.items())
        plane, Z, coords, CC = parse_json_c(precomp_file)
        er, Cs = decomp_constr_matrix(CC, Z)
        pdffilename = os.path.splitext(precomp_file)[0] + f"_er.pdf"
        kwargs["func_name"] = "f"
        kwargs_1 = kwargs.copy()
        if not "title" in kwargs.keys():
            kwargs_1["title"] = "Error function"
        vis_plane_vals(plane, er, coords, "z", pdffilename, **kwargs_1)
        for i, C in enumerate(Cs):
            pdffilename = os.path.splitext(precomp_file)[0] + f"_c{i+1}.pdf"
            kwargs["func_name"] = f"g_{i+1}"
            kwargs_1["func_name"] = f"g_{i+1}"
            if not "title" in kwargs.keys():
                kwargs_1["title"] = (
                    f"Constraint function {i+1}: {self.constraint_names[i]}"
                )
            vis_plane_vals(plane, C, coords, "z", pdffilename, **kwargs_1)
