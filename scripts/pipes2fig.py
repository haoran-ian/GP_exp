# fmt: off
import os
import sys
import time
import cv2
import ioh
import torch
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cdist
sys.path.insert(0, os.getcwd())
from problems.fluid_dynamics.topodiff import dist_util, logger
from problems.fluid_dynamics.topodiff.script_util import create_regressor
from gp_fgenerator.sampling import sampling
# from gp_fgenerator.compute_ela import bootstrap_ela
from gp_fgenerator.create_pset import *
from gp_fgenerator.gp_fgenerator import GP_func_generator
from gp_fgenerator.utils import runParallelFunction, read_pickle
from gp_fgenerator.utils import export_pickle, dataCleaning, dropFeatCorr
from LLaMEA.misc import aoc_logger, correct_aoc, OverBudgetException
from LLaMEA.llamea import LLaMEA, OpenAI_LLM
from test_alg import DE_PSO_Optimizer
# fmt: on


# Set up directory

# root_dir = '/content/drive/MyDrive/NTU/Programmes/FYP/IHPC/Test/' #Please change directory for file saving here
# os.chdir(root_dir)

# Existing definitions

device = 'cuda:0'


class Pipe:
    def __init__(self, entry_point, exit_point, path, thickness, control_pts):
        self.entry_point = entry_point
        self.exit_point = exit_point
        self.path = path
        self.thickness = thickness
        self.control_pts = control_pts


class pipes_topology:
    def __init__(self, num_pipes, start_points, end_points, img_res=(256, 512),
                 rect_width=2.0, rect_height=1.0, num_of_bezier_points=1500):
        self.num_pipes = num_pipes
        self.rect_width = rect_width
        self.rect_height = rect_height
        self.num_of_bezier_points = num_of_bezier_points
        self.dim = num_pipes * 19
        if len(start_points) != num_pipes or len(end_points) != num_pipes:
            print(
                f"Number of start points {len(start_points)} or end points {len(end_points)} is not equal to num_pipes {num_pipes}.")
            return -1
        self.start_points = start_points
        self.end_points = end_points
        self.img_res = img_res
        x = np.linspace(0 + self.rect_width / (
            self.img_res[1]*2), self.rect_width - self.rect_width/(self.img_res[1]*2), self.img_res[1])
        y = np.linspace(0 + self.rect_height / (
            self.img_res[0]*2), self.rect_height - self.rect_height/(self.img_res[0]*2), self.img_res[0])
        X, Y = np.meshgrid(x, y)
        self.grid_ls = np.column_stack((X.ravel(), Y.ravel()))
        self.eps = 1/self.img_res[0]
        self.lb = [0.08 for _ in range(self.num_pipes)]
        self.ub = [0.2 for _ in range(self.num_pipes)]
        for _ in range(self.num_pipes):
            self.lb += [0.05, -0.02, 0.1, -0.05, 0.2, -0.1, 0.3, 0.2,
                        0.6, 0.2, 1.4, 0.2, 0.2, -0.1, 0.1, -0.05, 0.05, -0.02]
            self.ub += [0.1, 0.02, 0.2, 0.05, 0.3, 0.1, 0.6, 0.8,
                        1.4, 0.8, 1.7, 0.8, 0.3, 0.1, 0.2, 0.05, 0.1, 0.02]
        self.lb = np.array(self.lb)
        self.ub = np.array(self.ub)
        X_pca_train = np.random.uniform(self.lb, self.ub,
                                        size=(1000*self.dim, self.lb.shape[0]))
        self.pca_dim = 2 + 7 * num_pipes
        self.pca = PCA(n_components=self.pca_dim)
        self.X_pca = self.pca.fit_transform(X_pca_train)
        pca_min = np.min(self.X_pca, axis=0)
        pca_max = np.max(self.X_pca, axis=0)
        self.pca_lb = pca_min - 0.1
        self.pca_ub = pca_max + 0.1
        regressor_p_diff_path = '/home/ubuntu/GP_Compare/problems/fluid_dynamics/model025600.pt'
        logger.log("loading regressor_p_diff...")
        self.regressor_p_diff = create_regressor(regressor_depth=4, in_channels=1,
                                                 image_size=64, regressor_use_fp16=False,
                                                 regressor_width=128,
                                                 regressor_attention_resolutions="32,16,8",
                                                 regressor_use_scale_shift_norm=True,
                                                 regressor_resblock_updown=True,
                                                 regressor_pool="spatial")
        self.regressor_p_diff.load_state_dict(
            dist_util.load_state_dict(
                regressor_p_diff_path, map_location="cuda", weights_only=True)
        )
        self.regressor_p_diff.to(dist_util.dev())
        self.regressor_p_diff.eval()

    def __call__(self, x):
        x = np.clip(x, self.pca_lb, self.pca_ub)
        x_inverse = self.pca.inverse_transform(x)
        design = self.pipes2fig(x_inverse)
        design = cv2.resize(
            design, [64, 64], interpolation=cv2.INTER_AREA) * -1
        x_tensor = torch.tensor(design).float().to(device)
        y = self.regressor_p_diff(x_tensor.reshape(1, 1, 64, 64),
                                  torch.zeros(1, ).to(device))
        y = y.tolist()[0][0]
        print(y)
        return y

    def generate_pipe(self, x, i, thickness):
        start = self.start_points[i]
        end = self.end_points[i]
        control_points = [None for _ in range(9)]
        for j in range(3):
            control_points[j] = [start[0] + x[i][j][0], start[1] + x[i][j][1]]
            control_points[j+3] = [x[i][j+3][0], x[i][j+3][1]]
            control_points[j+6] = [end[0] - x[i]
                                   [j+6][0], end[1] + x[i][j+6][1]]
        control_points = np.array(control_points)
        t = np.linspace(0, 1, self.num_of_bezier_points)
        path_x = (1 - t)**8 * start[0] + 8 * (1 - t)**7 * t * control_points[0, 0] + 28 * (1 - t)**6 * t**2 * control_points[1, 0] + 56 * (1 - t)**5 * t**3 * control_points[2, 0] + 70 * (
            1 - t)**4 * t**4 * control_points[3, 0] + 56 * (1 - t)**3 * t**5 * control_points[4, 0] + 28 * (1 - t)**2 * t**6 * control_points[5, 0] + 8 * (1 - t) * t**7 * control_points[6, 0] + t**8 * end[0]
        path_y = (1 - t)**8 * start[1] + 8 * (1 - t)**7 * t * control_points[0, 1] + 28 * (1 - t)**6 * t**2 * control_points[1, 1] + 56 * (1 - t)**5 * t**3 * control_points[2, 1] + 70 * (
            1 - t)**4 * t**4 * control_points[3, 1] + 56 * (1 - t)**3 * t**5 * control_points[4, 1] + 28 * (1 - t)**2 * t**6 * control_points[5, 1] + 8 * (1 - t) * t**7 * control_points[6, 1] + t**8 * end[1]
        path = np.column_stack((path_x, path_y))
        pipe = Pipe(start, end, path, thickness, control_points)
        return pipe

    def sign_flipper_v2(self, sdf, grid_resolution, to_multiply, upper_path, lower_path):
        horizontal_grid = np.linspace(
            0 + 2/(grid_resolution[0]*2), 2 - 2/(grid_resolution[0]*2), grid_resolution[0])
        vertical_grid = np.linspace(
            0 + 1/(grid_resolution[1]*2), 1 - 1/(grid_resolution[1]*2), grid_resolution[1])
        for j in range(grid_resolution[0]):
            horizontal_grid_val = horizontal_grid[j]
            vert_upper_y = upper_path[(
                np.argmin(np.abs(upper_path[:, 0] - horizontal_grid_val))), 1]
            vert_lower_y = lower_path[(
                np.argmin(np.abs(lower_path[:, 0] - horizontal_grid_val))), 1]
            for i in range(grid_resolution[1]):
                if (vertical_grid[i] < vert_upper_y) and (vertical_grid[i] > vert_lower_y):
                    sdf[i, j] *= -1
                    to_multiply[i, j] = -1
        return sdf, to_multiply

    def combine_vof(self, vof1, vof2):
        combine_vof_12 = 0.25*(vof1 + 1)*(vof2 + 1) - 0.25*(vof1 + 1) * \
            (vof2 - 1) - 0.25*(vof1 - 1)*(vof2 + 1) - 0.25*(vof1 - 1)*(vof2 - 1)
        return combine_vof_12

    def pipes2fig(self, x):
        sdf_list = []
        points = x[self.num_pipes:].reshape(self.num_pipes, 9, 2)
        for i in range(self.num_pipes):
            pipe = self.generate_pipe(points, i, x[i])
            # Extract path and thickness
            path = pipe.path
            thickness = pipe.thickness

            # Find upper and lower path curves
            upper_path = path + np.array([0, thickness/2])
            lower_path = path - np.array([0, thickness/2])

            # Calculate distances to upper and lower paths
            upper_distances = np.min(cdist(self.grid_ls, upper_path), axis=1)
            lower_distances = np.min(cdist(self.grid_ls, lower_path), axis=1)

            sdf = np.minimum(upper_distances, lower_distances)
            sdf = sdf.reshape(self.img_res)

            # Initialise a numpy array for multiplication for each pipe
            to_multiply = np.ones(self.img_res)

            # Apply the sign flip logic
            sdf_pipe, to_multiply_pipe = self.sign_flipper_v2(
                sdf, (self.img_res[1], self.img_res[0]), to_multiply, upper_path, lower_path)

            sdf_list.append(sdf_pipe)

        cur_vof = np.tanh(-sdf_list[0] / self.eps)
        for i in range(len(sdf_list)-1):
            old_vof = np.copy(cur_vof)
            new_vof = np.tanh(-sdf_list[i+1] / self.eps)
            cur_vof = self.combine_vof(old_vof, new_vof)
        return cur_vof


def find_optimal_pca_components(X, threshold=0.95):
    pca = PCA()
    pca.fit(X)

    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
    optimal_components = np.argmax(cumulative_variance >= threshold) + 1

    return optimal_components, cumulative_variance


def feature_importance_heatmap(X, feature_names=None, annot=False, cmap='RdBu_r',
                               center=0, title=""):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    num_pipes = int(X.shape[1]/19)
    # n_components=int(num_pipes*7+2)
    n_components = 11
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)
    loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
    loadings_df = pd.DataFrame(
        loadings,
        columns=[f'PC{i+1}' for i in range(loadings.shape[1])],
        index=feature_names
    )
    plt.figure(figsize=(20, 8))
    sns.heatmap(loadings_df.T, annot=annot, square=True, cmap='coolwarm',
                center=0, fmt='.2f', linewidths=0.5)
    plt.title(
        f'PCA Factor Loadings Matrix\n{num_pipes} pipes and n_components={n_components}')
    plt.tight_layout()
    plt.savefig(f"results/{num_pipes}pipes_PCA_{n_components}_loadings.png")
    plt.close()


# def get_ela(dim, f, X, y, exp_name):
#     path_base = os.path.join(os.getcwd(), f"ela_{exp_name}")
#     if not (os.path.isdir(path_base)):
#         os.makedirs(path_base)
#     df_ela = bootstrap_ela(X, y, bs_ratio=0.8, bs_repeat=5,
#                            lower_bound=30., upper_bound=250.)
#     fid = f.meta_data.problem_id
#     filepath = os.path.join(path_base, f'ela_{fid}.csv')
#     df_ela.to_csv(filepath, index=False)
# # END DEF


# %%
def get_ela_normalize(dim, exp_name):
    df_ela = pd.read_csv(f"ela_{exp_name}/ela_1121.csv")
    df_ela = dataCleaning(df_ela, replace_nan=False, inf_as_nan=True,
                          col_allnan=False, col_anynan=False, row_anynan=False,
                          col_null_var=False, row_dupli=False, filter_key=[],
                          reset_index=False, verbose=False)
    dict_min = {}
    dict_max = {}
    dict_mean = {}
    dict_std = {}
    for ela in df_ela.keys():
        dict_min[ela] = df_ela[ela].min(numeric_only=True)
        dict_max[ela] = df_ela[ela].max(numeric_only=True)
        dict_mean[ela] = df_ela[ela].mean(numeric_only=True)
        dict_std[ela] = df_ela[ela].std(numeric_only=True)
    export_pickle(os.path.join(
        os.getcwd(), f'ela_{exp_name}', 'ela_min.pickle'), dict_min)
    export_pickle(os.path.join(
        os.getcwd(), f'ela_{exp_name}', 'ela_max.pickle'), dict_max)
    export_pickle(os.path.join(
        os.getcwd(), f'ela_{exp_name}', 'ela_mean.pickle'), dict_mean)
    export_pickle(os.path.join(
        os.getcwd(), f'ela_{exp_name}', 'ela_std.pickle'), dict_std)
# END DEF


# %%
def get_ela_corr(dim, exp_name):
    df_ela = pd.read_csv(f"ela_{exp_name}/ela_1121.csv")
    df_clean = dataCleaning(df_ela, replace_nan=False, inf_as_nan=True,
                            col_allnan=True, col_anynan=True, row_anynan=False,
                            col_null_var=True, row_dupli=False, filter_key=[
                                'pca.expl_var.cov_x', 'pca.expl_var.cor_x',
                                'pca.expl_var_PC1.cov_x',
                                'pca.expl_var_PC1.cor_x'],
                            reset_index=False, verbose=True)
    df_corr, df_pair = dropFeatCorr(
        df_clean, corr_thres=0.9, corr_method='pearson', mode='pair',
        ignore_keys=[], verbose=True)
    list_ela = list(df_corr.keys())
    export_pickle(os.path.join(
        os.getcwd(), f"ela_{exp_name}", 'ela_corr.pickle'), list_ela)
# END DEF


if torch.cuda.is_available():
    print(
        f"CUDA is available. PyTorch is using GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU device count: {torch.cuda.device_count()}")
    print(f"Current device index: {torch.cuda.current_device()}")
else:
    print("CUDA is not available. Using CPU.")
torch.cuda.empty_cache()
# Execution code starts here
api_key = os.getenv("OPENAI_API_KEY")
ai_model = "gpt-4o"
experiment_name = "topology_3"
llm = OpenAI_LLM(api_key, ai_model)
# Llama-3.2-1B-Instruct, Llama-3.2-3B-Instruct,
# Meta-Llama-3.1-8B-Instruct, Meta-Llama-3.1-70B-Instruct,
# CodeLlama-7b-Instruct-hf, CodeLlama-13b-Instruct-hf,
# CodeLlama-34b-Instruct-hf, CodeLlama-70b-Instruct-hf,

evaluation_id = 0


def test_func1(x):
    x = np.clip(x, 0, 0)
    y = neg(cos(sub(sub(prod_vec(mul(prod_vec(mul(7.040400702545974, 4.70841758564605)),
            4.70841758564605)), x), prod_vec(prod_vec(prod_vec(2.337443560678286))))))
    return np.mean(y)+0.7


def test_func2(x):
    y = neg(cos(sub(sub(prod_vec(mul(prod_vec(mul(7.040400702545974, 4.70841758564605)), 4.70841758564605)), cos(sub(sub(prod_vec(prod_vec(mul(prod_vec(mul(
        7.040400702545974, 4.70841758564605)), 4.70841758564605))), x), prod_vec(prod_vec(prod_vec(2.337443560678286)))))), prod_vec(prod_vec(prod_vec(2.337443560678286))))))
    return np.mean(y)+0.6


def test_func3(x):
    y = neg(cos(sub(sub(prod_vec(mul(prod_vec(2.337443560678286), 4.70841758564605)),
            x), prod_vec(prod_vec(prod_vec(2.337443560678286))))))
    return np.mean(y)+0.6


def test_func4(x):
    y = neg(cos(sub(sub(prod_vec(cos(square(1.0156711080086))), x),
            prod_vec(prod_vec(prod_vec(2.337443560678286))))))
    return np.mean(y)+0.6


def evaluate_demo(solution, explogger=None, details=False):
    auc_mean = 0
    auc_std = 0
    detailed_aucs = [0, 0, 0, 0, 0]
    code = solution.code
    algorithm_name = solution.name
    exec(code, globals())

    error = ""

    aucs = []
    # detail_aucs = []
    algorithm = None
    # evaluation_id = 1
    # if evaluation_id < 100:
    problems = []
    ioh.problem.wrap_real_problem(test_func1,
                                  name="func_gp1",
                                  optimization_type=ioh.OptimizationType.MIN,)
    problems += [ioh.get_problem("func_gp1", dimension=23)]
    ioh.problem.wrap_real_problem(test_func2,
                                  name="func_gp2",
                                  optimization_type=ioh.OptimizationType.MIN,)
    problems += [ioh.get_problem("func_gp2", dimension=23)]
    ioh.problem.wrap_real_problem(test_func3,
                                  name="func_gp3",
                                  optimization_type=ioh.OptimizationType.MIN,)
    problems += [ioh.get_problem("func_gp3", dimension=23)]
    ioh.problem.wrap_real_problem(test_func4,
                                  name="func_gp4",
                                  optimization_type=ioh.OptimizationType.MIN,)
    problems += [ioh.get_problem("func_gp4", dimension=23)]
    # else:
    #     rect_width = 2.0
    #     rect_height = 1.0
    #     num_pipes = 3
    #     # exp_name = f"topology_{num_pipes}_{}"
    #     start_points = []
    #     end_points = []
    #     for i in range(num_pipes):
    #         entry_point = (0, np.random.uniform(0.1, 0.9) * rect_height)
    #         exit_point = (rect_width, np.random.uniform(
    #             0.1, 0.9) * rect_height)
    #         start_points += [entry_point]
    #         end_points += [exit_point]
    #     prob = pipes_topology(num_pipes, start_points, end_points)
    #     ioh.problem.wrap_real_problem(prob, name="topology",
    #                                   optimization_type=ioh.OptimizationType.MIN,)
    #     problem = ioh.get_problem("topology", dimension=prob.pca_dim)
    #     problem.bounds.lb = prob.pca_lb
    #     problem.bounds.ub = prob.pca_ub
    dim = problems[0].meta_data.n_variables
    budget = 1000 * dim
    for i in range(len(problems)):
        problem = problems[i]
        # problem = ioh.get_problem(16, 1, dim, problem_class=ioh.ProblemClass.BBOB)
        # lower = -0.7 if i==0 else -0.6
        l2 = aoc_logger(budget, lower=1e-8, upper=1.4,
                        triggers=[ioh.logger.trigger.ALWAYS])
        problem.attach_logger(l2)
        for rep in range(5):
            np.random.seed(rep)
            try:
                algorithm = globals()[algorithm_name](budget=budget, dim=dim)
                algorithm(problem)
            except OverBudgetException:
                pass
            auc = correct_aoc(problem, l2, budget)
            aucs.append(auc)
            l2.reset(problem)
            problem.reset()

    auc_mean = np.mean(aucs)
    auc_std = np.std(aucs)

    i = 0
    while os.path.exists(f"currentexp/aucs-{algorithm_name}-{i}.npy"):
        i += 1
    np.save(f"currentexp/aucs-{algorithm_name}-{i}.npy", aucs)

    feedback = f"The algorithm {algorithm_name} got an average Area over the convergence curve (AOCC, 1.0 is the best) score of {auc_mean:0.2f} with standard deviation {auc_std:0.2f}."
    # if details:
    #     feedback = (
    #         f"{feedback}\nThe mean AOCC score of the algorithm {algorithm_name} on Separable functions was {detailed_aucs[0]:.02f}, "
    #     )
    # feedback = ""

    print(algorithm_name, algorithm, auc_mean, auc_std)
    solution.add_metadata("aucs", aucs)
    solution.set_scores(auc_mean, feedback)

    return solution


if __name__ == "__main__":
    rect_width = 2.0
    rect_height = 1.0
    np.random.seed(42)
    # for num_pipes in range(3, 6):
    #     start_points = []
    #     end_points = []
    #     for i in range(num_pipes):
    #         entry_point = (0, np.random.uniform(0.1, 0.9) * rect_height)
    #         exit_point = (rect_width, np.random.uniform(
    #             0.1, 0.9) * rect_height)
    #         start_points += [entry_point]
    #         end_points += [exit_point]

    #     problem = pipes_topology(num_pipes, start_points, end_points)
    #     dim = problem.dim
    #     X = np.random.uniform(problem.lb, problem.ub,
    #                           size=(1000*dim, problem.lb.shape[0]))

    #     threshold = 0.95
    #     optimal_n, cum_var = find_optimal_pca_components(X)
    #     print(
    #         f"Recommended reserved dimensions: {optimal_n} (keep {threshold*100}% covariance)")
    #     feature_importance_heatmap(X, title=f"cov_heatmap_{num_pipes}")

    # x = np.random.uniform(problem.pca_lb, problem.pca_ub)
    # y = problem(x)
    # print(y)

    # num_pipes = 3
    # # exp_name = f"topology_{num_pipes}"
    # start_points = []
    # end_points = []
    # for i in range(num_pipes):
    #     entry_point = (0, np.random.uniform(0.1, 0.9) * rect_height)
    #     exit_point = (rect_width, np.random.uniform(0.1, 0.9) * rect_height)
    #     start_points += [entry_point]
    #     end_points += [exit_point]
    # prob = pipes_topology(num_pipes, start_points, end_points)
    # ioh.problem.wrap_real_problem(prob, name="topology",
    #                               optimization_type=ioh.OptimizationType.MIN,)
    # problem = ioh.get_problem("topology", dimension=prob.pca_dim)
    # problem.bounds.lb = prob.pca_lb
    # problem.bounds.ub = prob.pca_ub
    # dim = problem.meta_data.n_variables

    # ndoe = 150*dim
    # doe_x = sampling('sobol', n=ndoe, lower_bound=[-1.]*dim,
    #                  upper_bound=[1.]*dim, round_off=2, random_seed=42,
    #                  verbose=True).create_doe()
    # # # y = problem(doe_x)
    # # # np.save(f"{exp_name}.npy", y)
    # y = np.load(f"data/y/{exp_name}_23.npy")

    # # get_ela(dim, problem, doe_x, y, exp_name)
    # # get_elaormalize(dim, exp_name)
    # target_vector = pd.read_csv(f"ela_{exp_name}/ela_1121.csv").mean()
    # list_ela = read_pickle(f"ela_{exp_name}/ela_corr.pickle")
    # ela_min = read_pickle(f"ela_{exp_name}/ela_min.pickle")
    # ela_max = read_pickle(f"ela_{exp_name}/ela_max.pickle")
    # dist_metric = "cityblock"
    # r = 1
    # GP_fgen = GP_func_generator(doe_x,
    #                             target_vector,
    #                             bs_ratio=0.8,
    #                             bs_repeat=5,
    #                             list_ela=list_ela,
    #                             ela_min=ela_min,
    #                             ela_max=ela_max,
    #                             ela_weight={},
    #                             dist_metric=dist_metric,
    #                             problem_label=f'regressor_p_diff_{dim}',
    #                             filepath_save=f'GP_results/regressor_p_diff_{dim}D_{dist_metric}_rep{r}',
    #                             tree_size=(3, 12),
    #                             population=50,
    #                             cxpb=0.5,
    #                             mutpb=0.1,
    #                             ngen=50,
    #                             nhof=1,
    #                             seed=r,
    #                             verbose=True)
    # hof, pop = GP_fgen()

    # ioh.problem.wrap_real_problem(test_func1,
    #                                   name="func_gptest",
    #                                   optimization_type=ioh.OptimizationType.MIN,)
    # problem = ioh.get_problem("func_gp1", dimension=23)
    # problem.bounds.lb = prob.pca_lb
    # problem.bounds.ub = prob.pca_ub
    # x = np.random.uniform(problem.bounds.lb, problem.bounds.ub, (100000,23))
    # y = problem(x)
    # print(np.min(y), np.max(y))
    task_prompt = """
The optimization algorithm should handle a wide range of tasks, which is evaluated on the BBOB test suite of 24 noiseless functions. Your task is to write the optimization algorithm in Python code. The code should contain an `__init__(self, budget, dim)` function and the function `def __call__(self, func)`, which should optimize the black box function `func` using `self.budget` function evaluations.
The func() can only be called as many times as the budget allows, not more. Each of the optimization functions has a search space between -5.0 (lower bound) and 5.0 (upper bound). The dimensionality can be varied.
Give an excellent and novel heuristic algorithm to solve this task and also give it a one-line description with the main idea.
"""
    for experiment_i in range(5):
        # A 1+1 strategy
        es = LLaMEA(
            evaluate_demo,
            llm=llm,
            n_parents=1,
            n_offspring=1,
            task_prompt=task_prompt,
            experiment_name=experiment_name,
            elitism=True,
            HPO=False,
            budget=100,
        )
        print(es.run())

#     alg = DE_PSO_Optimizer(dim*1000, dim)
#     alg(problem)
