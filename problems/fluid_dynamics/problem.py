# fmt: off
import os
import sys
import cv2
import ioh
import time
import torch
import numpy as np
sys.path.insert(0, os.getcwd())
from sklearn.decomposition import PCA
from scipy.spatial.distance import cdist
from problems.fluid_dynamics.topodiff import dist_util
from problems.fluid_dynamics.topodiff.script_util import create_regressor
# fmt: on


class Pipe:
    def __init__(self, entry_point, exit_point, path, thickness, control_pts):
        self.entry_point = entry_point
        self.exit_point = exit_point
        self.path = path
        self.thickness = thickness
        self.control_pts = control_pts


class pipes_topology:
    def __init__(self, num_pipes, start_points, end_points, img_res=(256, 512),
                 rect_width=2.0, rect_height=1.0, num_of_bezier_points=1500,
                 device="cuda:0"):
        self.num_pipes = num_pipes
        self.rect_width = rect_width
        self.rect_height = rect_height
        self.num_of_bezier_points = num_of_bezier_points
        self.device = device
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
        x_tensor = torch.tensor(design).float().to(self.device)
        y = self.regressor_p_diff(x_tensor.reshape(1, 1, 64, 64),
                                  torch.zeros(1, ).to(self.device))
        y = y.tolist()[0][0]
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


def get_pipes_topology_problem(iid, num_pipes, img_res=(256, 512), rect_width=2.0,
                               rect_height=1.0, num_of_bezier_points=1500,
                               device="cuda:0"):
    np.random.seed(42+num_pipes+iid)
    start_points = []
    end_points = []
    for _ in range(num_pipes):
        entry_point = (0, np.random.uniform(0.1, 0.9) * rect_height)
        exit_point = (rect_width, np.random.uniform(
            0.1, 0.9) * rect_height)
        start_points += [entry_point]
        end_points += [exit_point]
    prob = pipes_topology(num_pipes, start_points, end_points, img_res=img_res,
                          rect_width=rect_width, rect_height=rect_height,
                          num_of_bezier_points=num_of_bezier_points,
                          device=device)
    ioh.problem.wrap_real_problem(prob, name=f"fluid_dynamics_{num_pipes}pipes_iid{iid}",
                                  optimization_type=ioh.OptimizationType.MIN)
    problem = ioh.get_problem(f"fluid_dynamics_{num_pipes}pipes_iid{iid}",
                              dimension=prob.pca_dim)
    problem.bounds.lb = prob.pca_lb
    problem.bounds.ub = prob.pca_ub
    return problem
