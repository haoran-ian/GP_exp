# fmt: off
import os
import sys
import ioh
import torch
import joblib
import numpy as np
sys.path.insert(0, os.getcwd())
from sklearn.decomposition import PCA
from problems.meta_surface.wideresnet import WideResNet
# fmt: on


class meta_surface:
    def __init__(self, RT_path="problems/meta_surface/model_best_IT.pth.tar",
                 IT_path="problems/meta_surface/model_best_IT.pth.tar",
                 device="cuda:0"):
        self.RT_path = RT_path
        self.IT_path = IT_path
        self.device = device
        self.regressor_real = WideResNet(depth=16, num_classes=100,
                                         widen_factor=4, dropRate=0.3)
        self.regressor_real = self.regressor_real.cuda()
        checkpoint = torch.load(self.RT_path, weights_only=True)
        self.regressor_real.load_state_dict(checkpoint['state_dict'])
        self.regressor_real.eval()
        self.regressor_imaginary = WideResNet(depth=16, num_classes=100,
                                              widen_factor=4, dropRate=0.3)
        self.regressor_imaginary = self.regressor_imaginary.cuda()
        checkpoint = torch.load(self.IT_path, weights_only=True)
        self.regressor_imaginary.load_state_dict(checkpoint['state_dict'])
        self.regressor_imaginary.eval()
        self.dim = 45
        self.lb = np.array([-1. for _ in range(self.dim)])
        self.ub = np.array([1. for _ in range(self.dim)])
        # X_pca_train = np.random.uniform(self.lb, self.ub,
        #                                 size=(1000*self.dim, self.lb.shape[0]))
        # self.pca_dim = 20
        # self.pca = PCA(n_components=self.pca_dim)
        # self.X_pca = self.pca.fit_transform(X_pca_train)
        # pca_min = np.min(self.X_pca, axis=0)
        # pca_max = np.max(self.X_pca, axis=0)
        # self.pca_lb = pca_min - 0.1
        # self.pca_ub = pca_max + 0.1

    def __call__(self, x):
        # x = self.pca.inverse_transform(x)
        x = np.where(x < 0, -1., 1.)
        triangle = self.vector_to_triangle(x)
        square_9 = self.reflect_triangle(triangle)
        square_18 = self.rotate_around_corner(square_9)
        x_in = self.create_final_image(square_18).reshape((1, 1, 36, 36))
        x_in = torch.tensor(x_in).float().to(self.device)
        logits_re = self.regressor_real(x_in)  # 1x100
        logits_im = self.regressor_imaginary(x_in)  # 1x100
        predicted = torch.sqrt(logits_re**2 + logits_im**2)  # Magnitude
        predicted = predicted.reshape(1, 100)
        x_axes = torch.linspace(0, 1, 100)  # Define x axis
        target = 1 - 2 * (x_axes - 0.5) ** 2  # Obtain y values
        target = target.reshape(1, -1)
        target = target.cuda()
        mae = torch.mean(torch.abs(predicted - target))  # MAE
        return mae - 0.237

    def vector_to_triangle(self, x):
        triangle = np.zeros((9, 9))
        idx = 0
        for i in range(9):
            for j in range(i + 1):
                triangle[i, j] = x[idx]
                idx += 1
        return triangle

    def reflect_triangle(self, triangle):
        square_9 = triangle.copy()
        for i in range(9):
            for j in range(i + 1, 9):
                square_9[i, j] = triangle[j, i]
        return square_9

    def rotate_around_corner(self, square_9):
        size = square_9.shape[0] * 2
        square_18 = np.zeros((size, size))
        rotated_90 = np.rot90(square_9)
        rotated_180 = np.rot90(square_9, 2)
        rotated_270 = np.rot90(square_9, 3)
        square_18[:square_9.shape[0], :square_9.shape[1]] = square_9
        square_18[:square_9.shape[0], square_9.shape[1]:] = rotated_270
        square_18[square_9.shape[0]:, square_9.shape[1]:] = rotated_180
        square_18[square_9.shape[0]:, :square_9.shape[1]] = rotated_90
        return square_18

    def create_final_image(self, square_18):
        square_36 = np.zeros((36, 36))
        for i in range(2):
            for j in range(2):
                start_i = i * 18
                end_i = (i + 1) * 18
                start_j = j * 18
                end_j = (j + 1) * 18
                square_36[start_i:end_i, start_j:end_j] = square_18
        return square_36


def get_meta_surface_problem(
        RT_path="problems/meta_surface/model_best_IT.pth.tar",
        IT_path="problems/meta_surface/model_best_IT.pth.tar",
        device="cuda:0"):
    prob = meta_surface(RT_path=RT_path, IT_path=IT_path, device=device)
    ioh.problem.wrap_real_problem(prob, name="meta_surface",
                                  optimization_type=ioh.OptimizationType.MIN)
    problem = ioh.get_problem("meta_surface", dimension=prob.dim)
    problem.bounds.lb = prob.lb
    problem.bounds.ub = prob.ub
    return problem
