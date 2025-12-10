# fmt: off
import os
import sys
import cv2
import ioh
import torch
import numpy as np
sys.path.insert(0, os.getcwd())
from problems.meta_surface.wideresnet import WideResNet
# fmt: on


class meta_surface:
    def __init__(self, triangle_edge_length: int = 18,
                 RT_path: str = 'problems/meta_surface/model_best_RT.pth.tar',
                 IT_path: str = 'problems/meta_surface/model_best_IT.pth.tar',
                 device: str = 'cuda:0'):
        self.triangle_edge_length = triangle_edge_length
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
        self.dim = int(self.triangle_edge_length *
                       (self.triangle_edge_length + 1) / 2)
        self.lb = np.array([0. for _ in range(self.dim)])
        self.ub = np.array([1. for _ in range(self.dim)])

    def __call__(self, x):
        x = np.where(x <= 0.5, 0., 1.)
        triangle = self.vector_to_triangle(x)
        quarter_square = self.reflect_triangle(triangle)
        square = self.rotate_around_corner(quarter_square)
        x_in = self.create_final_image(square).reshape((1, 1, 36, 36))
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
        return mae

    def vector_to_triangle(self, x):
        triangle = np.zeros((18, 18))
        idx = 0
        for i in range(18):
            for j in range(i + 1):
                triangle[i, j] = x[idx]
                idx += 1
        return triangle

    def reflect_triangle(self, triangle):
        quarter_square = triangle.copy()
        for i in range(18):
            for j in range(i + 1, 18):
                quarter_square[i, j] = triangle[j, i]
        return quarter_square

    def rotate_around_corner(self, quarter_square):
        size = quarter_square.shape[0] * 2
        square = np.zeros((size, size))
        rotated_90 = np.rot90(quarter_square)
        rotated_180 = np.rot90(quarter_square, 2)
        rotated_270 = np.rot90(quarter_square, 3)
        square[:quarter_square.shape[0],
               :quarter_square.shape[1]] = quarter_square
        square[:quarter_square.shape[0],
               quarter_square.shape[1]:] = rotated_270
        square[quarter_square.shape[0]:,
               quarter_square.shape[1]:] = rotated_180
        square[quarter_square.shape[0]:,
               :quarter_square.shape[1]] = rotated_90
        return square

    def create_final_image(self, square):
        square_36 = cv2.resize(square, (36, 36))
        square_36 = np.where(square_36 <= 0.5, 0., 1.)
        return square_36


def get_meta_surface_problem(
        RT_path='problems/meta_surface/model_best_IT.pth.tar',
        IT_path='problems/meta_surface/model_best_IT.pth.tar',
        device='cuda:0'):
    prob = meta_surface(RT_path=RT_path, IT_path=IT_path, device=device)
    ioh.problem.wrap_real_problem(prob, name='meta_surface',
                                  optimization_type=ioh.OptimizationType.MIN)
    problem = ioh.get_problem('meta_surface', dimension=prob.dim)
    problem.bounds.lb = prob.lb
    problem.bounds.ub = prob.ub
    return problem
