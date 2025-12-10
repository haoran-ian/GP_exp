"""
Like image_sample.py, but use a noisy image regressor to guide the sampling
process towards more realistic images.
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F



def transform(image_tensor):
    # Extract the 64x64 image
    x_t = image_tensor[0, 0, :, :]

    # Create the 32x32 matrix
    matrix = x_t[:32, :32]

    # Symmetrize the matrix
    matrix = torch.triu(matrix) + torch.triu(matrix, 1).T

    # Create mirrored matrices
    matrix_x_mirror = torch.flipud(matrix)
    matrix_y_mirror = torch.fliplr(matrix)
    matrix_xy_mirror = torch.flipud(torch.fliplr(matrix))

    # Combine the matrices
    matrix_final = torch.vstack([torch.hstack([matrix, matrix_y_mirror]),
                            torch.hstack([matrix_x_mirror, matrix_xy_mirror])])

    # Resize the matrix
    resized_matrix_final = F.interpolate(matrix_final[None, None, :, :], size=(36, 36), mode='bilinear')

    # binary
    resized_matrix_final = (resized_matrix_final > 0).float()

    return resized_matrix_final




file = np.load('data/PI_denoising_number_497_step_0.npz')
x_in_con_all, x_in_con, mae, traj = file['arr_0'], file['arr_1'], file['arr_3'], file['arr_5']

print(traj.shape)


x_in_trans = torch.tensor(x_in_con_all[1:2]).cpu() # Define x_in_trans variable to be transformed          
x_in_trans = transform(x_in_trans) # 1x1x36x36 tensor
array = x_in_trans.numpy()[0][0]
x_in = np.where(array <= 0, 0, 255)
cv2.imwrite("test00.png", x_in)
triangle_list = []
for i in range(18):
    for j in range(i + 1):
        triangle_list += [array[i][j]]
print(triangle_list)