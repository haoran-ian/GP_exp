"""
Like image_sample.py, but use a noisy image regressor to guide the sampling
process towards more realistic images.
"""

# fmt: off
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from problems.meta_surface.wideresnet import WideResNet
from problems.meta_surface.problem import meta_surface
# fmt: on


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
    resized_matrix_final = F.interpolate(
        matrix_final[None, None, :, :], size=(36, 36), mode='bilinear')

    # binary
    resized_matrix_final = (resized_matrix_final > 0).float()

    return resized_matrix_final


file = np.load(
    'data/PI_denoising_number_497_step_0.npz')
x_in_con_all, x_in_con, mae, traj = file['arr_0'], file['arr_1'], file['arr_3'], file['arr_5']

print(mae)

device = 'cuda:1'
# x_in_con = torch.tensor(x_in_con).to(device)
# x_in_con = transform(x_in_con)
# x_in_con = Index_Transformation_01(x_in_con)

resume_re = "problems/meta_surface/model_best_RT.pth.tar"
resume_im = "problems/meta_surface/model_best_IT.pth.tar"
# Regressor real
regressor_real = WideResNet(
    depth=16, num_classes=100, widen_factor=4, dropRate=0.3)

# for training on multiple GPUs.
# Use CUDA_VISIBLE_DEVICES=0,1 to specify which GPUs to use
# model = torch.nn.DataParallel(model).cuda()
regressor_real = regressor_real.cuda()

# optionally resume from a checkpoint
checkpoint = torch.load(resume_re)
# args.start_epoch = checkpoint['epoch']
# best_loss_valid = checkpoint['best_loss_valid']
regressor_real.load_state_dict(checkpoint['state_dict'])
# print("=> loaded checkpoint '{}' (epoch {})"
#         .format(args.resume_re, checkpoint['epoch']))
regressor_real.eval()

# Regressor imaginary
regressor_imaginary = WideResNet(
    depth=16, num_classes=100, widen_factor=4, dropRate=0.3)

# get the number of model parameters
print('Number of model parameters: {}'.format(
    sum([p.data.nelement() for p in regressor_imaginary.parameters()])))

# for training on multiple GPUs.
# Use CUDA_VISIBLE_DEVICES=0,1 to specify which GPUs to use
# model = torch.nn.DataParallel(model).cuda()
regressor_imaginary = regressor_imaginary.cuda()

# optionally resume from a checkpoint
checkpoint = torch.load(resume_im)
# args.start_epoch = checkpoint['epoch']
# best_loss_valid = checkpoint['best_loss_valid']
regressor_imaginary.load_state_dict(checkpoint['state_dict'])
# print("=> loaded checkpoint '{}' (epoch {})"
#         .format(args.resume_im, checkpoint['epoch']))
regressor_imaginary.eval()


# Define x_in_trans variable to be transformed
# x_in_trans = torch.tensor(x_in_con_all[1:2]).cuda()
# x_in_trans = transform(x_in_trans)  # 1x1x36x36 tensor

x = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0]
x = np.array(x)
problem = meta_surface()
triangle = problem.vector_to_triangle(x)
square_9 = problem.reflect_triangle(triangle)
square_18 = problem.rotate_around_corner(square_9)
x_in_trans = square_18.reshape((1, 1, 36, 36))
x_in_trans = torch.tensor(x_in_trans).float().to('cuda:0')
# np.set_printoptions(threshold=np.inf)
# print(x_in_trans.numpy()[0][0])

# Predicted vectors from Re and Im
logits_re = regressor_real(x_in_trans)  # 1x100
logits_im = regressor_imaginary(x_in_trans)  # 1x100

# Obtain magnitudes 1x100
predicted = torch.sqrt(logits_re**2 + logits_im**2)  # Magnitude
# predicted = logits_re.clone()
predicted = predicted.reshape(1, 100)


x_axes = np.linspace(0, 1, 100)  # Define x axis
target = 1 - 2 * (x_axes - 0.5) ** 2  # Obtain y values
# target = 0.5*np.ones((1, 100))  # Obtain y values
target = target.reshape(1, -1)
# print(target.shape)
plt.figure()
plt.plot(predicted.detach().cpu().numpy().reshape(100,), label='Trajectory')
plt.plot(target.reshape(100,), label='Target')
plt.show()

print(np.mean(np.abs(predicted.detach().cpu().numpy() - target)))  # MAE
