"""
Like image_sample.py, but use a noisy image regressor to guide the sampling
process towards more realistic images.
"""

import argparse
import os

#import pytorch_fid
#import cv2
import sys
import numpy as np
import torch as th
import torch.distributed as dist
import torch.nn.functional as F
import matplotlib.pyplot as plt
from wideresnet import WideResNet

from topodiff.cons_input_datasets import load_data
from topodiff import dist_util, logger
from topodiff.script_util import (
    model_and_diffusion_defaults,
    regressor_defaults,
    classifier_defaults,
    create_model_and_diffusion,
    create_regressor,
    create_classifier,
    add_dict_to_argparser,
    args_to_dict,
)


# os.environ['TOPODIFF_LOGDIR'] = './generated'
# th.cuda.set_device(0) # Please change GPU here
device = 'cuda:0'

def transform(image_tensor):

    # Create the 18x18 matrix
    matrix = image_tensor

    # Symmetrize the matrix
    matrix = th.triu(matrix) + th.triu(matrix, 1).T

    # Create mirrored matrices
    matrix_x_mirror = th.flipud(matrix)
    matrix_y_mirror = th.fliplr(matrix)
    matrix_xy_mirror = th.flipud(th.fliplr(matrix))

    # Combine the matrices
    matrix_final = th.vstack([th.hstack([matrix, matrix_y_mirror]),
                                th.hstack([matrix_x_mirror, matrix_xy_mirror])])
    
    matrix_final = matrix_final.reshape(1, 1, 36, 36)


    return matrix_final





def main():
    
    ### LOAD REGRESSORS ###

    # Regressor real
    logger.log("loading regressor_real...")
    regressor_real = WideResNet(depth=16, num_classes=100, widen_factor=4, dropRate=0.3)

    # for training on multiple GPUs.
    # Use CUDA_VISIBLE_DEVICES=0,1 to specify which GPUs to use
    # model = torch.nn.DataParallel(model).cuda()
    regressor_real = regressor_real.cuda()
    resume_re_path = "/home/weiz/Astar_Work/NAS/plain-diffusion-meta1/optimization/model_best_RT.pth.tar"
    # optionally resume from a checkpoint
    checkpoint = th.load(resume_re_path)
    regressor_real.load_state_dict(checkpoint['state_dict'])
    regressor_real.eval()
    
    # Regressor imaginary
    logger.log("loading regressor_imaginary...")
    regressor_imaginary = WideResNet(depth=16, num_classes=100, widen_factor=4, dropRate=0.3)

    # get the number of model parameters
    print('Number of model parameters: {}'.format(
        sum([p.data.nelement() for p in regressor_imaginary.parameters()])))

    # for training on multiple GPUs.
    # Use CUDA_VISIBLE_DEVICES=0,1 to specify which GPUs to use
    # model = torch.nn.DataParallel(model).cuda()
    regressor_imaginary = regressor_imaginary.cuda()
    resume_im_path = "/home/weiz/Astar_Work/NAS/plain-diffusion-meta1/optimization/model_best_IT.pth.tar"

    # optionally resume from a checkpoint
    checkpoint = th.load(resume_im_path)
    regressor_imaginary.load_state_dict(checkpoint['state_dict'])
    regressor_imaginary.eval()
    
    ### END OF LOAD REGRESSORS ###
 

    x_in = np.random.uniform(-1, 1, (18, 18))
    x_in = th.tensor(x_in).float().to(device)
    x_in_trans = transform(x_in) # 1x1x36x36 tensor

    # Predicted vectors from Re and Im
    logits_re = regressor_real(x_in_trans) # 1x100
    logits_im = regressor_imaginary(x_in_trans) # 1x100

    # Obtain magnitudes 1x100
    predicted = th.sqrt(logits_re**2 + logits_im**2) # Magnitude
    # predicted = logits_re.clone()
    predicted = predicted.reshape(1, 100)
        
    # Obtain target vector 1x100
    x_axes = th.linspace(0, 1, 100) # Define x axis
    target = 1 - 2 * (x_axes - 0.5) ** 2 # Obtain y values
    target = target.reshape(1, -1)
    target = target.cuda()

    mae = th.mean(th.abs(predicted - target))  # MAE
                
    return mae
        
    ### END OF GUIDANCE GRADIENT FUNCTION ###
        



if __name__ == "__main__":
    
    main()
