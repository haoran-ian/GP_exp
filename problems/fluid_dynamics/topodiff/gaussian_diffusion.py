import enum
import math
import random

import numpy as np
import torch as th

from .nn import mean_flat
from .losses import normal_kl, discretized_gaussian_log_likelihood

# from models.diffusion_related import Index_Transformation_01, Identify_Bubbles

import matplotlib.pyplot as plt


# seed_all = [20, 22, 30, 31]
# seed = seed_all[0]
# i_seed = int(0)

seed = 51
last_denoising_step = 49

def get_named_beta_schedule(schedule_name, num_diffusion_timesteps):
    """
    Get a pre-defined beta schedule for the given name.

    The beta schedule library consists of beta schedules which remain similar
    in the limit of num_diffusion_timesteps.
    Beta schedules may be added, but should not be removed or changed once
    they are committed to maintain backwards compatibility.
    """
    if schedule_name == "linear":
        # Linear schedule from Ho et al, extended to work for any number of
        # diffusion steps.
        scale = 1000 / num_diffusion_timesteps
        beta_start = scale * 0.0001
        beta_end = scale * 0.02
        return np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif schedule_name == "cosine":
        return betas_for_alpha_bar(
            num_diffusion_timesteps,
            lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2,
        )
    else:
        raise NotImplementedError(f"unknown beta schedule: {schedule_name}")


def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    """
    Create a beta schedule that discretizes the given alpha_t_bar function,
    which defines the cumulative product of (1-beta) over time from t = [0,1].

    :param num_diffusion_timesteps: the number of betas to produce.
    :param alpha_bar: a lambda that takes an argument t from 0 to 1 and
                      produces the cumulative product of (1-beta) up to that
                      part of the diffusion process.
    :param max_beta: the maximum beta to use; use values lower than 1 to
                     prevent singularities.
    """
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)


class ModelMeanType(enum.Enum):
    """
    Which type of output the model predicts.
    """

    PREVIOUS_X = enum.auto()  # the model predicts x_{t-1}
    START_X = enum.auto()  # the model predicts x_0
    EPSILON = enum.auto()  # the model predicts epsilon


class ModelVarType(enum.Enum):
    """
    What is used as the model's output variance.

    The LEARNED_RANGE option has been added to allow the model to predict
    values between FIXED_SMALL and FIXED_LARGE, making its job easier.
    """

    LEARNED = enum.auto()
    FIXED_SMALL = enum.auto()
    FIXED_LARGE = enum.auto()
    LEARNED_RANGE = enum.auto()


class LossType(enum.Enum):
    MSE = enum.auto()  # use raw MSE loss (and KL when learning variances)
    RESCALED_MSE = (
        enum.auto()
    )  # use raw MSE loss (with RESCALED_KL when learning variances)
    KL = enum.auto()  # use the variational lower-bound
    RESCALED_KL = enum.auto()  # like KL, but rescale to estimate the full VLB

    def is_vb(self):
        return self == LossType.KL or self == LossType.RESCALED_KL


class GaussianDiffusion:
    """
    Utilities for training and sampling diffusion models.

    Ported directly from here, and then adapted over time to further experimentation.
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/diffusion_utils_2.py#L42

    :param betas: a 1-D numpy array of betas for each diffusion timestep,
                  starting at T and going to 1.
    :param model_mean_type: a ModelMeanType determining what the model outputs.
    :param model_var_type: a ModelVarType determining how variance is output.
    :param loss_type: a LossType determining the loss function to use.
    :param rescale_timesteps: if True, pass floating point timesteps into the
                              model so that they are always scaled like in the
                              original paper (0 to 1000).
    """

    def __init__(
        self,
        *,
        betas,
        model_mean_type,
        model_var_type,
        loss_type,
        rescale_timesteps=False,
    ):
        self.model_mean_type = model_mean_type
        self.model_var_type = model_var_type
        self.loss_type = loss_type
        self.rescale_timesteps = rescale_timesteps

        # Use float64 for accuracy.
        betas = np.array(betas, dtype=np.float64)
        self.betas = betas
        assert len(betas.shape) == 1, "betas must be 1-D"
        assert (betas > 0).all() and (betas <= 1).all()

        self.num_timesteps = int(betas.shape[0])

        alphas = 1.0 - betas
        self.alphas_cumprod = np.cumprod(alphas, axis=0)
        self.alphas_cumprod_prev = np.append(1.0, self.alphas_cumprod[:-1])
        self.alphas_cumprod_next = np.append(self.alphas_cumprod[1:], 0.0)
        assert self.alphas_cumprod_prev.shape == (self.num_timesteps,)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = np.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod - 1)

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = (
            betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        # log calculation clipped because the posterior variance is 0 at the
        # beginning of the diffusion chain.
        self.posterior_log_variance_clipped = np.log(
            np.append(self.posterior_variance[1], self.posterior_variance[1:])
        )
        self.posterior_mean_coef1 = (
            betas * np.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev)
            * np.sqrt(alphas)
            / (1.0 - self.alphas_cumprod)
        )

    def q_mean_variance(self, x_start, t):
        """
        Get the distribution q(x_t | x_0).

        :param x_start: the [N x C x ...] tensor of noiseless inputs.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :return: A tuple (mean, variance, log_variance), all of x_start's shape.
        """
        mean = (
            _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        )
        variance = _extract_into_tensor(1.0 - self.alphas_cumprod, t, x_start.shape)
        log_variance = _extract_into_tensor(
            self.log_one_minus_alphas_cumprod, t, x_start.shape
        )
        return mean, variance, log_variance

    def q_sample(self, x_start, t, noise=None):
        """
        Diffuse the data for a given number of diffusion steps.

        In other words, sample from q(x_t | x_0).

        :param x_start: the initial data batch.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :param noise: if specified, the split-out normal noise.
        :return: A noisy version of x_start.
        """
        if noise is None:
            noise = th.randn_like(x_start)
        assert noise.shape == x_start.shape
        return (
            _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + _extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
            * noise
        )

    def q_posterior_mean_variance(self, x_start, x_t, t):
        """
        Compute the mean and variance of the diffusion posterior:

            q(x_{t-1} | x_t, x_0)

        """
        assert x_start.shape == x_t.shape
        posterior_mean = (
            _extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + _extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )

        # # identify the bubbles
        # global last_denoising_step
        # if t < last_denoising_step:
        #     x_in_TF0 = th.sigmoid(posterior_mean)
        #     x_in_TF0 = Index_Transformation_01(x_in_TF0)
        #     x_in_TF0 = Identify_Bubbles(x_in_TF0)
        #     posterior_mean[x_in_TF0 == 0] = x_t[x_in_TF0 == 0]
        
        # # keep first 5 cells same
        # if t < last_denoising_step:
        #     posterior_mean[:, :, :, 0:5] = x_t[:, :, :, 0:5]
            


        posterior_variance = _extract_into_tensor(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = _extract_into_tensor(
            self.posterior_log_variance_clipped, t, x_t.shape
        )
        assert (
            posterior_mean.shape[0]
            == posterior_variance.shape[0]
            == posterior_log_variance_clipped.shape[0]
            == x_start.shape[0]
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(
        self, model, x, cons, t, clip_denoised=True, denoised_fn=None, model_kwargs=None
    ):
        """
        Apply the model to get p(x_{t-1} | x_t), as well as a prediction of
        the initial x, x_0.

        :param model: the model, which takes a signal and a batch of timesteps
                      as input.
        :param x: the image of the topology at time t.
        :param cons: the constraints channels used as input by the main diffusion model (volume, physical fields, loads).
        :param t: a 1-D Tensor of timesteps.
        :param clip_denoised: if True, clip the denoised signal into [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample. Applies before
            clip_denoised.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model.
        :return: a dict with the following keys:
                 - 'mean': the model mean output.
                 - 'variance': the model variance output.
                 - 'log_variance': the log of 'variance'.
                 - 'pred_xstart': the prediction for x_0.
        """
        if model_kwargs is None:
            model_kwargs = {}
        B, C = x.shape[:2]
        assert t.shape == (B,)
        full_arr = th.cat((x, cons), dim = 1)
        model_output = model(full_arr, self._scale_timesteps(t), **model_kwargs)

        if self.model_var_type in [ModelVarType.LEARNED, ModelVarType.LEARNED_RANGE]:
            assert model_output.shape == (B, C * 2, *x.shape[2:])
            model_output, model_var_values = th.split(model_output, C, dim=1)
            if self.model_var_type == ModelVarType.LEARNED:
                model_log_variance = model_var_values
                model_variance = th.exp(model_log_variance)
            else:
                min_log = _extract_into_tensor(
                    self.posterior_log_variance_clipped, t, x.shape
                )
                max_log = _extract_into_tensor(np.log(self.betas), t, x.shape)
                # The model_var_values is [-1, 1] for [min_var, max_var].
                frac = (model_var_values + 1) / 2
                model_log_variance = frac * max_log + (1 - frac) * min_log
                model_variance = th.exp(model_log_variance)
        else:
            model_variance, model_log_variance = {
                # for fixedlarge, we set the initial (log-)variance like so
                # to get a better decoder log likelihood.
                ModelVarType.FIXED_LARGE: (
                    np.append(self.posterior_variance[1], self.betas[1:]),
                    np.log(np.append(self.posterior_variance[1], self.betas[1:])),
                ),
                ModelVarType.FIXED_SMALL: (
                    self.posterior_variance,
                    self.posterior_log_variance_clipped,
                ),
            }[self.model_var_type]
            model_variance = _extract_into_tensor(model_variance, t, x.shape)
            model_log_variance = _extract_into_tensor(model_log_variance, t, x.shape)

        def process_xstart(x):
            if denoised_fn is not None:
                x = denoised_fn(x)
            if clip_denoised:
                return x.clamp(-1, 1)
            return x

        if self.model_mean_type == ModelMeanType.PREVIOUS_X:
            pred_xstart = process_xstart(
                self._predict_xstart_from_xprev(x_t=x, t=t, xprev=model_output)
            )
            model_mean = model_output
        elif self.model_mean_type in [ModelMeanType.START_X, ModelMeanType.EPSILON]:
            if self.model_mean_type == ModelMeanType.START_X:
                pred_xstart = process_xstart(model_output)
            else:
                pred_xstart = process_xstart(
                    self._predict_xstart_from_eps(x_t=x, t=t, eps=model_output)
                )
            model_mean, _, _ = self.q_posterior_mean_variance(
                x_start=pred_xstart, x_t=x, t=t
            )
        else:
            raise NotImplementedError(self.model_mean_type)

        assert (
            model_mean.shape == model_log_variance.shape == pred_xstart.shape == x.shape
        )
        return {
            "mean": model_mean,
            "variance": model_variance,
            "log_variance": model_log_variance,
            "pred_xstart": pred_xstart,
        }

    def _predict_xstart_from_eps(self, x_t, t, eps):
        assert x_t.shape == eps.shape
        return (
            _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * eps
        )

    def _predict_xstart_from_xprev(self, x_t, t, xprev):
        assert x_t.shape == xprev.shape
        return (  # (xprev - coef2*x_t) / coef1
            _extract_into_tensor(1.0 / self.posterior_mean_coef1, t, x_t.shape) * xprev
            - _extract_into_tensor(
                self.posterior_mean_coef2 / self.posterior_mean_coef1, t, x_t.shape
            )
            * x_t
        )

    def _predict_eps_from_xstart(self, x_t, t, pred_xstart):
        return (
            _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - pred_xstart
        ) / _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)

    def _scale_timesteps(self, t):
        if self.rescale_timesteps:
            return t.float() * (1000.0 / self.num_timesteps)
        return t


    def condition_mean(self, cond_fn_1, cond_fn_2, p_mean_var, x, cons, t, model_kwargs=None):
        """
        Shift the mean computed by the diffusion model to take into account the deviation induced by the guidance strategy.
        cond_fn_1 computes the gradient of the regressor predicting compliance.

        This uses the conditioning strategy from Sohl-Dickstein et al. (2015).
        
        :param cond_fn_1: the function that computes the gradient of the regressor.
        :param p_mean_var: the mean and variance predicted by the plain diffusion model.
        :param x: the image at the previous timestep.
        :param cons: the channels corresponding to the volume and physical fields.
        :param t: the value of t, starting at 0 for the first diffusion step.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model.
        :return: the new shifted mean.
        """

        global nan_condition

        t_rs = (1 - t.float()/100)*1000

        # q_dot
        #gradient_1 = cond_fn_1(x, self._scale_timesteps(t), **model_kwargs)
        #gradient_1, logits_1 = cond_fn_1(x, self._scale_timesteps(t), **model_kwargs)  
        #_, logits_2 = cond_fn_1(p_mean_var["mean"].float(), self._scale_timesteps(t), **model_kwargs)  
        gradient_1, logits_1 = cond_fn_1(p_mean_var["mean"].float(), self._scale_timesteps(t), **model_kwargs)  
        
        new_mean = (
            # p_mean_var["mean"].float() + p_mean_var["variance"] * gradient_1.float() * (1.0 if t_rs < 500 else 100.)
            p_mean_var["mean"].float()
        )
        
        #_, logits_3 = cond_fn_1(new_mean, self._scale_timesteps(t), **model_kwargs)  
        
        # if int(t_rs) < 100:
        #     print ("time: %d ; x: %f; p_mean_before: %f; p_mean_after: %f" %(int(t_rs), logits_1, logits_2, logits_3))

        # if t == 9:
        #     cc = 1

        # p_diff
        # gradient_2, logits_2 = cond_fn_2(p_mean_var["mean"].float(), self._scale_timesteps(t), **model_kwargs)
        # gradient_2, logits_2 = cond_fn_2(p_mean_var["mean"].float(), self._scale_timesteps(t), p_mean_var["variance"], **model_kwargs)
        global x_in_TF, last_denoising_step
        gradient_2, logits_2, nan_condition, x_in_TF = cond_fn_2(p_mean_var["mean"].float(), self._scale_timesteps(t), p_mean_var["variance"], p_mean_var["pred_xstart"].float(), **model_kwargs)  # PI

        # basic_f = 1e-3
        # if 29 <= t <= 49:
        #     alpha = basic_f
        # elif 9 <= t < 29:
        #     alpha = basic_f*2
        # elif 5 <= t < 9:
        #     alpha = basic_f*5
        # elif t < 5:
        #     alpha = basic_f*9
        # else:
        #     alpha = 15

        new_mean1 = (
            # new_mean + p_mean_var["variance"] * gradient_2.float() * 10 # * (0.0 if t_rs < 500 else 0.)
            # new_mean + p_mean_var["variance"] * gradient_2.float() * (0.0 if t_rs > 100 else 10.)
            # new_mean + p_mean_var["variance"] * gradient_2.float() * (0.0 if t_rs < 1000 else 10.)
            # new_mean + p_mean_var["variance"] * gradient_2.float() * (0.0 if t > 9 else 3e7)
            # new_mean + gradient_2.float() * (0.0 if t > 9 else 9e4)  # CNN
            # new_mean + gradient_2.float() * (0.0 if t > 19 else 1.1e3)  # PI
            # new_mean + gradient_2.float() * (0.0 if t > 19 else 15) * x_in_TF  # PI
            # new_mean + gradient_2.float() * (0.0 if t > 19 else 15)  # PI
            # new_mean + gradient_2.float() * (0.0 if t > 9 else 50)  # PI
            # new_mean + gradient_2.float() * (0.0 if t > 9 else 50) * x_in_TF  # PI
            new_mean + gradient_2.float() * (0.0 if t > last_denoising_step else 1) * x_in_TF  # PI
            # new_mean + gradient_2.float() * (0.0 if t > 9 else 1e4)  # PI
            # new_mean + gradient_2.float() * (0.0 if t > 9 else 7e2)  # PI
            # new_mean + gradient_2.float() * alpha  # PI
            # new_mean + gradient_2.float() * (0.0 if t > 19 else 1e4)
            # new_mean
        )

        # # identify the bubbles
        # if t <= last_denoising_step:
        #     x_in_TF0 = th.sigmoid(new_mean1)
        #     x_in_TF0 = Index_Transformation_01(x_in_TF0)
        #     x_in_TF0 = Identify_Bubbles(x_in_TF0)
        #     new_mean1[x_in_TF0 == 0] = new_mean[x_in_TF0 == 0]

        # # save
        # name = 'diffusion_denoising_step1_{n}'.format(n = int(t))
        # # name = 'PI_denoising_step_{n}'.format(n = int(t))
        # np.savez(name, p_mean_var["mean"].detach().cpu().numpy())

        
        #print(p_mean_var["mean"].float().size())
        
        return new_mean1

    def p_sample(
        self,
        model,
        x,
        cons,
        t,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn_1=None,
        cond_fn_2=None,
        model_kwargs=None,
    ):
        """
        Sample x_{t-1} from the model at the given timestep.

        :param model: the model to sample from.
        :param x: the current image at x_{t-1}.
        :param cons: the channels corresponding to the volume and physical fields.
        :param t: the value of t, starting at 0 for the first diffusion step.
        :param clip_denoised: if True, clip the x_start prediction to [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample.
        :param cond_fn_1: gradient function for the compliance regressor.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model.
        :return: a dict containing the following keys:
                 - 'sample': a random sample from the model.
                 - 'pred_xstart': a prediction of x_0.
        """
        out = self.p_mean_variance(
            model,
            x,
            th.cat([cons], dim = 1),
            t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
        )
        noise = th.randn_like(x)
        nonzero_mask = (
            (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        )  # no noise when t == 0

        if cond_fn_1 is not None:
            out["mean"] = self.condition_mean(
                cond_fn_1, cond_fn_2, out, x, cons, t, model_kwargs=model_kwargs
            )
            temp = out["mean"]

        # # keep first 5 cells same
        # if t <= last_denoising_step:
        #     device = noise.device
        #     mask_5cells = th.ones_like(out["mean"]).to(device)
        #     mask_5cells[:, :, :, 0:5] = 0
        #     sample = out["mean"] + nonzero_mask * th.exp(0.5 * out["log_variance"]) * noise * mask_5cells
        # else:
        #     sample = out["mean"] + nonzero_mask * th.exp(0.5 * out["log_variance"]) * noise

        # # identify the bubbles
        # global x_in_TF, last_denoising_step
        # if t <= last_denoising_step:
        #     device = noise.device
        #     mask_5cells = th.ones_like(out["mean"]).to(device)
        #     mask_5cells[:, :, :, 0:5] = 0
        #     sample = out["mean"] + nonzero_mask * th.exp(0.5 * out["log_variance"]) * noise * mask_5cells

        #     # sample = out["mean"] + nonzero_mask * th.exp(0.5 * out["log_variance"]) * noise
        #     # identify the bubbles
        #     x_in_TF0 = th.sigmoid(sample)
        #     x_in_TF0 = Index_Transformation_01(x_in_TF0)
        #     x_in_TF0 = Identify_Bubbles(x_in_TF0)
        #     sample[x_in_TF0 == 0] = out["mean"][x_in_TF0 == 0]
        # else:
        #     sample = out["mean"] + nonzero_mask * th.exp(0.5 * out["log_variance"]) * noise
        
        sample = out["mean"] + nonzero_mask * th.exp(0.5 * out["log_variance"]) * noise

        return {"sample": sample, "pred_xstart": out["pred_xstart"]}

    def p_sample_loop(
        self,
        model,
        shape,
        cons,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn_1=None,
        cond_fn_2=None,
        model_kwargs=None,
        device=None,
        progress=False,
    ):
        """
        Generate samples from the model.

        :param model: the model module.
        :param shape: the shape of the samples, (N, C, H, W).
        :param cons: the channels corresponding to the volume and physical fields.
        :param noise: if specified, the noise from the encoder to sample.
                      Should be of the same shape as `shape`.
        :param clip_denoised: if True, clip x_start predictions to [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample.
        :param cond_fn_1: gradient function for the compliance regressor.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model.
        :param device: if specified, the device to create the samples on.
                       If not specified, use a model parameter's device.
        :param progress: if True, show a tqdm progress bar.
        :return: a non-differentiable batch of samples.
        """
        final = None
        for sample in self.p_sample_loop_progressive(
            model,
            shape,
            cons,
            noise=noise,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            cond_fn_1=cond_fn_1,
            cond_fn_2=cond_fn_2,
            model_kwargs=model_kwargs,
            device=device,
            progress=progress,
        ):
            final = sample
        return final["sample"]

    def p_sample_loop_progressive(
        self,
        model,
        shape,
        cons,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn_1=None,
        cond_fn_2=None,
        model_kwargs=None,
        device=None,
        progress=False,
    ):
        """
        Generate samples from the model and yield intermediate samples from
        each timestep of diffusion.

        Arguments are the same as p_sample_loop().
        Returns a generator over dicts, where each dict is the return value of
        p_sample().
        """
        if device is None:
            device = next(model.parameters()).device
        assert isinstance(shape, (tuple, list))
        if noise is not None:
            img = noise
        else:
            global seed, i_seed
            # seed = 0
            th.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)
            img = th.randn(*shape, device=device)
        indices = list(range(self.num_timesteps))[::-1]
        print('Number of intermediate samples: ', len(indices))

        # i_seed = i_seed + 1
        # seed = seed_all[i_seed]
        
        seed = seed + 1

        if progress:
            # Lazy import so that we don't depend on tqdm.
            from tqdm.auto import tqdm

            indices = tqdm(indices)

        # Code block to save denoised images
        #store_imgs = []

        global nan_condition
        nan_condition = False

        for i in indices:
            t = th.tensor([i] * shape[0], device=device)
            with th.no_grad():
                out = self.p_sample(
                    model,
                    img,
                    cons,
                    t,
                    clip_denoised=clip_denoised,
                    denoised_fn=denoised_fn,
                    cond_fn_1=cond_fn_1,
                    cond_fn_2=cond_fn_2,
                    model_kwargs=model_kwargs,
                )
                yield out
                img = out["sample"]
                
            if nan_condition == True:
                break

        #        if i % 5 == 0:
        #            image = out["sample"].cpu().numpy().squeeze()
        #            plt.imshow(image, cmap="gray")
        #            plt.title(f"Denoised Image: Step {100 - i}")
        #            plt.show()

        #            # Get the number of existing files (assuming naming convention)
        #            import os
        #            save_dir = '/content/drive/MyDrive/main/datas/intermediate_imgs'   
        #            existing_files = [f for f in os.listdir(save_dir) if f.startswith("image") and f.endswith(".png")]
        #            num_existing_images = len(existing_files)

        #            # Construct filename with unique image number (j)
        #            filename = f"image_{num_existing_images//20}_denoised_step_{100 - i}.png"
        #            filepath = os.path.join(save_dir, filename)

                    # Save the image
        #            plt.imsave(filepath, image, cmap="gray")
        #            print(f"Saved image: {filepath}")

        #            image = th.from_numpy(image)
        #            image = image.unsqueeze(0)
        #            image = image.unsqueeze(0)
        #            store_imgs.append(image)

        #if store_imgs:
            # Concatenate images along a new dimension (usually 0 for batch)
        #    concatenated_images = th.cat(store_imgs, dim=0)
        #    filename = f"intermediate_array_{num_existing_images//20}.pt"
        #    filepath = os.path.join(save_dir, filename)
        #    th.save(concatenated_images, filepath)




    def _vb_terms_bpd(
        self, model, x_start, x_t, cons, t, clip_denoised=True, model_kwargs=None
    ):
        """
        Get a term for the variational lower-bound.

        The resulting units are bits (rather than nats, as one might expect).
        This allows for comparison to other papers.

        :return: a dict with the following keys:
                 - 'output': a shape [N] tensor of NLLs or KLs.
                 - 'pred_xstart': the x_0 predictions.
        """
        true_mean, _, true_log_variance_clipped = self.q_posterior_mean_variance(
            x_start=x_start, x_t=x_t, t=t
        )
        out = self.p_mean_variance(
            model, x_t, cons, t, clip_denoised=clip_denoised, model_kwargs=model_kwargs
        )
        kl = normal_kl(
            true_mean, true_log_variance_clipped, out["mean"], out["log_variance"]
        )
        kl = mean_flat(kl) / np.log(2.0)

        decoder_nll = -discretized_gaussian_log_likelihood(
            x_start, means=out["mean"], log_scales=0.5 * out["log_variance"]
        )
        assert decoder_nll.shape == x_start.shape
        decoder_nll = mean_flat(decoder_nll) / np.log(2.0)

        # At the first timestep return the decoder NLL,
        # otherwise return KL(q(x_{t-1}|x_t,x_0) || p(x_{t-1}|x_t))
        output = th.where((t == 0), decoder_nll, kl)
        return {"output": output, "pred_xstart": out["pred_xstart"]}

    def training_losses(self, model, x_start, cons, t, model_kwargs=None, noise=None):
        """
        Compute training losses for a single timestep.

        :param model: the model to evaluate loss on.
        :param x_start: the [N x C x ...] tensor of inputs.
        :param cons: the channels corresponding to the constraints used by the diffusion model.
        :param t: a batch of timestep indices.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model.
        :param noise: if specified, the specific Gaussian noise to try to remove.
        :return: a dict with the key "loss" containing a tensor of shape [N].
                 Some mean or variance settings may also have other keys.
        """
        if model_kwargs is None:
            model_kwargs = {}
        if noise is None:
            noise = th.randn_like(x_start)
        x_t = self.q_sample(x_start, t, noise=noise)

        terms = {}

        if self.loss_type == LossType.KL or self.loss_type == LossType.RESCALED_KL:
            terms["loss"] = self._vb_terms_bpd(
                model=model,
                x_start=x_start,
                x_t=x_t,
                cons=cons,
                t=t,
                clip_denoised=False,
                model_kwargs=model_kwargs,
            )["output"]
            if self.loss_type == LossType.RESCALED_KL:
                terms["loss"] *= self.num_timesteps
        elif self.loss_type == LossType.MSE or self.loss_type == LossType.RESCALED_MSE:
            full_arr = th.cat((x_t, cons), dim = 1)
            model_output = model(full_arr, self._scale_timesteps(t), **model_kwargs)

            if self.model_var_type in [
                ModelVarType.LEARNED,
                ModelVarType.LEARNED_RANGE,
            ]:
                B, C = x_t.shape[:2]
                assert model_output.shape == (B, C * 2, *x_t.shape[2:])
                model_output, model_var_values = th.split(model_output, C, dim=1)
                # Learn the variance using the variational bound, but don't let
                # it affect our mean prediction.
                frozen_out = th.cat([model_output.detach(), model_var_values], dim=1)
                terms["vb"] = self._vb_terms_bpd(
                    model=lambda *args, r=frozen_out: r,
                    x_start=x_start,
                    x_t=x_t,
                    cons=cons,
                    t=t,
                    clip_denoised=False,
                )["output"]
                if self.loss_type == LossType.RESCALED_MSE:
                    # Divide by 1000 for equivalence with initial implementation.
                    # Without a factor of 1/1000, the VB term hurts the MSE term.
                    terms["vb"] *= self.num_timesteps / 1000.0

            target = {
                ModelMeanType.PREVIOUS_X: self.q_posterior_mean_variance(
                    x_start=x_start, x_t=x_t, t=t
                )[0],
                ModelMeanType.START_X: x_start,
                ModelMeanType.EPSILON: noise,
            }[self.model_mean_type]
            assert model_output.shape == target.shape == x_start.shape
            terms["mse"] = mean_flat((target - model_output) ** 2)
            if "vb" in terms:
                terms["loss"] = terms["mse"] + terms["vb"]
            else:
                terms["loss"] = terms["mse"]
        else:
            raise NotImplementedError(self.loss_type)

        return terms

    def _prior_bpd(self, x_start):
        """
        Get the prior KL term for the variational lower-bound, measured in
        bits-per-dim.

        This term can't be optimized, as it only depends on the encoder.

        :param x_start: the [N x C x ...] tensor of inputs.
        :return: a batch of [N] KL values (in bits), one per batch element.
        """
        batch_size = x_start.shape[0]
        t = th.tensor([self.num_timesteps - 1] * batch_size, device=x_start.device)
        qt_mean, _, qt_log_variance = self.q_mean_variance(x_start, t)
        kl_prior = normal_kl(
            mean1=qt_mean, logvar1=qt_log_variance, mean2=0.0, logvar2=0.0
        )
        return mean_flat(kl_prior) / np.log(2.0)

    def calc_bpd_loop(self, model, x_start, cons, clip_denoised=True, model_kwargs=None):
        """
        Compute the entire variational lower-bound, measured in bits-per-dim,
        as well as other related quantities.

        :param model: the model to evaluate loss on.
        :param x_start: the [N x C x ...] tensor of inputs.
        :param cons: the channels corresponding to the constraints used by the diffusion model.
        :param clip_denoised: if True, clip denoised samples.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model.

        :return: a dict containing the following keys:
                 - total_bpd: the total variational lower-bound, per batch element.
                 - prior_bpd: the prior term in the lower-bound.
                 - vb: an [N x T] tensor of terms in the lower-bound.
                 - xstart_mse: an [N x T] tensor of x_0 MSEs for each timestep.
                 - mse: an [N x T] tensor of epsilon MSEs for each timestep.
        """
        device = x_start.device
        batch_size = x_start.shape[0]

        vb = []
        xstart_mse = []
        mse = []
        for t in list(range(self.num_timesteps))[::-1]:
            t_batch = th.tensor([t] * batch_size, device=device)
            noise = th.randn_like(x_start)
            x_t = self.q_sample(x_start=x_start, t=t_batch, noise=noise)
            # Calculate VLB term at the current timestep
            with th.no_grad():
                out = self._vb_terms_bpd(
                    model,
                    x_start=x_start,
                    x_t=x_t,
                    cons = cons,
                    t=t_batch,
                    clip_denoised=clip_denoised,
                    model_kwargs=model_kwargs,
                )
            vb.append(out["output"])
            xstart_mse.append(mean_flat((out["pred_xstart"] - x_start) ** 2))
            eps = self._predict_eps_from_xstart(x_t, t_batch, out["pred_xstart"])
            mse.append(mean_flat((eps - noise) ** 2))

        vb = th.stack(vb, dim=1)
        xstart_mse = th.stack(xstart_mse, dim=1)
        mse = th.stack(mse, dim=1)

        prior_bpd = self._prior_bpd(x_start)
        total_bpd = vb.sum(dim=1) + prior_bpd
        return {
            "total_bpd": total_bpd,
            "prior_bpd": prior_bpd,
            "vb": vb,
            "xstart_mse": xstart_mse,
            "mse": mse,
        }


def _extract_into_tensor(arr, timesteps, broadcast_shape):
    """
    Extract values from a 1-D numpy array for a batch of indices.

    :param arr: the 1-D numpy array.
    :param timesteps: a tensor of indices into the array to extract.
    :param broadcast_shape: a larger shape of K dimensions with the batch
                            dimension equal to the length of timesteps.
    :return: a tensor of shape [batch_size, 1, ...] where the shape has K dims.
    """
    res = th.from_numpy(arr).to(device=timesteps.device)[timesteps].float()
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape)
