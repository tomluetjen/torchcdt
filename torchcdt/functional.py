import torch
from torchskradon.functional import skiradon, skradon

import torchcdt.helpers as helpers


def cdt(s, x=None, s_ref=None, x_ref=None, eps=1e-6):
    device = s.device
    if x is None:
        x = torch.linspace(0, 1, s.shape[-1])[None, ...].to(device=device)
    if s_ref is None:
        s_ref = torch.ones_like(s)
    if x_ref is None:
        x_ref = torch.linspace(0, 1, s_ref.shape[-1])[None, ...].to(device=device)
    s = helpers.make_positive_density(s, eps=eps)
    s_ref = helpers.make_positive_density(s_ref, eps=eps)
    step_size = (x_ref[..., -1] - x_ref[..., 0]) / (x_ref.shape[-1] - 1)
    s_cdf = torch.cumsum(s, dim=-1)
    s_ref_cdf = torch.cumsum(s_ref, dim=-1)
    if torch.unique(s_ref).numel() == 1:
        s_hat = helpers.interp(
            torch.broadcast_to(x_ref[:, *((None,) * (s.ndim - 2)), :], s_ref_cdf.shape),
            s_cdf,
            torch.broadcast_to(x[:, *((None,) * (s.ndim - 2)), :], s_cdf.shape),
        )
    else:
        s_hat = helpers.interp(
            step_size * s_ref_cdf,
            step_size * s_cdf,
            torch.broadcast_to(x[:, *((None,) * (s.ndim - 2)), :], s_cdf.shape),
        )
    return s_hat


def icdt(s_hat, x=None, s_ref=None, x_ref=None, eps=1e-6):
    device = s_hat.device
    if x is None:
        x = torch.linspace(0, 1, s_hat.shape[-1])[None, ...].to(device=device)
    if s_ref is None:
        s_ref = torch.ones_like(s_hat)
    if x_ref is None:
        x_ref = torch.linspace(0, 1, s_ref.shape[-1])[None, ...].to(device=device)
    # torch.gradient is not batchable w.r.t. spacing
    grad_s_hat = torch.gradient(s_hat, dim=-1)[0]
    grad_x_ref = torch.gradient(x_ref[:, *((None,) * (s_hat.ndim - 2)), :], dim=-1)[0]
    grad = grad_s_hat / grad_x_ref
    # We need to clamp to guarantee that we do not divide by 0
    grad = torch.clamp(grad, min=torch.finfo(grad.dtype).eps)
    s = helpers.interp(
        torch.broadcast_to(x[:, *((None,) * (s_hat.ndim - 2)), :], s_hat.shape),
        s_hat,
        s_ref / grad,
    )
    s = helpers.make_positive_density(s, eps=eps)
    return s


def rcdt(s, x=None, s_ref=None, x_ref=None, normalization=None, eps=1e-6, **kwargs):
    device = s.device
    s_sinogram = torch.transpose(skradon(s, **kwargs), -2, -1)
    if x is None:
        x = torch.linspace(0, 1, s_sinogram.shape[-1])[None, ...].to(device=device)
    if s_ref is None:
        s_ref_sinogram = torch.ones_like(s_sinogram)
    else:
        s_ref_sinogram = torch.transpose(skradon(s_ref, **kwargs), -2, -1)
    if x_ref is None:
        x_ref = torch.linspace(0, 1, s_ref_sinogram.shape[-1])[None, ...].to(device=device)
    s_hat = cdt(
        s_sinogram,
        x,
        s_ref_sinogram,
        x_ref,
        eps=eps,
    )
    s_hat = torch.transpose(s_hat, -2, -1)
    if normalization is None:
        pass
    elif normalization == "mean":
        s_hat_mean = torch.mean(s_hat, dim=-2, keepdim=True)
        s_hat_std = torch.std(s_hat, dim=-2, keepdim=True)
        s_hat = (s_hat - s_hat_mean) / s_hat_std
        s_hat = torch.mean(s_hat, dim=-1, keepdim=True)
    elif normalization == "max":
        s_hat_mean = torch.mean(s_hat, dim=-2, keepdim=True)
        s_hat_std = torch.std(s_hat, dim=-2, keepdim=True)
        s_hat = (s_hat - s_hat_mean) / s_hat_std
        s_hat = torch.amax(s_hat, dim=-1, keepdim=True)
    else:
        raise ValueError(f"Unknown normalization type: {normalization}")

    return s_hat


def ircdt(s_hat, x=None, s_ref=None, x_ref=None, eps=1e-6, **kwargs):
    device = s_hat.device
    s_hat = torch.transpose(s_hat, -2, -1)
    if x is None:
        x = torch.linspace(0, 1, s_hat.shape[-1])[None, ...].to(device=device)
    if s_ref is None:
        s_ref_sinogram = torch.ones_like(s_hat)
    else:
        s_ref_sinogram = torch.transpose(s_ref, -2, -1)
    if x_ref is None:
        x_ref = torch.linspace(0, 1, s_ref_sinogram.shape[-1])[None, ...].to(device=device)

    s_sinogram = icdt(
        s_hat,
        x,
        s_ref_sinogram,
        x_ref,
        eps=eps,
    )
    s = skiradon(torch.transpose(s_sinogram, -2, -1), **kwargs)
    s = helpers.make_positive_density(s, dim=(-2, -1), eps=eps)
    return s
