import helpers
import torch
from torchskradon.functional import skiradon, skradon


def cdt(s, x=None, s_ref=None, x_ref=None, eps=1e-6):
    if x is None:
        x = torch.linspace(0, 1, s.shape[-1])[None, ...]
    if s_ref is None:
        s_ref = torch.ones_like(s)
    if x_ref is None:
        x_ref = torch.linspace(0, 1, s_ref.shape[-1])[None, ...]
    s = helpers.make_positive_density(s, eps=eps)
    s_ref = helpers.make_positive_density(s_ref, eps=eps)
    step_size = (x_ref[..., -1] - x_ref[..., 0]) / (x_ref.shape[-1] - 1)
    s_cdf = torch.cumsum(s, dim=-1)
    s_ref_cdf = torch.cumsum(s_ref, dim=-1)
    if s_ref is None:
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
    if x is None:
        x = torch.linspace(0, 1, s_hat.shape[-1])[None, ...]
    if s_ref is None:
        s_ref = torch.ones_like(s_hat)
    if x_ref is None:
        x_ref = torch.linspace(0, 1, s_ref.shape[-1])[None, ...]
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


def rcdt(s, x=None, s_ref=None, x_ref=None, eps=1e-6, *args):
    s_sinogram = torch.transpose(skradon(s, *args, circle=False), -2, -1)
    if x is None:
        x = torch.linspace(0, 1, s_sinogram.shape[-1])[None, ...]
    if s_ref is None:
        s_ref_sinogram = torch.ones_like(s_sinogram)
    else:
        s_ref_sinogram = torch.transpose(skradon(s_ref, *args, circle=False), -2, -1)
    if x_ref is None:
        x_ref = torch.linspace(0, 1, s_ref_sinogram.shape[-1])[None, ...]
    s_hat = cdt(
        s_sinogram,
        x,
        s_ref_sinogram,
        x_ref,
        eps=eps,
    )
    return torch.transpose(s_hat, -2, -1)


def ircdt(s_hat, x=None, s_ref=None, x_ref=None, eps=1e-6, *args):
    s_hat = torch.transpose(s_hat, -2, -1)
    if x is None:
        x = torch.linspace(0, 1, s_hat.shape[-1])[None, ...]
    if s_ref is None:
        s_ref_sinogram = torch.ones_like(s_hat)
    else:
        s_ref_sinogram = torch.transpose(skradon(s_ref, *args, circle=False), -2, -1)
    if x_ref is None:
        x_ref = torch.linspace(0, 1, s_ref_sinogram.shape[-1])[None, ...]

    s_sinogram = icdt(
        s_hat,
        x,
        s_ref_sinogram,
        x_ref,
        eps=eps,
    )
    s = skiradon(torch.transpose(s_sinogram, -2, -1), *args, circle=False)
    s = helpers.make_positive_density(s, dim=(-2, -1), eps=eps)
    return s
