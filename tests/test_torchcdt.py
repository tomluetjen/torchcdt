import os
import sys

import matplotlib.pyplot as plt
import pytest
import torch
from scipy.stats import norm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from torchcdt.functional import cdt, icdt

debug_plot = False
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N = 1000


def check_cdt(s_ref, x_ref):
    x = torch.linspace(-10, 10, N).unsqueeze(0).to(device)
    s = (
        (1 / torch.sqrt(torch.tensor(2 * torch.pi)) * torch.exp(-(x**2) / 2))
        .unsqueeze(0)
        .to(device)
    )
    if x_ref is None:
        x_ref_ = torch.linspace(0, 1, N).unsqueeze(0).to(device)
    else:
        x_ref_ = x_ref

    s_hat = cdt(s, x, s_ref, x_ref)
    phi_inv = (
        torch.from_numpy(norm.ppf(x_ref_.cpu().numpy()))
        .unsqueeze(0)
        .to(device, s_hat.dtype)
    )
    if debug_plot:
        plt.plot(s_hat.squeeze().cpu().numpy())
        plt.plot(phi_inv.squeeze().cpu().numpy())
        plt.show()
    assert torch.allclose(s_hat[:, :, 1:-1], phi_inv[:, :, 1:-1], atol=3e-1)


@pytest.mark.parametrize("s_ref", [None, torch.ones(1, 1, N).to(device)])
@pytest.mark.parametrize(
    "x_ref", [None, torch.linspace(0, 1, N).unsqueeze(0).to(device)]
)
def test_cdt(s_ref, x_ref):
    check_cdt(s_ref, x_ref)


def check_cdt_icdt(s_ref, x_ref):
    x = torch.linspace(-10, 10, N).unsqueeze(0).to(device)
    s = (
        (1 / torch.sqrt(torch.tensor(2 * torch.pi)) * torch.exp(-(x**2) / 2))
        .unsqueeze(0)
        .to(device)
    )

    s_hat = cdt(s, x, s_ref, x_ref)
    s_reco = icdt(s_hat, x, s_ref, x_ref)
    if debug_plot:
        plt.plot(s.squeeze().cpu().numpy())
        plt.plot(s_reco.squeeze().cpu().numpy())
        plt.show()
    assert torch.allclose(s, s_reco, atol=1e-3)


@pytest.mark.parametrize("s_ref", [None, torch.ones(1, 1, N).to(device)])
@pytest.mark.parametrize(
    "x_ref", [None, torch.linspace(0, 1, N).unsqueeze(0).to(device)]
)
def test_cdt_icdt(s_ref, x_ref):
    check_cdt_icdt(s_ref, x_ref)
