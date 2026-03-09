import matplotlib.pyplot as plt
import pytest
import torch
from scipy.stats import norm

import torchcdt.helpers as helpers
from torchcdt.functional import cdt, icdt, ircdt, rcdt

torch.manual_seed(123456789)
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
    phi_inv = torch.from_numpy(norm.ppf(x_ref_.cpu().numpy())).unsqueeze(0).to(device, s_hat.dtype)
    if debug_plot:
        plt.plot(s_hat.squeeze().cpu().numpy())
        plt.plot(phi_inv.squeeze().cpu().numpy())
        plt.show()
    assert torch.allclose(s_hat[:, :, 1:-1], phi_inv[:, :, 1:-1], atol=3e-1)


@pytest.mark.parametrize("s_ref", [None, torch.ones(1, 1, N).to(device)])
@pytest.mark.parametrize("x_ref", [None, torch.linspace(0, 1, N).unsqueeze(0).to(device)])
def test_cdt(s_ref, x_ref):
    check_cdt(s_ref, x_ref)


def check_cdt_icdt(s_ref, x_ref):
    x = torch.linspace(-10, 10, N).unsqueeze(0).to(device)
    s = (
        (1 / torch.sqrt(torch.tensor(2 * torch.pi)) * torch.exp(-(x**2) / 2))
        .unsqueeze(0)
        .to(device)
    )
    s = helpers.make_positive_density(s, eps=1e-6)

    s_hat = cdt(s, x, s_ref, x_ref)
    s_reco = icdt(s_hat, x, s_ref, x_ref)
    if debug_plot:
        plt.plot(s.squeeze().cpu().numpy())
        plt.plot(s_reco.squeeze().cpu().numpy())
        plt.show()
    assert torch.allclose(s, s_reco, atol=1e-4)


@pytest.mark.parametrize("s_ref", [None, torch.ones(1, 1, N).to(device)])
@pytest.mark.parametrize("x_ref", [None, torch.linspace(0, 1, N).unsqueeze(0).to(device)])
def test_cdt_icdt(s_ref, x_ref):
    check_cdt_icdt(s_ref, x_ref)


def check_cdt_autograd(s_ref, x_ref):
    x = torch.linspace(-10, 10, N).unsqueeze(0).to(device)
    s = (
        (1 / torch.sqrt(torch.tensor(2 * torch.pi)) * torch.exp(-(x**2) / 2))
        .unsqueeze(0)
        .to(device)
        .to(torch.double)
    )

    s.requires_grad = True
    if torch.unique(s_ref).numel() > 1:
        s_ref = s_ref.to(torch.double)
        s_ref.requires_grad = True
        x_ref.requires_grad = False
    if x is not None:
        x = x.to(torch.double)
        x.requires_grad = True

    assert torch.autograd.gradcheck(cdt, (s, x, s_ref, x_ref), fast_mode=True, nondet_tol=1e-8)


@pytest.mark.parametrize("s_ref", [torch.randn(1, 1, N).to(device), torch.ones(1, 1, N).to(device)])
@pytest.mark.parametrize("x_ref", [torch.linspace(0, 1, N).unsqueeze(0).to(device)])
def test_cdt_autograd(s_ref, x_ref):
    check_cdt_autograd(s_ref, x_ref)


def check_icdt_autograd(s_ref, x_ref):
    x = torch.linspace(-10, 10, N).unsqueeze(0).to(device)
    s = (
        (1 / torch.sqrt(torch.tensor(2 * torch.pi)) * torch.exp(-(x**2) / 2))
        .unsqueeze(0)
        .to(device)
        .to(torch.double)
    )

    s_hat = cdt(s, x, s_ref, x_ref)

    s_hat.requires_grad = True
    x = x.to(torch.double)
    x.requires_grad = True
    s_ref = s_ref.to(torch.double)
    s_ref.requires_grad = True
    x_ref = x_ref.to(torch.double)
    x_ref.requires_grad = True

    assert torch.autograd.gradcheck(icdt, (s_hat, x, s_ref, x_ref), fast_mode=True, nondet_tol=1e-8)


@pytest.mark.parametrize("s_ref", [torch.ones(1, 1, N).to(device)])
@pytest.mark.parametrize("x_ref", [torch.linspace(0, 1, N).unsqueeze(0).to(device)])
def test_icdt_autograd(s_ref, x_ref):
    check_icdt_autograd(s_ref, x_ref)


# From hereinafter, we use circe=True to make shapes of s_ref and x_ref more straightforward
def check_rcdt(s_ref, x_ref, normalization):
    s = torch.zeros((1, 1, N, N), device=device)
    x = torch.linspace(-10, 10, N, device=device)
    y = torch.linspace(-10, 10, N, device=device)
    xx, yy = torch.meshgrid(x, y, indexing="ij")
    x = x.unsqueeze(0)
    y = y.unsqueeze(0)
    xx = xx.unsqueeze(0)
    yy = yy.unsqueeze(0)
    s[0, 0] = torch.exp(-(xx**2 + yy**2) / 2) / (2 * torch.pi)
    s = helpers.make_positive_density(s, dim=(-2, -1), eps=1e-6)
    s_hat = rcdt(s, x, s_ref, x_ref, circle=True, normalization=normalization)
    if x_ref is None:
        x_ref_ = torch.linspace(0, 1, N).unsqueeze(0).to(device)
    else:
        x_ref_ = x_ref
    phi_inv = torch.from_numpy(norm.ppf(x_ref_.cpu().numpy())).unsqueeze(0).to(device, s_hat.dtype)
    if debug_plot is True:
        fig, axs = plt.subplots(1)
        for i in range(s_hat.shape[-1]):
            axs.plot(phi_inv[0, 0, :].cpu().numpy())
            axs.plot(s_hat[0, 0, :, i].cpu().numpy())
        plt.show()
        plt.close()
    if normalization == "mean" or normalization == "max":
        assert torch.allclose(s_hat[:, :, 1:-1, :], phi_inv[:, :, 1:-1, None], atol=3e-1)
    else:
        assert torch.allclose(s_hat[:, :, 1:-1, :], phi_inv[:, :, 1:-1, None], atol=3e-1)


@pytest.mark.parametrize("s_ref", [None, torch.ones(1, 1, N, 180).to(device)])
@pytest.mark.parametrize("x_ref", [None, torch.linspace(0, 1, N).unsqueeze(0).to(device)])
@pytest.mark.parametrize("normalization", [None, "mean", "max"])
def test_rcdt(s_ref, x_ref, normalization):
    check_rcdt(s_ref, x_ref, normalization)


def check_rcdt_ircdt(s_ref, x_ref):
    s = torch.zeros((1, 1, N, N), device=device)
    x = torch.linspace(-10, 10, N, device=device)
    y = torch.linspace(-10, 10, N, device=device)
    xx, yy = torch.meshgrid(x, y, indexing="ij")
    x = x.unsqueeze(0)
    y = y.unsqueeze(0)
    xx = xx.unsqueeze(0)
    yy = yy.unsqueeze(0)
    s[0, 0] = torch.exp(-(xx**2 + yy**2) / 2) / (2 * torch.pi)
    s = helpers.make_positive_density(s, dim=(-2, -1), eps=1e-6)
    s_hat = rcdt(s, x, s_ref, x_ref, circle=True)

    s_reco = ircdt(s_hat, x, s_ref, x_ref, eps=1e-6)
    if debug_plot is True:
        fig, axs = plt.subplots(2)
        axs[0].imshow(s[0, 0, :, :].cpu().numpy())
        axs[1].imshow(s_reco[0, 0, :, :].cpu().numpy())
        plt.show()
        plt.close()

    assert torch.allclose(s, s_reco, atol=1e-4)


@pytest.mark.parametrize("s_ref", [None, torch.ones(1, 1, N, 180).to(device)])
@pytest.mark.parametrize("x_ref", [None, torch.linspace(0, 1, N).unsqueeze(0).to(device)])
def test_rcdt_ircdt(s_ref, x_ref):
    check_rcdt_ircdt(s_ref, x_ref)


def check_rcdt_autograd(s_ref, x_ref, normalization):
    s = torch.zeros((1, 1, N, N), device=device)
    x = torch.linspace(-10, 10, N, device=device)
    y = torch.linspace(-10, 10, N, device=device)
    xx, yy = torch.meshgrid(x, y, indexing="ij")
    x = x.unsqueeze(0)
    y = y.unsqueeze(0)
    xx = xx.unsqueeze(0)
    yy = yy.unsqueeze(0)
    s[0, 0] = torch.exp(-(xx**2 + yy**2) / 2) / (2 * torch.pi)
    s = s.to(torch.double)
    s.requires_grad = True
    if torch.unique(s_ref).numel() > 1:
        s_ref = s_ref.to(torch.double)
        s_ref.requires_grad = True
        x_ref.requires_grad = False
    if x is not None:
        x = x.to(torch.double)
        x.requires_grad = True
    assert torch.autograd.gradcheck(
        rcdt, (s, x, s_ref, x_ref, normalization), fast_mode=True, nondet_tol=1e-8
    )


@pytest.mark.parametrize(
    "s_ref", [torch.randn(1, 1, N, 180).to(device), torch.ones(1, 1, N, 180).to(device)]
)
@pytest.mark.parametrize("x_ref", [torch.linspace(0, 1, N).unsqueeze(0).to(device)])
@pytest.mark.parametrize("normalization", [None, "mean", "max"])
def test_rcdt_autograd(s_ref, x_ref, normalization):
    check_rcdt_autograd(s_ref, x_ref, normalization)


def check_ircdt_autograd(s_ref, x_ref):
    s = torch.zeros((1, 1, N, N), device=device)
    x = torch.linspace(-10, 10, N, device=device)
    y = torch.linspace(-10, 10, N, device=device)
    xx, yy = torch.meshgrid(x, y, indexing="ij")
    x = x.unsqueeze(0)
    y = y.unsqueeze(0)
    xx = xx.unsqueeze(0)
    yy = yy.unsqueeze(0)
    s[0, 0] = torch.exp(-(xx**2 + yy**2) / 2) / (2 * torch.pi)
    s = s.to(torch.double)
    s_hat = rcdt(s, x, s_ref, x_ref, circle=True)

    s_hat.requires_grad = True
    x = x.to(torch.double)
    x.requires_grad = True
    s_ref = s_ref.to(torch.double)
    s_ref.requires_grad = True
    x_ref = x_ref.to(torch.double)
    x_ref.requires_grad = True

    assert torch.autograd.gradcheck(
        ircdt, (s_hat, x, s_ref, x_ref), fast_mode=True, nondet_tol=1e-8
    )


@pytest.mark.parametrize("s_ref", [torch.ones(1, 1, N, 180).to(device)])
@pytest.mark.parametrize("x_ref", [torch.linspace(0, 1, N).unsqueeze(0).to(device)])
def test_ircdt_autograd(s_ref, x_ref):
    check_ircdt_autograd(s_ref, x_ref)
