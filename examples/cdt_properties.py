# https://github.com/rohdelab/PyTransKit/blob/master/tutorials/01_tutorial_cdt.ipynb
import matplotlib.pyplot as plt
import torch
from scipy.stats import norm

import torchcdt.helpers as helpers
from torchcdt.functional import cdt, icdt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

N = 500
x = torch.linspace(-10, 10, N, device=device).unsqueeze(0)
s = helpers.make_positive_density(
    (1 / torch.sqrt(torch.tensor(2) * torch.pi) * torch.exp(-(x**2) / 2)).unsqueeze(0)
)

s_ref = torch.ones((1, 1, N), device=device)
x_ref = torch.linspace(0, 1, N, device=device).unsqueeze(0)
s_hat = cdt(s, x, s_ref, x_ref, eps=1e-6)

# Translation
s_trans = helpers.make_positive_density(
    (1 / torch.sqrt(torch.tensor(2) * torch.pi) * torch.exp(-((x + 2.5) ** 2) / 2)).unsqueeze(0)
)
s_trans_hat = cdt(s_trans, x, s_ref, x_ref, eps=1e-6)

fig, ax = plt.subplots(1, 2, figsize=(15, 5))
ax[0].plot(x[0].cpu().numpy(), s[0, 0].cpu().numpy(), label="Original")
ax[0].plot(x[0].cpu().numpy(), s_trans[0, 0].cpu().numpy(), label="Translated")
ax[0].set_title("Gaussian")
ax[0].legend()
ax[1].plot(x_ref[0].cpu().numpy(), s_hat[0, 0].cpu().numpy(), label="Original")
ax[1].plot(x_ref[0].cpu().numpy(), s_trans_hat[0, 0].cpu().numpy(), label="Translated")
ax[1].set_title("CDT")
ax[1].legend()
plt.show()

# scaling
s_scale = helpers.make_positive_density(
    (1 / torch.sqrt(torch.tensor(2) * torch.pi) * torch.exp(-((2 * x) ** 2) / 2)).unsqueeze(0)
)
s_scale_hat = cdt(s_scale, x, s_ref, x_ref, eps=1e-6)
fig, ax = plt.subplots(1, 2, figsize=(15, 5))
ax[0].plot(x[0].cpu().numpy(), s[0, 0].cpu().numpy(), label="Original")
ax[0].plot(x[0].cpu().numpy(), s_scale[0, 0].cpu().numpy(), label="Scaled")
ax[0].set_title("Gaussian")
ax[0].legend()
ax[1].plot(x_ref[0].cpu().numpy(), s_hat[0, 0].cpu().numpy(), label="Original")
ax[1].plot(x_ref[0].cpu().numpy(), s_scale_hat[0, 0].cpu().numpy(), label="Scaled")
ax[1].set_title("CDT")
ax[1].legend()
plt.show()

# composition
s_comp = helpers.make_positive_density(
    (
        2 * x / torch.sqrt(torch.tensor(2) * torch.pi) * torch.exp(-((2 * x + 2.5) ** 2) / 2)
    ).unsqueeze(0)
)
s_comp_hat = cdt(s_comp, x, s_ref, x_ref)

fig, ax = plt.subplots(1, 2, figsize=(15, 5))
ax[0].plot(x[0].cpu().numpy(), s[0, 0].cpu().numpy(), label="Original")
ax[0].plot(x[0].cpu().numpy(), s_comp[0, 0].cpu().numpy(), label="Composed")
ax[0].set_title("Gaussian")
ax[0].legend()
ax[1].plot(x_ref[0].cpu().numpy(), s_hat[0, 0].cpu().numpy(), label="Original")
ax[1].plot(x_ref[0].cpu().numpy(), s_comp_hat[0, 0].cpu().numpy(), label="Composed")
ax[1].set_title("CDT")
ax[1].legend()
plt.show()

# Reconstruction
phi_inv = (
    torch.from_numpy(norm.ppf(x_ref[0].cpu().numpy())).unsqueeze(0).unsqueeze(0).to(device=device)
)
s_reco = icdt(s_hat, x, s_ref, x_ref, eps=1e-6)

fig, ax = plt.subplots(1, 4, figsize=(15, 5))
ax[0].plot(x[0].cpu().numpy(), s[0, 0].cpu().numpy())
ax[0].set_title("Gaussian")
ax[1].plot(x_ref[0].cpu().numpy(), s_hat[0, 0].cpu().numpy())
ax[1].set_title("CDT")
ax[1].set_ylim(-3, 3)
ax[2].plot(x[0].cpu().numpy(), s_reco[0, 0].cpu().numpy())
ax[2].set_title("Reconstructed signal")
ax[3].plot(x_ref[0].cpu().numpy(), phi_inv[0, 0].cpu().numpy())
ax[3].set_title(r"$\phi^{-1}$")
plt.show()
