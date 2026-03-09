import matplotlib.pyplot as plt
import torch
from torchskradon.functional import skradon

import torchcdt.helpers as helpers
from torchcdt.functional import ircdt, rcdt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

N = 128
s = torch.zeros((1, 1, N, N), device=device)
x = torch.linspace(-10, 10, N, device=device)
y = torch.linspace(-10, 10, N, device=device)
xx, yy = torch.meshgrid(x, y, indexing="ij")
x = x.unsqueeze(0)
y = y.unsqueeze(0)
xx = xx.unsqueeze(0)
yy = yy.unsqueeze(0)
s[0, 0] = torch.exp(-(xx**2 + yy**2) / 2) / (2 * torch.pi)

# This is not needed unless we want the reconstruction to have the exact same scaling
s = helpers.make_positive_density(s, dim=(-2, -1), eps=1e-6)
s_sinogram = skradon(s, circle=False)

fig, axs = plt.subplots(1, 2, figsize=(15, 5))
axs[0].imshow(s[0, 0].cpu().numpy())
axs[0].set_title("Input image")
axs[1].imshow(s_sinogram[0, 0].cpu().numpy())
axs[1].set_title("Sinogram")
plt.show()

s_hat = rcdt(s, circle=False)
s_reco = ircdt(s_hat, circle=False)

fig, ax = plt.subplots(1, 2, figsize=(15, 5))
ax[0].imshow(s_hat[0, 0].cpu().numpy())
ax[0].set_title("R-CDT")
ax[1].imshow(s_reco[0, 0].cpu().numpy())
ax[1].set_title("Reconstructed image")
plt.show()
