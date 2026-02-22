import matplotlib.pyplot as plt
import torch
import torchvision.transforms as transforms
from torchskradon.functional import skradon

from torchcdt.functional import ircdt, rcdt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

s = torch.zeros((1, 1, 128, 128), device=device)
s[0, 0, 64, 64] = 1
s = transforms.GaussianBlur(6 * 20 + 1, sigma=20)(s)

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
