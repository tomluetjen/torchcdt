# torchcdt

[![LICENSE](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE) [![Python](https://img.shields.io/badge/python-3.11%20|%203.12%20|%203.13-blue)](#) [![PyPI](https://img.shields.io/pypi/v/torchcdt.svg?label=PyPI&logo=pypi)](https://pypi.org/project/torchcdt/) [![CI](https://github.com/tomluetjen/torchcdt/actions/workflows/python-app.yml/badge.svg?branch=main)](https://github.com/tomluetjen/torchcdt/actions/workflows/python-app.yml) [![Coverage](https://codecov.io/gh/tomluetjen/torchcdt/branch/main/graph/badge.svg)](https://codecov.io/gh/tomluetjen/torchcdt)

## About
`torchcdt` implements the (Radon-) Cumulative Distribution Transform [1, 2] and its' inverse with different normalization approaches [3, 4] to enhance feature extraction. All transforms work with batched multi-channel data and are fully differentiable. This allows backpropagation through `torchcdt` transforms to train neural networks or to solve optimization problems with [`torch.optim`](https://docs.pytorch.org/docs/stable/optim.html) (see [examples](#examples)).

## Installation
```console
pip install torchcdt
```

## Basic Usage
```python
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
```

## Examples
For more detailed examples and use cases, see the `examples` directory:

- [`examples/cdt_properties.py`](examples/cdt_properties.py) - Basic properties of forward and inverse CDT transforms
- [`examples/linmnist_classifictation.ipynb`](examples/linmnist_classifictation.ipynb) - SVM classification of LinMNIST dataset

## Other Packages
This package is inspired by

1. [`PyTransKit`](https://github.com/rohdelab/PyTransKit)

2. [`Cumulative-Distribution-Transform`](https://github.com/skolouri/Cumulative-Distribution-Transform)

3. [`Radon-Cumulative-Distribution-Transform`](https://github.com/skolouri/Radon-Cumulative-Distribution-Transform)

4. [`NR-CDT`](https://github.com/DrBeckmann/NR-CDT)

## Acknowledgements
I would like to thank Dr. Matthias Beckmann for his supervision and guidance during a reading course on this topic.

## References
1. Park SR, Kolouri S, Kundu S, Rohde GK, ["The cumulative distribution transform and linear pattern classification"](https://www.sciencedirect.com/science/article/pii/S1063520317300076), Applied and Computational Harmonic Analysis, 2017

2. Kolouri S, Park SR, Rohde GK, ["The Radon Cumulative Distribution Transform and Its Application to Image Classification"](https://ieeexplore.ieee.org/abstract/document/7358128), IEEE transactions on image processing, 2016

3. Beckmann M, Beinert R, Bresch J, ["Max-Normalized Radon Cumulative Distribution Transform for
   Limited Data Classification"](https://doi.org/10.1007/978-3-031-92366-1_19), International Conference on Scale Space and Variational Methods in Computer Vision (SSVM), 2025

4. Beckmann M, Beinert R, Bresch J, ["Normalized Radon Cumulative Distribution Transforms for
   Invariance and Robustness in Optimal Transport Based
   Image Classification"](https://doi.org/10.48550/arXiv.2506.08761), International Conference on Scale Space and Variational Methods in Computer Vision (SSVM), 2025