import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from dataset import LinMNIST
from torch.utils.data import DataLoader

# from torchskradon.functional import skradon
from torchcdt.functional import rcdt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Load MNIST dataset
train_dataset = LinMNIST(
    train=True,
    num_templates_per_class=1,
    num_samples_per_template=50,
    transform=transforms.RandomAffine(
        degrees=360, translate=(0.2, 0.2), scale=(0.9, 1.1), shear=(20, 20, 20, 20)
    ),
)
test_dataset = LinMNIST(
    # We need to set train to True to get the same templates as the training set.
    train=True,
    num_templates_per_class=1,
    num_samples_per_template=50,
    transform=transforms.RandomAffine(
        degrees=360, translate=(0.2, 0.2), scale=(0.9, 1.1), shear=(20, 20, 20, 20)
    ),
)

batch_size = int(10000)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
# Load the first batch from the dataloader
train_iter = iter(train_loader)
images_train, labels_train = next(train_iter)
test_iter = iter(test_loader)
images_test, labels_test = next(test_iter)
fig, axs = plt.subplots(2, 10, figsize=(15, 5))
for i in range(10):
    axs[0, i].imshow(images_train[i].squeeze(), cmap="gray")
    axs[0, i].set_title(f"Train: {labels_train[i].item()}")
    axs[0, i].axis("off")
    axs[1, i].imshow(images_test[i].squeeze(), cmap="gray")
    axs[1, i].set_title(f"Test: {labels_test[i].item()}")
    axs[1, i].axis("off")
plt.suptitle("Sample images from the training and test sets")
plt.tight_layout()
plt.show()
img_feat = images_train[0].numel()
rcdt_feat = rcdt(images_train)[0].numel()
mean_rcdt_feat = rcdt(images_train, normalization="mean")[0].numel()
max_rcdt_feat = rcdt(images_train, normalization="max")[0].numel()


class LinearSVM(nn.Module):
    def __init__(self, num_features):
        super(LinearSVM, self).__init__()
        self.num_features = num_features
        self.linear = nn.Linear(num_features, 10)

    def forward(self, x):
        x = x.reshape(-1, self.num_features)
        return self.linear(x)


loss_fn = nn.MultiMarginLoss(p=2)
model = LinearSVM(num_features=img_feat).to(device)
optimizer = torch.optim.Adam(model.parameters())

# Training loop
loss_vals = list()
for i in range(1000):
    loss_vals.append(0)
    for batch_img, batch_labels in train_loader:
        optimizer.zero_grad()
        pred = model(batch_img.to(device))
        # Compute loss between predicted and target labels
        loss_value = loss_fn(pred, batch_labels.to(device))
        # Backward pass
        loss_value.backward()
        # Update model parameters
        optimizer.step()
        loss_vals[i] += loss_value.item()
    loss_vals[i] /= len(train_loader)
    if i % 10 == 0:
        print(f"Iteration {i}, Loss: {loss_vals[i]:.6f}")

plt.plot(loss_vals)
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.title("Training Loss over Iterations")
plt.grid()
plt.show()

model.eval()
with torch.no_grad():
    num_correct = 0
    for batch_img, batch_labels in test_loader:
        pred = model(batch_img.to(device))
        predicted_labels = torch.argmax(pred, dim=1)
        num_correct += (predicted_labels == batch_labels.to(device)).sum().item()
    accuracy = num_correct / len(test_dataset)
    print(f"Test Accuracy: {accuracy * 100:.2f}%")

print("Classification in euclidean space completed!")

########################################################################################

loss_fn = nn.MultiMarginLoss(p=2)
model = LinearSVM(num_features=rcdt_feat).to(device)
optimizer = torch.optim.Adam(model.parameters())

# Training loop
loss_vals = list()
for i in range(1000):
    loss_vals.append(0)
    for batch_img, batch_labels in train_loader:
        optimizer.zero_grad()
        pred = model(rcdt(batch_img.to(device)))
        # Compute loss between predicted and target labels
        loss_value = loss_fn(pred, batch_labels.to(device))
        # Backward pass
        loss_value.backward()
        # Update model parameters
        optimizer.step()
        loss_vals[i] += loss_value.item()
    loss_vals[i] /= len(train_loader)

    if i % 10 == 0:
        print(f"Iteration {i}, Loss: {loss_vals[i]:.6f}")

plt.plot(loss_vals)
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.title("Training Loss over Iterations")
plt.grid()
plt.show()

model.eval()
with torch.no_grad():
    num_correct = 0
    for batch_img, batch_labels in test_loader:
        pred = model(rcdt(batch_img.to(device)))
        predicted_labels = torch.argmax(pred, dim=1)
        num_correct += (predicted_labels == batch_labels.to(device)).sum().item()
    accuracy = num_correct / len(test_dataset)
    print(f"Test Accuracy: {accuracy * 100:.2f}%")

print("Classification in R-CDT space completed!")

#########################################################################################
loss_fn = nn.MultiMarginLoss(p=2)
model = LinearSVM(num_features=mean_rcdt_feat).to(device)
optimizer = torch.optim.Adam(model.parameters())

# Training loop
loss_vals = list()
for i in range(1000):
    loss_vals.append(0)
    for batch_img, batch_labels in train_loader:
        optimizer.zero_grad()
        pred = model(rcdt(batch_img.to(device), normalization="mean"))
        # Compute loss between predicted and target labels
        loss_value = loss_fn(pred, batch_labels.to(device))
        # Backward pass
        loss_value.backward()
        # Update model parameters
        optimizer.step()
        loss_vals[i] += loss_value.item()
    loss_vals[i] /= len(train_loader)

    if i % 10 == 0:
        print(f"Iteration {i}, Loss: {loss_vals[i]:.6f}")

plt.plot(loss_vals)
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.title("Training Loss over Iterations")
plt.grid()
plt.show()

model.eval()
with torch.no_grad():
    num_correct = 0
    for batch_img, batch_labels in test_loader:
        pred = model(rcdt(batch_img.to(device), normalization="mean"))
        predicted_labels = torch.argmax(pred, dim=1)
        num_correct += (predicted_labels == batch_labels.to(device)).sum().item()
    accuracy = num_correct / len(test_dataset)
    print(f"Test Accuracy: {accuracy * 100:.2f}%")

print("Classification in mean-normalized R-CDT space completed!")


########################################################################################
loss_fn = nn.MultiMarginLoss(p=2)
model = LinearSVM(num_features=max_rcdt_feat).to(device)
optimizer = torch.optim.Adam(model.parameters())

# Training loop
loss_vals = list()
for i in range(1000):
    loss_vals.append(0)
    for batch_img, batch_labels in train_loader:
        optimizer.zero_grad()
        pred = model(rcdt(batch_img.to(device), normalization="max"))
        # Compute loss between predicted and target labels
        loss_value = loss_fn(pred, batch_labels.to(device))
        # Backward pass
        loss_value.backward()
        # Update model parameters
        optimizer.step()
        loss_vals[i] += loss_value.item()
    loss_vals[i] /= len(train_loader)

    if i % 10 == 0:
        print(f"Iteration {i}, Loss: {loss_vals[i]:.6f}")

plt.plot(loss_vals)
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.title("Training Loss over Iterations")
plt.grid()
plt.show()

model.eval()
with torch.no_grad():
    num_correct = 0
    for batch_img, batch_labels in test_loader:
        pred = model(rcdt(batch_img.to(device), normalization="max"))
        predicted_labels = torch.argmax(pred, dim=1)
        num_correct += (predicted_labels == batch_labels.to(device)).sum().item()
    accuracy = num_correct / len(test_dataset)
    print(f"Test Accuracy: {accuracy * 100:.2f}%")

print("Classification in max-normalized R-CDT space completed!")
