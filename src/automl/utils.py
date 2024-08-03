import random
from typing import Any, Tuple

import numpy as np
import torch
import torch.nn as nn
from torchvision.datasets import VisionDataset
from torch.utils.data import DataLoader, Subset
from torchvision import transforms


def print_train_time(start: float,
                     end: float,
                     device: torch.device = None):
    total_time = end - start
    print(f"Train time on  {device} : {total_time:.3f} seconds.")


def create_reduced_dataset(dataset: VisionDataset, ratio=0.1) -> VisionDataset:
    """Create a reduced subset of the dataset."""
    dataset_size = len(dataset)
    subset_size = int(dataset_size * ratio)
    indices = random.sample(range(dataset_size), subset_size)
    return Subset(dataset, indices)


def calculate_mean_std(dataset: VisionDataset):
    """Calculate the mean and standard deviation of a subset of the image dataset."""
    mean = 0.
    std = 0.
    total_images_count = 0

    loader = DataLoader(dataset, batch_size=64, shuffle=False)

    for images, _ in loader:
        images.to(torch.device("cpu") if not torch.cuda.is_available()
                  else torch.device("cuda"))
        batch_samples = images.size(0)
        images = images.view(batch_samples, images.size(1), -1)
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)
        total_images_count += batch_samples

    mean /= total_images_count
    std /= total_images_count

    return mean, std


def get_optimizer(config, model):
    lr = config["learning_rate"]
    momentum = config["momentum"]
    weight_decay = config["weight_dec_adam"] if config["optimizer"] == "adam" else config["weight_dec_sgd"]

    if config['optimizer'] == 'Adam':
        optimizer = torch.optim.Adam(
            params=model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
    else:
        optimizer = torch.optim.SGD(
            params=model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            momentum=momentum
        )

    return optimizer


def evaluate_loss(
        model: nn.Module,
        data_loader: DataLoader,
        device: torch.device = torch.device("cpu")
) -> float:
    # Set the model in evaluation mode (no gradient computation).
    model.eval()

    correct = 0
    total = 0

    # Disable gradient computation for efficiency.
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(data_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)

            # Get the predicted class for each input.
            a, predicted = torch.max(output.data, 1)
            print("_:", a)
            print("predicted:", predicted)
            print("y:", target)
            # Update the correct and total counts.
            correct += (predicted == target).sum().item()
            total += target.size(0)

    # Calculate the accuracy and return the error rate.
    accuracy = correct / total
    error_rate = 1 - accuracy
    return error_rate


def train_epoch(
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        train_loader: DataLoader,
        device: torch.device = torch.device("cpu")
) -> float:
    """
    Function that trains the model for one epoch returns the mean of the training loss.
    """
    loss_per_batch = []
    model.train()
    train_loader.dataset.transform = transforms.Compose([
        transforms.Resize((224, 224)), transforms.ToTensor()])
    for _, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        loss_per_batch.append(loss.item())

    mean_loss = np.mean(loss_per_batch)

    return mean_loss


def evaluate_validation_epoch(
        model: nn.Module,
        criterion: nn.Module,
        validation_loader: DataLoader,
        device: torch.device = torch.device("cpu")
) -> Tuple[float, Any]:
    """
    Function that trains the model for one epoch returns the mean of the training loss.
    """
    model.eval()

    incorrect_images = []

    with torch.no_grad():
        validation_loss_per_batch = []
        for _, (data, target) in enumerate(validation_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            validation_loss_per_batch.append(loss.item())

            # Get the predicted class for each input.
            predicted_labels = torch.argmax(output, dim=1)
            # Update the correct and total counts.
            incorrect_mask = predicted_labels != target
            incorrect_images.append(data[incorrect_mask])

        if len(incorrect_images) > 0:
            incorrect_images = torch.cat(incorrect_images, dim=0)

        validation_loss = np.mean(validation_loss_per_batch)

    return validation_loss, incorrect_images
