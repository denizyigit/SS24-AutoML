import random
from typing import Any, Tuple

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


# Define the training step. Return the validation error and
# misclassified images.
def evaluate_loss(model: nn.Module, data_loader: DataLoader) -> float:
    # Set the model in evaluation mode (no gradient computation).
    model.eval()

    correct = 0
    total = 0

    # Disable gradient computation for efficiency.
    with torch.no_grad():
        for x, y in data_loader:
            output = model(x)

            # Get the predicted class for each input.
            _, predicted = torch.max(output.data, 1)

            # Update the correct and total counts.
            correct += (predicted == y).sum().item()
            total += y.size(0)

    # Calculate the accuracy and return the error rate.
    accuracy = correct / total
    error_rate = 1 - accuracy
    return error_rate


def training(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    train_loader: DataLoader,
    validation_loader: DataLoader,
) -> Tuple[float, torch.Tensor]:
    """
    Function that trains the model for one epoch and evaluates the model
    on the validation set.

    Args:
        model (nn.Module): Model to be trained.
        optimizer (torch.optim.Optimizer): Optimizer used to train the weights.
        criterion (nn.Module) : Loss function to use.
        train_loader (DataLoader): DataLoader containing the training data.
        validation_loader (DataLoader): DataLoader containing the validation data.

    Returns:
    Tuple[float, torch.Tensor]: A tuple containing the validation error (float)
                                and a tensor of misclassified images.
    """
    incorrect_images = []
    model.train()

    for x, y in train_loader:
        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()

        predicted_labels = torch.argmax(output, dim=1)
        incorrect_mask = predicted_labels != y
        incorrect_images.append(x[incorrect_mask])

    # Calculate validation loss using the loss_ev function.
    validation_loss = evaluate_loss(model, validation_loader)

    # Return the misclassified image by during model training.
    if len(incorrect_images) > 0:
        incorrect_images = torch.cat(incorrect_images, dim=0)

    return (validation_loss, incorrect_images)
