import os
import random
from typing import Any, Tuple

import neps
import numpy as np
import torch
import torch.nn as nn
from torchvision.datasets import VisionDataset
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torch.optim.lr_scheduler import ReduceLROnPlateau, OneCycleLR
from automl.datasets import EmotionsDataset, FashionDataset, FlowersDataset
from automl.scheduler_wrapper import LRSchedulerWrapper


class GrayscaleToRGB:
    """Convert a grayscale image to RGB by duplicating the single channel."""

    def __call__(self, image):
        return image.convert('RGB')


class AddGaussianNoise(object):
    def __init__(self, mean=0.0, std=1.0):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


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


def train_epoch(
        config: dict[str, Any],
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        criterion: nn.Module,
        train_loader: DataLoader,
        device: torch.device = torch.device("cpu")
) -> float:
    """
    Function that trains the model for one epoch returns the mean of the training loss.
    """
    loss_per_batch = []
    model.train()

    for _, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        loss_per_batch.append(loss.item())

        # If the scheduler is oneCycleLR, step the scheduler on each batch
        if config["scheduler"] == "oneCycleLR":
            scheduler.step()

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


def get_transform(
        config: dict[str, Any] = None,
        mean: float = 0.5,
        std: float = 0.5,
        num_channels: int = None
):
    # By default, resize the image to 224x224 because of MobileNetV2
    transform_list = [
        transforms.Resize((224, 224)),
    ]

    # If the number of channels is not 3, convert the grayscale image to RGB
    if num_channels is not None and num_channels != 3:
        transform_list.append(GrayscaleToRGB())

    if config:
        # If config is given, return the transform with the specifig augmentations
        transform_list.extend([
            transforms.RandomHorizontalFlip(
                p=config["random_horizontal_flip_prob"]
            ),
            transforms.RandomVerticalFlip(
                p=config["random_vertical_flip_prob"]
            ),
            transforms.RandomApply([transforms.RandomRotation(
                config["random_rotation_deg"])],
                p=config["random_rotation_prob"]
            ),
            transforms.RandomApply(
                [
                    transforms.ColorJitter(
                        brightness=config["brightness"],
                        contrast=config["contrast"],
                        saturation=config["saturation"])
                ], p=0.5
            ),  # Assume a fixed probability for ColorJitter
        ])

    transform_list.append(transforms.ToTensor())
    transform_list.append(transforms.Normalize(mean=mean, std=std))

    if config:
        transform_list.append(transforms.RandomApply(
            [
                AddGaussianNoise(
                    config["random_gaussian_noise_mean"], config["random_gaussian_noise_std"])
            ],
            p=config["random_gaussian_noise_prob"]
        ))

    return transforms.Compose(transform_list)


def get_dataset_class(dataset: str):
    match dataset:
        case "fashion":
            return FashionDataset
        case "flowers":
            return FlowersDataset
        case "emotions":
            return EmotionsDataset
        case _:
            raise ValueError(f"Invalid dataset: {dataset}")


def get_best_config_from_results(root_directory: str, num_process: int):
    absolute_best_config = None
    absolute_best_loss = float("inf")
    absolute_best_config_id = None

    if (num_process > 1):
        # Iterate over each subdirectory
        for subdirectory_index in range(num_process):
            subdirectory_path = os.path.join(
                root_directory, f"task_PID_{subdirectory_index}")

            # Get the summary dictionary from the current subdirectory
            summary = neps.get_summary_dict(subdirectory_path)

            best_config = summary["best_config"]
            best_config_id = summary["best_config_id"]
            best_loss = summary["best_loss"]

            # Compare and update the absolute best configuration
            if best_loss < absolute_best_loss:
                absolute_best_config = best_config
                absolute_best_loss = best_loss
                absolute_best_config_id = best_config_id
    else:
        # Get the summary dictionary from the current subdirectory
        summary = neps.get_summary_dict(root_directory)

        best_config = summary["best_config"]
        best_config_id = summary["best_config_id"]
        best_loss = summary["best_loss"]

        # Compare and update the absolute best configuration
        if best_loss < absolute_best_loss:
            absolute_best_config = best_config
            absolute_best_loss = best_loss
            absolute_best_config_id = best_config_id

    # Save the absolute best config
    return absolute_best_config, absolute_best_loss, absolute_best_config_id


def get_scheduler(config, optimizer, train_loader):
    scheduler = None

    if config["scheduler"] == "reduceLROnPlateau":
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.1,
            patience=2,
            # verbose=True # deprecated, use our own wrapper to log learning rate updates
        )
    else:
        scheduler = OneCycleLR(
            optimizer,
            max_lr=1e-1,
            epochs=config["epochs"],
            steps_per_epoch=len(train_loader),
            # verbose=True # deprecated, use our own wrapper to log learning rate updates
        )

    return LRSchedulerWrapper(scheduler, verbose=True, config=config)
