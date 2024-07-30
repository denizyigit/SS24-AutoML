import random
from typing import Any

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import transforms


def print_train_time(start: float,
                     end: float,
                     device: torch.device = None):
    total_time = end - start
    print(f"Train time on  {device} : {total_time:.3f} seconds.")


def create_reduced_dataset(dataset, ratio=0.1):
    """Create a reduced subset of the dataset."""
    dataset_size = len(dataset)
    subset_size = int(dataset_size * ratio)
    indices = random.sample(range(dataset_size), subset_size)
    return Subset(dataset, indices)


def calculate_mean_std(dataset_class: Any, ratio: float = 0.1):
    """Calculate the mean and standard deviation of a subset of the image dataset."""
    mean = 0.
    std = 0.
    total_images_count = 0

    full_dataset = dataset_class(
        root="./data",
        split='train',
        download=True,
        transform=transforms.ToTensor()
    )

    # Create a reduced dataset
    reduced_dataset = create_reduced_dataset(full_dataset, ratio)

    loader = DataLoader(reduced_dataset, batch_size=64, shuffle=False)

    for images, _ in loader:
        images.to(torch.device("cpu") if not torch.cuda.is_available() else torch.device("cuda"))
        batch_samples = images.size(0)
        images = images.view(batch_samples, images.size(1), -1)
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)
        total_images_count += batch_samples

    mean /= total_images_count
    std /= total_images_count

    return mean, std
