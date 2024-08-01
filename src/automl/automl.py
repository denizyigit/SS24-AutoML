"""AutoML class for regression tasks.

This module contains an example AutoML class that simply returns predictions of a quickly trained MLP.
You do not need to use this setup, and you can modify this however you like.
"""
from __future__ import annotations

from typing import Any, Tuple

import torch
import random
import numpy as np
import logging

from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms

from automl.dummy_model import DummyNN
from automl.utils import calculate_mean_std


logger = logging.getLogger(__name__)


class AutoML:

    def __init__(
        self,
        seed: int,
    ) -> None:
        self.seed = seed
        self._model: nn.Module | None = None

    def fit(
        self,
        dataset_class: Any,
    ) -> AutoML:
        """A reference/toy implementation of a fitting function for the AutoML class.
        """
        # set seed for pytorch training
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)

        # Ensure deterministic behavior in CuDNN
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        self._transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(*calculate_mean_std(dataset_class)),
            ]
        )
        dataset = dataset_class(
            root="./data",
            split='train',
            download=True,
            transform=self._transform
        )
        train_loader = DataLoader(dataset, batch_size=64, shuffle=True)

        input_size = dataset_class.width * dataset_class.height * dataset_class.channels

        model = DummyNN(input_size, dataset_class.num_classes)

        # Loss function
        criterion = nn.CrossEntropyLoss()

        # Optimizer, to adjust parameters of the model
        optimizer = optim.Adam(model.parameters(), lr=0.003)

        # The model is set to training mode
        # This is important because certain layers like dropout and batch normalization behave differently during training and evaluation.
        model.train()

        for epoch in range(5):
            loss_per_batch = []

            # Iterate over the training dataset in batches
            for _, (data, target) in enumerate(train_loader):
                # The gradients of the model parameters are reset to zero before each batch to prevent accumulation from previous batches.
                optimizer.zero_grad()

                # Forward pass, input data is passed through the model to get output predictions
                output = model(data)

                # Calculate the loss using model's predictions and the target
                loss = criterion(output, target)

                # Backward pass, the gradients of the loss w.r.t. the model parameters are calculated
                loss.backward()

                # The optimizer updates the model parameters using the gradients
                optimizer.step()

                # Track the loss for the current batch
                loss_per_batch.append(loss.item())
            logger.info(f"Epoch {epoch + 1}, Loss: {np.mean(loss_per_batch)}")


        model.eval()
        self._model = model

        return self

    def predict(self, dataset_class) -> Tuple[np.ndarray, np.ndarray]:
        """A reference/toy implementation of a prediction function for the AutoML class.
            Returns predicted labels and target labels.
        """
        dataset = dataset_class(
            root="./data",
            split='test',
            download=True,
            transform=self._transform
        )
        data_loader = DataLoader(dataset, batch_size=100, shuffle=False)

        # Store the predicted labels and true labels in lists
        predictions = []
        labels = []

        # The model is set to evaluation mode using model.eval().
        # This ensures that layers like dropout and batch normalization behave appropriately during inference.
        self._model.eval()

        # Gradient calculation is disabled using torch.no_grad()
        # to improve performance and reduce memory usage during inference.
        with torch.no_grad():
            for data, target in data_loader:
                output = self._model(data)
                predicted = torch.argmax(output, 1)
                labels.append(target.numpy())
                predictions.append(predicted.numpy())
        predictions = np.concatenate(predictions)
        labels = np.concatenate(labels)

        return predictions, labels
