"""AutoML class for regression tasks.

This module contains an example AutoML class that simply returns predictions of a quickly trained MLP.
You do not need to use this setup, and you can modify this however you like.
"""
from __future__ import annotations

from typing import Any, Tuple

import neps
import torch
import random
import numpy as np
import logging

from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms
from neps.plot.tensorboard_eval import tblogger

from src.automl.dummy_model import *
from src.automl.utils import calculate_mean_std, create_reduced_dataset

logger = logging.getLogger(__name__)


class AutoML:

    def __init__(self, seed: int, reduced_dataset_ratio: float = 1.0) -> None:
        self.seed = seed
        self._model: nn.Module | None = None
        self._transform = None
        self.reduced_dataset_ratio = float(reduced_dataset_ratio)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def fit(self, dataset_class: Any, ) -> AutoML:
        """A reference/toy implementation of a fitting function for the AutoML class.
        """
        pipeline_space = dict(
            batch_size=neps.IntegerParameter(lower=1, upper=100, log=True),
            learning_rate=neps.FloatParameter(lower=1e-6, upper=1e-1, log=True),
            epochs=neps.IntegerParameter(lower=1, upper=20),
            optimizer=neps.CategoricalParameter(["adam", "sgd"]),
        )

        root_directory = f"results_{dataset_class.__name__}"

        def run_pipeline(**config):
            random.seed(self.seed)
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)
            torch.cuda.manual_seed(self.seed)
            # Ensure deterministic behavior in CuDNN
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

            print(f"Cuda available: {torch.cuda.is_available()}")
            print(f"Training on dataset: {dataset_class.__name__}, type: {dataset_class.__class__})")
            print(f"Reduced dataset ratio: {self.reduced_dataset_ratio} is used for faster training.")

            mean, std = calculate_mean_std(dataset_class, ratio=self.reduced_dataset_ratio)

            self._transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize(mean=mean, std=std),
                ]
            )

            full_dataset = dataset_class(
                root="./data",
                split='train',
                download=True,
                transform=self._transform
            )

            reduced_dataset = create_reduced_dataset(full_dataset, self.reduced_dataset_ratio)

            train_loader = DataLoader(reduced_dataset, batch_size=64, shuffle=True)

            input_size = dataset_class.width * dataset_class.height * dataset_class.channels

            model = DummyCNN(input_channels=dataset_class.channels,hidden_channels=30, output_channels=dataset_class.num_classes,image_width=dataset_class.width)
            print(model)
            model.to(self.device)

            criterion = nn.CrossEntropyLoss()

            if config["optimizer"] == "adam":
                optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])
            else:
                optimizer = optim.SGD(model.parameters(), lr=config["learning_rate"])

            model.train()

            for epoch in range(config["epochs"]):
                loss_per_batch = []
                for a, (data, target) in enumerate(train_loader):
                    data, target = data.to(self.device), target.to(self.device)
                    optimizer.zero_grad()
                    output = model(data)
                    loss = criterion(output, target)
                    loss.backward()
                    optimizer.step()
                    loss_per_batch.append(loss.item())


                mean_loss = np.mean(loss_per_batch)

                tblogger.log(
                    loss=mean_loss,
                    current_epoch=epoch,
                    write_summary_incumbent=True,  # Set to `True` for a live incumbent trajectory.
                    writer_config_scalar=True,  # Set to `True` for a live loss trajectory for each config.
                    writer_config_hparam=True,
                    # Set to `True` for live parallel coordinate, scatter plot matrix, and table view.
                    # Appending extra data

                )

                logger.info(f"Epoch {epoch + 1}, Loss: {mean_loss}")
            model.eval()
            self._model = model

            return mean_loss


        neps.run(
            run_pipeline=run_pipeline,
            pipeline_space=pipeline_space,
            root_directory=root_directory,
            max_evaluations_total=2,
            overwrite_working_directory=True,
            seed=self.seed,
            post_run_summary=True,
        )

        previous_results, pending_configs = neps.status(root_directory=root_directory)

        # Find the best configuration
        best_config = min(previous_results, key=lambda x: x["loss"])

        print(best_config)



    def predict(self, dataset_class) -> Tuple[np.ndarray, np.ndarray]:
        """A reference/toy implementation of a prediction function for the AutoML class.
        """
        dataset = dataset_class(
            root="./data",
            split='test',
            download=True,
            transform=self._transform
        )
        data_loader = DataLoader(dataset, batch_size=100, shuffle=False)
        predictions = []
        labels = []
        self._model.eval()
        with torch.no_grad():
            for data, target in data_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self._model(data)
                predicted = torch.argmax(output, 1)
                labels.append(target.to("cpu").numpy())
                predictions.append(predicted.to("cpu").numpy())
        predictions = np.concatenate(predictions)
        labels = np.concatenate(labels)

        return predictions, labels
