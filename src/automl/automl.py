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

from src.automl.dummy_model import DummyNN, resNet18_Pretrained
from src.automl.utils import calculate_mean_std, create_reduced_dataset, get_optimizer

logger = logging.getLogger(__name__)


class AutoML:

    def __init__(self, seed: int, reduced_dataset_ratio: float = 1.0) -> None:
        self.seed = seed
        self.model: nn.Module | None = None
        self.best_config = None
        self.reduced_dataset_ratio = float(reduced_dataset_ratio)
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

        self.dataset_class_mean = None
        self.dataset_class_std = None

    def fit(self, dataset_class: Any, ) -> AutoML:
        # TODO: Move pipeline_space to a separate file
        pipeline_space = dict(
            batch_size=neps.IntegerParameter(lower=1, upper=100, log=True),
            learning_rate=neps.FloatParameter(
                lower=1e-6, upper=1e-1, log=True),
            epochs=neps.IntegerParameter(lower=1, upper=3),

            optimizer=neps.CategoricalParameter(["adam", "sgd"]),

            momentum=neps.FloatParameter(lower=0.1, upper=0.999, default=0.4),
            weight_dec_active=neps.CategoricalParameter(
                ["True", "False"], default="False"),
            weight_dec_adam=neps.FloatParameter(
                lower=0.00001, upper=0.1, default=1e-4),
            weight_dec_sgd=neps.FloatParameter(
                lower=0.00001, upper=0.1, default=1e-4),

            # image resizing, to be used in the transform
            resize=neps.IntegerParameter(lower=224, upper=256),
            # random rotation, to be used in the transform
            rotation=neps.FloatParameter(lower=0.0, upper=30.0),
            # random horizontal flip, to be used in the transform
            horizontal_flip=neps.CategoricalParameter(choices=[True, False])
        )

        # Root directory for neps
        root_directory = f"results_{dataset_class.__name__}"

        # Calculate mean and std of dataset for normalization
        mean, std = calculate_mean_std(
            dataset_class, ratio=self.reduced_dataset_ratio)
        self.dataset_class_mean = mean
        self.dataset_class_std = std

        def target_function(**config):
            random.seed(self.seed)
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)
            torch.cuda.manual_seed(self.seed)
            # Ensure deterministic behavior in CuDNN
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

            print(f"Cuda available: {torch.cuda.is_available()}")
            print(
                f"Training on dataset: {dataset_class.__name__}, type: {dataset_class.__class__})")
            print(
                f"Reduced dataset ratio: {self.reduced_dataset_ratio} is used for faster training.")

            # TODO:
            # 1. Implement a more sophisticated transform with respect to the pipeline_space (e.g. data augmentation, normalization, etc.)
            # 2. Apply the same transform composition in the final training as well (using self.best_config this time)
            transform = transforms.Compose(
                [
                    # transforms.Resize(resize),
                    # transforms.RandomRotation(rotation),
                    # transforms.RandomHorizontalFlip() if horizontal_flip else transforms.Lambda(lambda x: x),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=mean, std=std),
                ]
            )

            full_dataset = dataset_class(
                root="./data",
                split='train',
                download=True,
                transform=transform
            )

            # TODO: Remove this line
            # Reduce dataset size for faster training
            reduced_dataset = create_reduced_dataset(
                full_dataset, self.reduced_dataset_ratio)

            # TODO: Use batch_size from config
            train_loader = DataLoader(
                reduced_dataset, batch_size=64, shuffle=True)

            # TODO: Unused variable
            input_size = dataset_class.width * dataset_class.height * dataset_class.channels

            # TODO: Implement a more sophisticated acrhitecture selection with respect to the pipeline_space (for the sake of NAS)
            model = resNet18_Pretrained(num_classes=dataset_class.num_classes,
                                        number_of_channels=dataset_class.channels)
            model.to(self.device)

            criterion = nn.CrossEntropyLoss()

            optimizer = get_optimizer(config, model)

            # Train the model, calculate the loss
            model.train()

            for epoch in range(config["epochs"]):
                loss_per_batch = []
                for _, (data, target) in enumerate(train_loader):
                    data, target = data.to(self.device), target.to(self.device)
                    optimizer.zero_grad()
                    output = model(data)
                    loss = criterion(output, target)
                    loss.backward()
                    optimizer.step()
                    loss_per_batch.append(loss.item())
                mean_loss = np.mean(loss_per_batch)
                logger.info(f"Epoch {epoch + 1}, Loss: {mean_loss}")
            model.eval()

            return mean_loss

        # Run optimization pipeline with NEPS and save results to root_directory
        neps.run(
            run_pipeline=target_function,
            pipeline_space=pipeline_space,
            root_directory=root_directory,
            max_evaluations_total=3,
            overwrite_working_directory=True,
            seed=self.seed,
            post_run_summary=True,
        )

        # ------------------ GET THE BEST CONFIG ------------------
        summary = neps.get_summary_dict(root_directory)
        best_config = summary["best_config"]
        best_config_id = summary["best_config_id"]
        best_loss = summary["best_loss"]

        # Save best config
        self.best_config = best_config

        print("\n\tBEST CONFIG:")
        print(
            f"\t\t{best_config}\n\t\tLoss: {best_loss}\n\t\tConfig ID: {best_config_id}\n")

        # ------------------ TRAIN A FINAL MODEL WITH THE BEST CONFIG ------------------
        print("\nFinal training with the best config...\n")
        # TODO:
        # 1. Implement a more sophisticated transform with respect to the pipeline_space (e.g. data augmentation, normalization, etc.)
        # 2. Apply the same transform composition we used in target_function (using self.best_config this time)
        transform = transforms.Compose(
            [
                # transforms.Resize(resize),
                # transforms.RandomRotation(rotation),
                # transforms.RandomHorizontalFlip() if horizontal_flip else transforms.Lambda(lambda x: x),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        )

        full_dataset = dataset_class(
            root="./data",
            split='train',
            download=True,
            transform=transform
        )

        # TODO: Remove this line
        # Reduce dataset size for faster training
        reduced_dataset = create_reduced_dataset(
            full_dataset, self.reduced_dataset_ratio)

        # TODO: Use batch_size from self.best_config
        train_loader = DataLoader(reduced_dataset, batch_size=64, shuffle=True)

        model = resNet18_Pretrained(num_classes=dataset_class.num_classes,
                                    number_of_channels=dataset_class.channels)
        model.to(self.device)

        criterion = nn.CrossEntropyLoss()

        optimizer = get_optimizer(self.best_config, model)

        model.train()

        for epoch in range(best_config["epochs"]):
            loss_per_batch = []
            for _, (data, target) in enumerate(train_loader):
                data, target = data.to(self.device), target.to(self.device)
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                loss_per_batch.append(loss.item())
            mean_loss = np.mean(loss_per_batch)
            logger.info(f"Epoch {epoch + 1}, Loss: {mean_loss}")
        model.eval()

        # Save the final model
        self.model = model

        return self

    def predict(self, dataset_class, reduced_dataset_ratio: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
        mean, std = calculate_mean_std(
            dataset_class, ratio=reduced_dataset_ratio)

        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        )
        dataset = dataset_class(
            root="./data",
            split='test',
            download=True,
            transform=transform
        )
        data_loader = DataLoader(dataset, batch_size=100, shuffle=False)
        predictions = []
        labels = []
        self.model.eval()
        with torch.no_grad():
            for data, target in data_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                predicted = torch.argmax(output, 1)
                labels.append(target.to("cpu").numpy())
                predictions.append(predicted.to("cpu").numpy())
        predictions = np.concatenate(predictions)
        labels = np.concatenate(labels)

        return predictions, labels
