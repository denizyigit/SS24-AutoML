"""AutoML class for regression tasks.

This module contains an example AutoML class that simply returns predictions of a quickly trained MLP.
You do not need to use this setup, and you can modify this however you like.
"""
from __future__ import annotations

from typing import Any, Tuple

import neps
import neps.optimizers
import neps.optimizers.multi_fidelity
import neps.optimizers.multi_fidelity.hyperband
import neps.optimizers.multi_fidelity.sampling_policy
import torch
import random
import numpy as np
import logging

from torchvision.datasets import VisionDataset
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms
from neps.plot.tensorboard_eval import tblogger

from src.automl.dummy_model import *
from src.automl.utils import calculate_mean_std, create_reduced_dataset, evaluate_validation_epoch, get_optimizer, get_transform, train_epoch
from src.automl.pipeline_space import pipeline_space
import time

logger = logging.getLogger(__name__)


class AutoML:

    def __init__(self, seed: int, dataset_class: VisionDataset, reduced_dataset_ratio: float = 1.0, max_evaluations_total=10) -> None:
        self.seed = seed

        self.model: nn.Module | None = None
        self.best_config = None
        self.best_model_loss = None

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

        # Save the dataset class for later use
        # Use reduced_dataset_ratio to reduce the dataset size for faster training
        self.dataset_class = dataset_class
        self.reduced_dataset_ratio = float(reduced_dataset_ratio)
        # self.max_evaluations_total = max_evaluations_total

        self.mean_train = None
        self.std_train = None

    def fit(self) -> AutoML:
        # Root directory for neps
        root_directory = f"results_{self.dataset_class.__name__}"

        print(f"Cuda available: {torch.cuda.is_available()}")
        print(
            f"Training on dataset: {self.dataset_class.__name__}, type: {self.dataset_class.__class__})")
        print(
            f"Reduced dataset ratio: {self.reduced_dataset_ratio} is used for faster training.")

        # Calculate mean and std of training dataset for normalization transforms, both for training and testing
        # Then assign them to the class variables
        dataset_train = self.dataset_class(
            root="./data",
            split='train',
            download=True,
            transform=transforms.ToTensor(),
        )
        self.mean_train, self.std_train = calculate_mean_std(dataset_train)

        # Define the target function for neps
        def target_function(**config):
            print("\n------------------")
            print("Evaluation with config:")
            print(config)

            # Calculate how much time is spent on the evaluation
            # This is useful for multi-fidelity optimization
            start_time = time.time()

            random.seed(self.seed)
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)
            torch.cuda.manual_seed(self.seed)
            # Ensure deterministic behavior in CuDNN
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

            # Configure the transform for the dataset
            # TODO:
            # 1. Implement a more sophisticated transform with respect to the pipeline_space (e.g. data augmentation, normalization, etc.)
            # 2. Apply the same transform composition in the final training as well (using self.best_config this time)
            # 3. birden fazla architecture ismi ve parameteresi ( architecutre'ın width ve height size'ı gibi) eklenmeli, ona göre resize edilmeli
            transform = get_transform(
                config=config,
                mean=self.mean_train,
                std=self.std_train,
                num_channels=self.dataset_class.channels
            )

            # Get train dataset with the defined transform
            dataset_train = self.dataset_class(
                root="./data",
                split='train',
                download=True,
                transform=transform
            )

            # Reduce dataset size if needed for faster training
            if self.reduced_dataset_ratio < 1.0:
                dataset_train = create_reduced_dataset(
                    dataset_train, ratio=self.reduced_dataset_ratio)

            # Split train dataset into train and validation datasets
            dataset_train, dataset_val = torch.utils.data.random_split(
                dataset_train,
                [0.8, 0.2]
            )

            # Create data loaders for train and validation datasets
            # TODO: Use batch_size from config
            train_loader = DataLoader(
                dataset_train, batch_size=64, shuffle=True)

            # TODO: Use batch_size from config
            validation_loader = DataLoader(
                dataset_val, batch_size=64)

            # TODO: Unused variable
            input_size = self.dataset_class.width * \
                self.dataset_class.height * self.dataset_class.channels

            # Create a CNN model
            # TODO: Implement a more sophisticated acrhitecture selection with respect to the pipeline_space (for the sake of NAS)
            # See https://github.com/automl/neps/blob/master/neps_examples/basic_usage/architecture_and_hyperparameters.py

            model = DummyCNN(
                input_channels=self.dataset_class.channels,
                hidden_channels=30,
                output_channels=self.dataset_class.num_classes,
                image_width=self.dataset_class.width
            )

            # model = VGG16(
            #     input_channels=self.dataset_class.channels,
            #     output_channels=self.dataset_class.num_classes,
            #     mean=self.mean_train,
            #     std=self.std_train,
            # )

            model.to(self.device)

            criterion = nn.CrossEntropyLoss()

            # Get optimizer based on the config
            optimizer = get_optimizer(config, model)

            # Train the model, calculate the training loss
            best_epoch_loss = None
            for epoch in range(config["epochs"]):
                training_loss = train_epoch(
                    optimizer=optimizer,
                    model=model,
                    criterion=criterion,
                    train_loader=train_loader,
                    device=self.device
                )
                logger.info(
                    f"Epoch {epoch + 1}, Training Loss: {training_loss}")

                # Evaluate the model on the validation set
                validation_loss, incorrect_images = evaluate_validation_epoch(
                    model=model,
                    criterion=criterion,
                    validation_loader=validation_loader,
                    device=self.device
                )

                logger.info(
                    f"Epoch {epoch + 1}, Validation Loss: {validation_loss}")

                # Save the validation_loss to best_epoch_loss
                # If and only if it is the least loss encountered so far
                if best_epoch_loss is None or validation_loss < best_epoch_loss:
                    best_epoch_loss = validation_loss

                # If best_epoch_loss is the least loss encountered so far
                # Save the model as the best model to be used for predictions
                if self.best_model_loss is None or best_epoch_loss < self.best_model_loss:
                    self.best_model_loss = best_epoch_loss
                    self.model = model  # 👈👈👈

                ###################### Start Tensorboard Logging ######################

                # The following tblogge` will result in:

                # 1. Loss curves of each configuration at each epoch.
                # 2. Decay curve of the learning rate at each epoch.
                # 3. Wrongly classified images by the model.
                # 4. First two layer gradients passed as scalar configs.
                tblogger.log(
                    loss=validation_loss,
                    current_epoch=epoch,
                    # Set to `True` for a live loss trajectory for each config.
                    write_summary_incumbent=True,
                    writer_config_scalar=True,
                    writer_config_hparam=True,

                    extra_data={
                        "miss_img": tblogger.image_logging(image=incorrect_images, counter=2, seed=self.seed),
                    },
                )
                ###################### End Tensorboard Logging ######################

            end_time = time.time()

            print(
                f"\nBest epoch loss: {best_epoch_loss}, Time: {end_time - start_time}\n")

            return {"loss": best_epoch_loss, "cost": end_time - start_time}

        # Run optimization pipeline with NEPS and save results to root_directory
        neps.run(
            run_pipeline=target_function,
            pipeline_space=pipeline_space,
            root_directory=root_directory,
            max_evaluations_total=3,
            overwrite_working_directory=True,
            post_run_summary=True,
            searcher={
                "strategy": "priorband",
                "eta": 3,
                "initial_design_type": "max_budget",
            },
            # Total cost, we use the time spent on evaluation as cost (seconds)
            max_cost_total=1000,
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

        return self

    def predict(self) -> Tuple[np.ndarray, np.ndarray]:
        # Configure the transform for the test dataset, without data augmentation
        transform = get_transform(
            mean=self.mean_train,
            std=self.std_train,
            num_channels=self.dataset_class.channels
        )

        dataset_test = self.dataset_class(
            root="./data",
            split='test',
            download=True,
            transform=transform,
        )

        # Reduce dataset size if needed for faster training
        if self.reduced_dataset_ratio < 1.0:
            dataset_test = create_reduced_dataset(
                dataset_test, ratio=self.reduced_dataset_ratio)

        data_loader = DataLoader(
            dataset_test, batch_size=100, shuffle=False)
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
