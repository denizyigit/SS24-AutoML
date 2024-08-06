"""AutoML class for regression tasks.

This module contains an example AutoML class that simply returns predictions of a quickly trained MLP.
You do not need to use this setup, and you can modify this however you like.
"""
from __future__ import annotations

import multiprocessing
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

from automl.datasets import EmotionsDataset, FashionDataset, FlowersDataset
from automl.early_stopping import EarlyStopping
from src.automl.dummy_model import *
from src.automl.utils import calculate_mean_std, create_reduced_dataset, evaluate_validation_epoch, get_best_config_from_results, get_dataset_class, get_optimizer, get_scheduler, get_transform, train_epoch
from src.automl.pipeline_space import PipelineSpace
import time

logger = logging.getLogger(__name__)


# Define the target function for neps
def target_function(**config):
    # Calculate how much time is spent on the evaluation
    # This is useful for multi-fidelity optimization
    start_time = time.time()

    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    torch.cuda.manual_seed(config["seed"])
    # Ensure deterministic behavior in CuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")

    # *********************** PREPARE DATASETS ***********************
    dataset_class = get_dataset_class(config["dataset"])

    # Calculate mean and std of training dataset for normalization transforms, both for training and testing
    # Then assign them to the class variables
    dataset_train = dataset_class(
        root="./data",
        split='train',
        download=True,
        transform=transforms.ToTensor(),
    )

    # Reduce dataset size if needed for faster training
    if config["reduced_dataset_ratio"] < 1.0:
        dataset_train = create_reduced_dataset(
            dataset_train, ratio=config["reduced_dataset_ratio"])

    # Calculate mean and std of the training dataset
    mean_train, std_train = calculate_mean_std(dataset_train)

    # Split train dataset into train and validation datasets
    dataset_train, dataset_val = torch.utils.data.random_split(
        dataset_train,
        [0.8, 0.2]
    )

    # Configure the transform for the dataset
    # TODO:
    # 1. Implement a more sophisticated transform with respect to the pipeline_space (e.g. data augmentation, normalization, etc.)
    # 2. Apply the same transform composition in the final training as well (using self.best_config this time)
    # 3. birden fazla architecture ismi ve parameteresi ( architecutre'ın width ve height size'ı gibi) eklenmeli, ona göre resize edilmeli
    transform = get_transform(
        config=config,
        mean=mean_train,
        std=std_train,
        num_channels=dataset_class.channels
    )

    # Get train dataset with the defined transform
    dataset_train = dataset_class(
        root="./data",
        split='train',
        download=True,
        transform=transform
    )

    # Reduce dataset size if needed for faster training
    if config["reduced_dataset_ratio"] < 1.0:
        dataset_train = create_reduced_dataset(
            dataset_train, ratio=config["reduced_dataset_ratio"])

    # Create data loaders for train and validation datasets
    # TODO: Use batch_size from config
    train_loader = DataLoader(
        dataset_train, batch_size=int(config["batch_size"]), shuffle=True)

    # TODO: Use batch_size from config
    validation_loader = DataLoader(
        dataset_val, batch_size=int(config["batch_size"]))

    # TODO: Unused variable
    # input_size = self.dataset_class.width * \
    #     self.dataset_class.height * self.dataset_class.channels

    # Create a CNN model
    # TODO: Implement a more sophisticated acrhitecture selection with respect to the pipeline_space (for the sake of NAS)
    # See https://github.com/automl/neps/blob/master/neps_examples/basic_usage/architecture_and_hyperparameters.py

    # model = MobileNet(output_channels=dataset_class.num_classes)
    model = DummyCNN(
        input_channels=dataset_class.channels,
        hidden_channels=30,
        output_channels=dataset_class.num_classes,
        image_width=dataset_class.width
    )
    model.to(device)

    criterion = nn.CrossEntropyLoss()

    # Get optimizer based on the config
    optimizer = get_optimizer(config, model)

    # LR scheduler
    scheduler = get_scheduler(config, optimizer, train_loader)

    early_stopping = EarlyStopping(
        patience=7,
        delta=0,
        verbose=True,
        pid=config["pid"]
    )

    # Train the model, calculate the training loss
    best_validation_loss = None

    for epoch in range(config["epochs"]):
        training_loss = train_epoch(
            config=config,
            optimizer=optimizer,
            scheduler=scheduler,
            model=model,
            criterion=criterion,
            train_loader=train_loader,
            device=device
        )

        # Evaluate the model on the validation set
        validation_loss, incorrect_images = evaluate_validation_epoch(
            model=model,
            criterion=criterion,
            validation_loader=validation_loader,
            device=device
        )

        print(
            f"PID_{config['pid']}: Epoch {epoch + 1}, Training Loss: {training_loss}, Validation Loss: {validation_loss}")

        # Check if early stopping is needed
        early_stopping(validation_loss)

        if early_stopping.early_stop:
            break

        # If the scheduler is reduceLROnPlateau, step the scheduler on each epoch
        if config["scheduler"] == "reduceLROnPlateau":
            scheduler.step(validation_loss)

        # Update the best_validation_loss of current config
        # In the end, we will return best_validation_loss as the loss of the current config
        if (best_validation_loss is None or validation_loss < best_validation_loss):
            best_validation_loss = validation_loss

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
                "miss_img": tblogger.image_logging(image=incorrect_images, counter=1, seed=config["seed"]),
                "lr_decay": tblogger.scalar_logging(value=scheduler.get_last_lr()[0])
            },
        )
        ###################### End Tensorboard Logging ######################

    end_time = time.time()

    print(
        f"\nPID_{config['pid']}: Validation loss: {best_validation_loss}, Time spent: {end_time - start_time} \n")

    return {"loss": best_validation_loss, "cost": end_time - start_time}

# Function to be targeted by multiple processors
# Each processor will run the optimization pipeline with NEPS in parallel


def neps_run_pipeline(pid: int, seed: int, dataset: str, reduced_dataset_ratio: float, root_directory: str):
    # Get the pipeline space for the optimization
    pipeline_space = PipelineSpace().get_pipeline_space(
        pid=pid,
        seed=seed,
        dataset=dataset,
        reduced_dataset_ratio=reduced_dataset_ratio,
    )

    # Run optimization pipeline with NEPS and save results to root_directory
    neps.run(
        run_pipeline=target_function,
        pipeline_space=pipeline_space,
        root_directory=root_directory,
        max_evaluations_total=20,
        overwrite_working_directory=False,
        post_run_summary=True,
        searcher={
            "strategy": "priorband",
            "eta": 3,
        },
        # Total cost, we use the time spent on evaluation as cost (seconds)
        max_cost_total=1000,
        task_id=f'PID_{pid}' if pid != -1 else None,
    )


def neps_run_pipeline_multiprocessor(num_process: int, seed: int, dataset: str, reduced_dataset_ratio: float, root_directory: str):
    if num_process > 1:
        # Create multiple processes to run the optimization pipeline in parallel
        processes = []
        for i in range(num_process):
            p = multiprocessing.Process(
                target=neps_run_pipeline,
                args=(
                    i,
                    seed,
                    dataset,
                    reduced_dataset_ratio,
                    root_directory,
                )
            )
            p.start()
            processes.append(p)

        # Wait for all processes to finish
        for p in processes:
            p.join()
    else:
        neps_run_pipeline(
            -1,  # pid=-1 means that no task_id will be used in neps.run
            seed,
            dataset,
            reduced_dataset_ratio,
            root_directory
        )


class AutoML:

    def __init__(self, seed: int, dataset: str, reduced_dataset_ratio: float = 1.0, max_evaluations_total=10) -> None:
        self.seed = seed

        self.best_model: nn.Module | None = None
        self.best_config = None

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

        # Save the dataset class for later use
        # Use reduced_dataset_ratio to reduce the dataset size for faster training
        self.dataset = dataset
        self.reduced_dataset_ratio = float(reduced_dataset_ratio)
        # self.max_evaluations_total = max_evaluations_total

        self.mean_train = None
        self.std_train = None

        # Number of processes to run the optimization pipeline in parallel, default 1 means no parallelization
        self.num_process = 1

    def fit(self) -> AutoML:
        # Root directory for neps
        root_directory = f"results_{self.dataset}"

        print(f"Cuda available: {torch.cuda.is_available()}")
        print(f"Training on dataset: {self.dataset}")
        print(
            f"Reduced dataset ratio: {self.reduced_dataset_ratio} is used for faster training.")

        neps_run_pipeline_multiprocessor(
            self.num_process,
            self.seed,
            self.dataset,
            self.reduced_dataset_ratio,
            root_directory
        )

        # ------------------ GET THE BEST CONFIG FROM RESULTS ------------------
        best_config, best_loss, best_config_id = get_best_config_from_results(
            root_directory,
            self.num_process
        )

        # Save the best config for later use
        self.best_config = best_config

        print("\n\tBEST CONFIG:")
        print(
            f"\t\t{best_config}\n\t\tLoss: {best_loss}\n\t\tConfig ID: {best_config_id}\n")

        # ------------------ GET A FINAL MODEL WITH THE BEST CONFIG ------------------
        print("\n\tTRAINING A FINAL MODEL WITH THE BEST CONFIG\n")
        dataset_class = get_dataset_class(self.dataset)

        dataset_train = dataset_class(
            root="./data",
            split='train',
            download=True,
            transform=transforms.ToTensor(),
        )

        # Reduce dataset size if needed for faster training
        if self.best_config["reduced_dataset_ratio"] < 1.0:
            dataset_train = create_reduced_dataset(
                dataset_train, ratio=self.best_config["reduced_dataset_ratio"])

        # Calculate mean and std of the training dataset
        self.mean_train, self.std_train = calculate_mean_std(dataset_train)

        # Configure the transform for the dataset
        transform = get_transform(
            config=self.best_config,
            mean=self.mean_train,
            std=self.std_train,
            num_channels=dataset_class.channels
        )

        # Get train dataset with the defined transform
        dataset_train = dataset_class(
            root="./data",
            split='train',
            download=True,
            transform=transform
        )

        if self.reduced_dataset_ratio < 1.0:
            dataset_train = create_reduced_dataset(
                dataset_train, ratio=self.reduced_dataset_ratio)

        train_loader = DataLoader(
            dataset_train, batch_size=int(self.best_config["batch_size"]), shuffle=True)

        # model = MobileNet(output_channels=dataset_class.num_classes)
        model = DummyCNN(
            input_channels=dataset_class.channels,
            hidden_channels=30,
            output_channels=dataset_class.num_classes,
            image_width=dataset_class.width
        )
        model.to(self.device)

        criterion = nn.CrossEntropyLoss()

        optimizer = get_optimizer(self.best_config, model)

        scheduler = get_scheduler(self.best_config, optimizer, train_loader)

        # Train the model, calculate the training loss
        for epoch in range(self.best_config["epochs"]):
            training_loss = train_epoch(
                config=self.best_config,
                optimizer=optimizer,
                scheduler=scheduler,
                model=model,
                criterion=criterion,
                train_loader=train_loader,
                device=self.device
            )
            logger.info(
                f"Epoch {epoch + 1}, Training Loss: {training_loss}")

        # Save the final model to be used for predictions
        self.best_model = model

        return self

    def predict(self) -> Tuple[np.ndarray, np.ndarray]:
        # Configure the transform for the test dataset, without data augmentation
        dataset_class = get_dataset_class(self.dataset)

        transform = get_transform(
            mean=self.mean_train,
            std=self.std_train,
            num_channels=dataset_class.channels
        )

        dataset_test = dataset_class(
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
        self.best_model.eval()
        with torch.no_grad():
            for data, target in data_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.best_model(data)
                # predicted = torch.argmax(output, 1)
                labels.append(target.to("cpu").numpy())
                predictions.append(output.to("cpu").numpy())
        predictions = np.concatenate(predictions)
        labels = np.concatenate(labels)

        return predictions, labels
