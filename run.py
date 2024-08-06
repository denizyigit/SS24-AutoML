"""An example run file which trains a dummy AutoML system on the training split of a dataset
and logs the accuracy score on the test set.

In the example data you are given access to the labels of the test split, however
in the test dataset we will provide later, you will not have access
to this and you will need to output your predictions for the images of the test set
to a file, which we will grade using github classrooms!
"""
from __future__ import annotations

from pathlib import Path
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, f1_score, precision_score, recall_score, top_k_accuracy_score
import numpy as np
from automl.utils import get_dataset_class
from src.automl.automl import AutoML
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
import logging


logger = logging.getLogger(__name__)


def plot_confusion_matrix(cm, class_names):
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                  display_labels=class_names)
    disp.plot()
    plt.show()


def main(
        dataset: str,
        output_path: Path,
        seed: int,
        reduced_dataset_ratio: float,
        max_evaluations_total: int,
        max_epochs: int,
        num_process: int
):
    logger.info("Fitting AutoML")

    # Create an instance of the AutoML class
    automl = AutoML(
        seed=seed,
        dataset=dataset,
        reduced_dataset_ratio=reduced_dataset_ratio,
        max_evaluations_total=max_evaluations_total,
        max_epochs=max_epochs,
        num_process=num_process
    )

    # Start automation pipeline to find the best configuration and train a model with it
    automl.fit()

    # Get the predictions of the test set from the best model (You must have run automl.fit() before this)
    test_preds, test_labels = automl.predict()

    # Write the predictions of X_test to disk
    logger.info("Writing predictions to disk")
    with output_path.open("wb") as f:
        np.save(f, test_preds)

    # In case of running on the test data, also add the predictions.npy
    # to the correct location for autoevaluation.
    if dataset == "skin_cancer":
        test_output_path = Path("data/exam_dataset/predictions.npy")
        test_output_path.parent.mkdir(parents=True, exist_ok=True)
        with test_output_path.open("wb") as f:
            np.save(f, test_preds)

    # Calculate the top-1 and top-5 accuracy score of the model on the test set
    if not np.isnan(test_labels).any():
        acc_top_1 = top_k_accuracy_score(test_labels, test_preds, k=1, labels=range(
            get_dataset_class(dataset).num_classes))
        acc_top_5 = top_k_accuracy_score(test_labels, test_preds, k=5, labels=range(
            get_dataset_class(dataset).num_classes))

        print(f"Test labels length: {len(test_labels)}")
        logger.info(f"Top 1 Accuracy on test set: {acc_top_1}")
        logger.info(f"Top 5 Accuracy on test set: {acc_top_5}")

        test_preds_labels = np.argmax(test_preds, axis=1)

        # Plot confusion matrix
        if dataset == "emotions":
            #class_names = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
            class_names = range(get_dataset_class(dataset).num_classes)
            cm = confusion_matrix(test_labels, test_preds_labels, labels=class_names)
            plot_confusion_matrix(cm, class_names)
            logger.info(
                f"Confusion matrix saved to {'confusion_matrix.png'}")

        # Precision, Recall, and F1-Score
        precision = precision_score(
            test_labels, test_preds_labels, average='macro')
        recall = recall_score(test_labels, test_preds_labels, average='macro')
        f1 = f1_score(test_labels, test_preds_labels, average='macro')

        logger.info(f"Precision: {precision}")
        logger.info(f"Recall: {recall}")
        logger.info(f"F1 Score: {f1}")

    else:
        # This is the setting for the exam dataset, you will not have access to the labels
        logger.info(f"No test split for dataset '{dataset}'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="The name of the dataset to run on.",
        choices=["fashion", "flowers", "emotions", "skin_cancer"]
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("predictions.npy"),
        help=(
            "The path to save the predictions to."
            " By default this will just save to './predictions.npy'."
        )
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help=(
            "Random seed for reproducibility if you are using and randomness,"
            " i.e. torch, numpy, pandas, sklearn, etc."
        )
    )

    parser.add_argument(
        "--reduced-dataset-ratio",
        type=float,
        default=1.0,
        help=(
            "Ratio of the dataset to use for training."
        )
    )

    parser.add_argument(
        "--max-evaluations-total",
        type=int,
        default=10,
        help=(
            "Total number of configurations to select & evaluate during model training. Default is 10."
        )
    )

    parser.add_argument(
        "--max-epochs",
        type=int,
        default=15,
        help=(
            "Maximum number of epochs (budget) to train the model for with a selected configuration. Default is 15."
        )
    )

    parser.add_argument(
        "--num-process",
        type=int,
        default=1,
        help=(
            "Number of configs processes to run neps.run() in parallel. Default is 1."
        )
    )

    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Whether to log only warnings and errors."
    )

    args = parser.parse_args()

    if not args.quiet:
        logging.basicConfig(level=logging.INFO)
    else:
        logging.basicConfig(level=logging.WARNING)

    logger.info(
        f"Running dataset {args.dataset}"
        f"\n{args}"
    )

    main(
        dataset=args.dataset,
        output_path=args.output_path,
        seed=args.seed,
        reduced_dataset_ratio=args.reduced_dataset_ratio,
        max_evaluations_total=args.max_evaluations_total,
        max_epochs=args.max_epochs,
        num_process=args.num_process
    )
