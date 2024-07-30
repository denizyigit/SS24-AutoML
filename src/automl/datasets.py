"""
This module contains the datasets used in the AutoML exam.
If you want to edit this file be aware that we will later 
  push the test set to this file which might cause problems.

"""


from pathlib import Path
from typing import Any, Callable, Optional, Tuple, Union

import PIL.Image
import pandas as pd
from torchvision.datasets import VisionDataset
from torchvision.datasets.utils import download_and_extract_archive, check_integrity


BASE_URL = "https://ml.informatik.uni-freiburg.de/research-artifacts/automl-exam-24-vision/"


class BaseVisionDataset(VisionDataset):
    """A base class for all vision datasets.

    Args:
        root: str or Path
            Root directory of the dataset.
        split: string (optional)
            The dataset split, supports `train` (default), `val`, or `test`.
        transform: callable (optional)
            A function/transform that takes in a PIL image and returns a transformed version. 
            E.g, `transforms.RandomCrop`.
        target_transform: callable (optional)
            A function/transform that takes in the target and transforms it.
        download: bool (optional)
            If true, downloads the dataset from the internet and puts it in root directory. 
            If dataset is already downloaded, it is not downloaded again.
    """
    _download_url_prefix = BASE_URL
    _download_file = Tuple[str, str] # Checksum that is provided here is not used.
    _dataset_name: str
    _md5_train: str
    _md5_test: str
    width: int
    height: int
    channels: int
    num_classes: int

    def __init__(
        self,
        root: Union[str, Path],
        split: str = "train",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
    ) -> None:
        super().__init__(root, transform=transform, target_transform=target_transform)
        assert split in ["train", "test"], f"Split {split} not supported"
        self._split = split
        self._base_folder = Path(self.root) / self._dataset_name

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError(
                "Dataset not found or corrupted. You can use download=True to download it "
                f"or download it manually from {self._download_url_prefix}{self._download_file[0]}"
            )

        data = pd.read_csv(self._base_folder / f"{self._split}.csv")
        self._labels = data['label'].tolist()
        self._image_files = data['image_file_name'].tolist()

    def _check_integrity(self):
        train_images_folder = self._base_folder / "images_train"
        test_images_folder = self._base_folder / "images_test"
        if not (train_images_folder.exists() and train_images_folder.is_dir()) or \
           not (test_images_folder.exists() and test_images_folder.is_dir()):
            return False

        for filename, md5 in [("train.csv", self._md5_train), ("test.csv", self._md5_test)]:
            if not check_integrity(str(self._base_folder / filename), md5):
                return False
        return True

    def download(self):
        """Download the dataset from the URL.
        """
        if self._check_integrity():
            return
        download_and_extract_archive(
            f"{self._download_url_prefix}{self._download_file[0]}",
            str(self._base_folder),
        )

    def extra_repr(self) -> str:
        """String representation of the dataset.
        """
        return f"split={self._split}"

    def __getitem__(self, idx: int) -> Tuple[Any, Any]:
        image_file, label = self._image_files[idx], self._labels[idx]
        image = PIL.Image.open(self._base_folder / f"images_{self._split}" / image_file)
        if self.channels == 1:
            image = image.convert("L")
        elif self.channels == 3:
            image = image.convert("RGB")
        else:
            raise ValueError(f"Unsupported number of channels: {self.channels}")

        if self.transform:
            image = self.transform(image)

        if self.target_transform:
            label = self.target_transform(label)

        return image, label

    def __len__(self) -> int:
        return len(self._image_files)


class EmotionsDataset(BaseVisionDataset):
    """ Emotions Dataset.

    This dataset contains images of faces displaying in to one of seven emotions
    (0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral).
    """
    _download_file = ("emotions.tgz", "e8302a10bc38a7bfb2e60c67b6bab1e4")
    _dataset_name = "emotions"
    _md5_train = "7a48baafcddeb5b9caaa01c5b9fcd309"
    _md5_test = "6a4b219c98be434ca0c79da2df3b2f35"
    width = 48
    height = 48
    channels = 1
    num_classes = 7


class FlowersDataset(BaseVisionDataset):
    """Flower Dataset.

    This dataset contains images of 102 types of flowers. The task is to classify the flower type.
    """
    _download_file = ("flowers.tgz", "31ff68dec06e95997aa4d77cd1eb5744")
    _dataset_name = "flowers"
    _md5_train = "08f3283cfa42d37755bcf972ed368264"
    _md5_test = "778c82088dc9fc3659e9f14614b20735"
    width = 512
    height = 512
    channels = 3
    num_classes = 102


class FashionDataset(BaseVisionDataset):
    """Fashion Dataset.

    This dataset contains images of fashion items. The task is to classify what kind of fashion item it is.
    """
    _download_file = ("fashion.tgz", "ec70b7addb6493d4e3d57939ff76e2d5")
    _dataset_name = "fashion"
    _md5_train = "a364148066eb5bace445e4c9e7fb95d4"
    _md5_test = "1d0bf72b43a3280067abb82d91c0c245"
    width = 28
    height = 28
    channels = 1
    num_classes = 10
