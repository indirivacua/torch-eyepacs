import glob
import os
import zipfile
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import torch

from kaggle.api.kaggle_api_extended import KaggleApi
from PIL import Image
from torchvision.datasets import VisionDataset
from torchvision.datasets.utils import check_integrity, extract_archive


def check_exists(root: Path, resources: List[Tuple[str, str]]) -> bool:
    return all(check_integrity(root / file, md5) for file, md5 in resources)


class EyePACS(VisionDataset):

    image_size = 640
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    # Create EyePACS dataset class attributes
    image_extension = ".jpeg"
    result_label = "level"
    label_csv_name = "trainLabels.csv"
    classes_names = ["No DR", "Mild", "Moderate", "Severe", "Proliferative DR"]

    main_zip_name = "diabetic-retinopathy-detection.zip"
    main_zip_hash = "596cf4ecabf92e5e621ac7e9f9181471"

    def __init__(
        self,
        targ_dir: Path = "",  # also called root
        split: str = "train",
        transform=None,
        target_transform=None,
    ) -> None:
        assert split in ["train", "val", "test"]

        if isinstance(targ_dir, str):
            targ_dir = Path(targ_dir)

        targ_dir = targ_dir.expanduser()

        self.split = split

        # Setup train or testing path as root
        images_root = targ_dir / split
        # Setup transforms
        super().__init__(
            images_root, transform=transform, target_transform=target_transform
        )

        self.data_path = self.root.parent

        self.api = KaggleApi()
        self.api.authenticate()

        self.__download_dataset()
        self.__extract_dataset()

        # Get all image paths
        self.paths = list(Path(images_root).glob("*.jpeg"))

        self.df_labels = pd.read_csv(self.data_path / self.label_csv_name, sep=",")

    def __download_dataset(
        self,
    ) -> None:
        main_zip_resource = [(self.main_zip_name, self.main_zip_hash)]
        if check_exists(self.data_path, main_zip_resource):
            return

        # If the image folder doesn't exist, download it and prepare it...
        if not self.root.is_dir():
            self.root.mkdir(parents=True, exist_ok=True)

            # Download diabetic-retinopathy-detection.zip
            print("Downloading dataset (88.29gbs), this may take a while...")

            self.api.competition_download_files(
                "diabetic-retinopathy-detection", path=self.data_path
            )

            if not check_exists(self.data_path, main_zip_resource):
                raise OSError(
                    f"File {self.main_zip_name} has not been downloaded correctly."
                )

    def __extract_dataset(
        self,
    ) -> None:
        # Unzip diabetic-retinopathy-detection.zip
        # extract_archive(self.data_path / self.main_zip_name, self.data_path)

        zip_prefix = f"{self.split}.zip."

        with zipfile.ZipFile(self.data_path / self.main_zip_name, 'r') as zip_ref:
            for file in zip_ref.namelist():
                if file.startswith(zip_prefix):
                    zip_ref.extract(file, self.data_path)
                if not file.startswith(('train.zip', 'test.zip')):
                    zip_ref.extract(file, self.data_path)

        # N number of parts
        parts = glob.glob(str(self.data_path / (zip_prefix + "*")))
        n = len(parts)

        # Concatenate
        with open(self.data_path / f"{self.split}.zip", "wb") as outfile:
            for i in range(1, n + 1):
                filename = zip_prefix + str(i).zfill(3)
                with open(self.data_path / filename, "rb") as infile:
                    outfile.write(infile.read())

        for filename in os.listdir(self.data_path):
            if filename.startswith(zip_prefix):
                os.remove(self.data_path / filename)

        # Extract
        for file_path in os.listdir(self.data_path):
            if os.path.isfile(os.path.join(self.data_path, file_path)):
                main_zip_path = self.data_path / file_path
                if os.path.basename(main_zip_path) == self.main_zip_name:
                    continue
                if main_zip_path.suffix == ".zip":
                    print(f"Extracting {main_zip_path} to {self.data_path}...")
                    extract_archive(self.data_path / file_path, self.data_path)

        os.remove(self.data_path / f"{self.split}.zip")

    def load_image(self, index: int) -> Image.Image:
        "Opens an image via a path and returns it."
        image_name = self.df_labels["image"].iloc[index]
        image_name = image_name + self.image_extension
        image_path = self.root / image_name

        return Image.open(image_path)

    # Overwrites the __len__() method (optional but recommended for subclasses of torch.utils.data.Dataset)
    def __len__(self) -> int:
        "Returns the total number of samples."
        return len(self.paths)

    # Overwrites the __getitem__() method (required for subclasses of torch.utils.data.Dataset)
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        "Returns one sample of data, data and label (X, y)."

        img = self.load_image(index)
        class_idx = self.df_labels[self.result_label].iloc[index]

        # Transform if necessary
        if self.transform:
            img = self.transform(img)

        return img, class_idx  # return data, label (X, y)


if __name__ == "__main__":

    from torchvision import transforms

    data_transform = transforms.Compose(
        [
            # Resize and Crop the images to 640x640
            transforms.Resize(EyePACS.image_size),
            transforms.CenterCrop(EyePACS.image_size),
            # Turn the image into a torch.Tensor
            transforms.ToTensor(),  # this also converts all pixel values from 0 to 255 to be between 0.0 and 1.0
            transforms.Normalize(EyePACS.mean, EyePACS.std),
        ]
    )

    # Setup path to data folder
    data_path = Path("~/.torchvision/")
    image_path = data_path / "eyepacs"

    split = "train"

    dataset = EyePACS(image_path, split=split, transform=data_transform)  # train_data
    n = len(dataset)
    print(f"EyePACS, split {split}, has {n} samples.")

    ################################################################################################

    import matplotlib.pyplot as plt
    import torchvision

    def display_image(
        dataset: torchvision.datasets.VisionDataset, targ_sample: int
    ) -> None:
        targ_image, targ_label = dataset[targ_sample][0], dataset[targ_sample][1]

        # Adjust image tensor shape for plotting (rearrange the order of dimensions)
        # [color_channels, height, width] -> [height, width, color_channels]
        targ_image_adjust = (
            targ_image.permute(1, 2, 0) if torch.is_tensor(targ_image) else targ_image
        )

        # Setup plot
        plt.imshow(targ_image_adjust)
        plt.axis("off")
        title = f"class: {EyePACS.classes_names[targ_label]}"
        if torch.is_tensor(targ_image_adjust):
            title = title + f"\nshape: {targ_image_adjust.shape}"
        plt.title(title)

        # Plot the adjusted sample
        plt.show()

    display_image(dataset, 2)  # 13_left

    ################################################################################################

    from torch.utils.data import DataLoader

    train_dataloader_custom = DataLoader(
        dataset=dataset,  # use custom created train Dataset
        batch_size=1,  # how many samples per batch?
        num_workers=0,  # how many subprocesses to use for data loading? (higher = more)
        shuffle=True,  # shuffle the data?
    )

    # Get image and label from custom DataLoader
    img_custom = None
    while img_custom is None:
        try:
            img_custom, label_custom = next(iter(train_dataloader_custom))
        except (StopIteration, TypeError):
            break
        except:
            pass

    # Batch size will now be 1, try changing the batch_size parameter above and see what happens
    print(
        f"Image shape: {img_custom.shape} -> [batch_size, color_channels, height, width]"
    )
    print(f"Label shape: {label_custom.shape}")
