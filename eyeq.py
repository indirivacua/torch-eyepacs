import os
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import requests
import torch
from PIL import Image
from torchvision.datasets import VisionDataset
from torchvision.datasets.utils import check_integrity, extract_archive


def check_exists(root: Path, resources: List[Tuple[str, str]]) -> bool:
    return all(check_integrity(root / file, md5) for file, md5 in resources)


class EyeQ(VisionDataset):

    # Create EyeQ dataset class attributes
    image_size = 640
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    classes_names = ["Gradable", "Usable", "Ungradable"]

    # TODO: get diabetic-retinopathy-detection hashes
    # > certutil -hashfile sample.zip MD5
    main_zip_name = "sample.zip"
    main_zip_hash = "d8da149561d61d6f97256f0dea3552ea"

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

        # Setup train or testing path
        images_root = targ_dir / split
        # Setup transforms
        super().__init__(
            images_root, transform=transform, target_transform=target_transform
        )

        self.__download()
        self.__preprocess()

        # Get all image paths
        self.paths = list(Path(images_root).glob("*.jpeg"))

    def __download(
        self,
    ) -> None:
        root: Path = self.root

        data_path = root.parent

        main_zip_resource = [(EyeQ.main_zip_name, EyeQ.main_zip_hash)]
        if check_exists(data_path, main_zip_resource):
            return

        # If the image folder doesn't exist, download it and prepare it...
        if not root.is_dir():
            root.mkdir(parents=True, exist_ok=True)

            # Download diabetic-retinopathy-detection.zip
            # TODO: change sample.zip to train/test data
            command = "kaggle competitions download -c diabetic-retinopathy-detection -f sample.zip"
            current_path = Path.cwd()
            os.chdir(data_path)
            print("Downloading dataset (88.29gbs) this may take a while...")
            subprocess.run(command, shell=True)
            os.chdir(current_path)
            if not check_exists(data_path, main_zip_resource):
                raise OSError(
                    f"File {EyeQ.main_zip_name} has not been downloaded correctly."
                )

            # Download EyeQ results
            # TODO: download EyeQ_process_main.py over the DR-dataset (clone it from GitHub?)
            with open(data_path / f"Label_EyeQ_{self.split}.csv", "wb") as f:
                request = requests.get(
                    f"https://raw.githubusercontent.com/HzFu/EyeQ/master/data/Label_EyeQ_{self.split}.csv"
                )
                print(f"Downloading Label_EyeQ_{self.split}.csv...")
                f.write(request.content)

            # Unzip diabetic-retinopathy-detection.zip
            for file_path in os.listdir(data_path):
                if os.path.isfile(os.path.join(data_path, file_path)):
                    if (data_path / file_path).suffix == ".zip":
                        extract_archive(data_path / file_path, data_path)

            # TEST PURPOSES
            from shutil import move, rmtree

            rmtree(data_path / "train"), move(data_path / "sample", data_path / "train")

    def __preprocess(
        self,
    ) -> None:
        # TODO: execute EyeQ_process_main.py over the DR-dataset
        pass

    def load_image(self, index: int) -> Image.Image:
        "Opens an image via a path and returns it."
        image_path = self.paths[index]
        return Image.open(image_path)

    # Overwrites the __len__() method (optional but recommended for subclasses of torch.utils.data.Dataset)
    def __len__(self) -> int:
        "Returns the total number of samples."
        return len(self.paths)

    # Overwrites the __getitem__() method (required for subclasses of torch.utils.data.Dataset)
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        "Returns one sample of data, data and label (X, y)."

        df = pd.read_csv(self.root.parent / f"Label_EyeQ_{self.split}.csv", sep=",")

        img = self.load_image(index)

        query = df[df["image"] == str(self.paths[index]).rsplit("\\", 1)[-1]]
        class_idx = query["quality"].iat[0]

        # Transform if necessary
        if self.transform:
            return self.transform(img), class_idx  # return data, label (X, y)
        else:
            return img, class_idx  # return data, label (X, y)


if __name__ == "__main__":

    from torchvision import transforms

    data_transform = transforms.Compose(
        [
            # Resize and Crop the images to 640x640
            transforms.Resize(EyeQ.image_size),
            transforms.CenterCrop(EyeQ.image_size),
            # Turn the image into a torch.Tensor
            transforms.ToTensor(),  # this also converts all pixel values from 0 to 255 to be between 0.0 and 1.0
            transforms.Normalize(EyeQ.mean, EyeQ.std),
        ]
    )

    split = "train"

    # Setup path to data folder
    data_path = Path("~/.torchvision/")
    image_path = data_path / "eyeq"

    dataset = EyeQ(image_path, split=split, transform=data_transform)  # train_data
    n = len(dataset)
    print(f"EyeQ, split {split}, has {n} samples.")

    ################################################################################################

    import matplotlib.pyplot as plt
    import torchvision

    def display_image(
        dataset: torchvision.datasets.VisionDataset, targ_sample: int
    ) -> None:
        targ_image, targ_label = dataset[targ_sample][0], dataset[targ_sample][1]

        # Adjust image tensor shape for plotting:
        # [color_channels, height, width] -> [height, width, color_channels]
        targ_image_adjust = targ_image.permute(
            1, 2, 0
        )  # Rearrange the order of dimensions

        # Setup plot
        plt.imshow(targ_image_adjust)
        plt.axis("off")
        plt.title(
            f"class: {EyeQ.classes_names[targ_label]} \nshape: {targ_image_adjust.shape}"
        )

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
        except:
            pass

    # Batch size will now be 1, try changing the batch_size parameter above and see what happens
    print(
        f"Image shape: {img_custom.shape} -> [batch_size, color_channels, height, width]"
    )
    print(f"Label shape: {label_custom.shape}")
