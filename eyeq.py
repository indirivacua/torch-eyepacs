from pathlib import Path

import pandas as pd
import requests
import torch
from eyepacs import EyePACS

from EyeQ_process import process


class EyeQ(EyePACS):

    # Create EyeQ dataset class attributes
    image_extension = ".jpg"
    result_label = "quality"
    classes_names = ["Gradable", "Usable", "Ungradable"]

    def __init__(
        self,
        targ_dir: Path = "",  # also called root
        split: str = "train",
        transform=None,
        target_transform=None,
    ) -> None:
        super().__init__(targ_dir, split, transform, target_transform)

        self.__preprocess()

        self.label_csv_name = f"Label_EyeQ_{split}.csv"

        self.__download_results()

        self.df_labels = pd.read_csv(self.data_path / self.label_csv_name, sep=",")

        self.df_labels["image"] = self.df_labels["image"].apply(
            lambda S: S.strip(".jpeg")
        )

    def __download_results(
        self,
    ) -> None:
        # Download EyeQ results
        with open(self.data_path / self.label_csv_name, "wb") as f:
            request = requests.get(
                f"https://raw.githubusercontent.com/HzFu/EyeQ/master/data/{self.label_csv_name}"
            )
            print(f"Downloading Label_EyeQ_{self.split}.csv...")
            f.write(request.content)

    def __preprocess(
        self,
    ) -> None:
        image_list = [str(img) for img in self.paths]
        save_path = str(self.data_path / self.split)

        process(image_list, save_path, self.image_extension)

        self.paths = list(Path(save_path).glob("*" + self.image_extension))


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

    # Setup path to data folder
    data_path = Path("~/.torchvision/")
    image_path = data_path / "eyeq"

    split = "train"

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

        # Adjust image tensor shape for plotting (rearrange the order of dimensions)
        # [color_channels, height, width] -> [height, width, color_channels]
        targ_image_adjust = (
            targ_image.permute(1, 2, 0) if torch.is_tensor(targ_image) else targ_image
        )

        # Setup plot
        plt.imshow(targ_image_adjust)
        plt.axis("off")
        title = f"class: {EyeQ.classes_names[targ_label]}"
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
