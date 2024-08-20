from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.data import random_split
from torchvision import transforms
from torchvision.models import densenet161
from torchvision.models import DenseNet161_Weights


class XrayDataset(Dataset):
    """Defines dataset for xray data

    Parameters
    ----------
    Dataset : Dataset
        base dataset class from pytorch
    """

    def __init__(self, image_dir, labels, transform=None):
        self.image_dir = image_dir
        self.labels = labels
        self.transform = transform
        self.image_names = os.listdir(image_dir)

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        img_name = self.image_names[idx]
        img_path = os.path.join(self.image_dir, img_name)
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        label = self.labels[idx]

        return image, label


class XrayDataModule(pl.LightningDataModule):
    """ Pytorch lightning data module

    Parameters
    ----------
    pl.LightningDataModule : pl.LightningDataModule
        base class from pytorch lightning
    """

    def __init__(
        self, image_dir, labels, batch_size=4,
        transform=None, val_split=0.2,
    ):
        super().__init__()
        self.image_dir = image_dir
        self.labels = labels
        self.batch_size = batch_size
        self.transform = transform
        self.val_split = val_split

    def setup(self, stage=None):
        dataset_size = len(self.labels)
        val_size = int(dataset_size * self.val_split)
        train_size = dataset_size - val_size
        train_indices, val_indices = random_split(
            range(dataset_size), [train_size, val_size],
        )

        # NEED TO USE TRAIN-INDICES AND VAL-INDICES Data
        # Create train and validation datasets
        self.dataset = XrayDataset(
            image_dir=self.image_dir,
            labels=self.labels,
            transform=self.transform,
        )
        # self.val_dataset = XrayDataset(image_dir=self.image_dir,
        # labels=self.labels,
        # transform=self.transform)

    def train_dataloader(self):
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=8,
            pin_memory=True,
        )

    # def valid_dataloader(self):
    #     return DataLoader(self.dataset,
    #                       batch_size=self.batch_size,
    #                       shuffle=True)


class DenseNetLightning(pl.LightningModule):
    """Defines DenseNet Model

    Parameters
    ----------
    pl : _type_
        _description_
    """

    def __init__(self):
        super().__init__()
        self.model = densenet161(weights=DenseNet161_Weights.IMAGENET1K_V1)
        self.model.classifier = nn.Linear(self.model.classifier.in_features, 1)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch):
        inputs, targets = batch
        inputs = inputs.float()
        targets = targets.float()
        outputs = self(inputs).squeeze(dim=1)
        loss = nn.MSELoss()(outputs, targets)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)


def train_model(
    DATASET_PATH: Path,
    CHECKPOINT_PATH: Path,
    which_label: str,
    seed: int,
    max_epochs: int,
    batch_size: int,
    image_size: int,
):
    """Train lightning model

    Instantiates data, model and trainer classes then fits model

    Parameters
    ----------
    DATASET_PATH : Path
        Path to the image and csv data
    CHECKPOINT_PATH : Path
        Path to save model training files
    which_label : str
        Which functional outcome to train to
        (i.e. HAQ, HAQ-change, pain, SF36)
    seed : int
        random seed for everything pl related
    max_epochs : int
        num training epochs
    batch_size : int
        number of batches
    image_size : int
        size to resize image to
        (note this is a square so we only take one number)
    """
    # Function for setting the seed
    pl.seed_everything(seed)

    # Ensure that all operations are deterministc for reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Create checkpoint path if it doesn't exist yet
    os.makedirs(CHECKPOINT_PATH, exist_ok=True)

    # Define any transformations
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])

    # NEED TO DROP IF GREATERN THAN MAX VALUE FOR HAQ FIRST
    # NEED A GENERALISABLE SOLUTION DEPENDENT ON OUTCOME CHOSEN TO TRAIN TO

    labels_df = pd.read_csv(
        DATASET_PATH / 'dataframes' /
        'xrays_train.csv',
    ).dropna(subset=[which_label])
    data_module = XrayDataModule(
        image_dir=DATASET_PATH / 'xray_images', labels=list(
            labels_df[which_label],
        ), batch_size=batch_size, transform=transform,
    )
    data_module.setup()
    train_loader = data_module.train_dataloader()
    # val_loader = data_module.val_dataloader()

    torch.set_float32_matmul_precision('medium')

    model = DenseNetLightning()
    trainer = pl.Trainer(
        deterministic=True,
        max_epochs=max_epochs, log_every_n_steps=50,
    )

    trainer.fit(model, train_loader)
