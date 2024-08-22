"""Splits train valid data and trains model."""
from __future__ import annotations

import os
import random
from pathlib import Path

import pandas as pd
import pytorch_lightning as pl
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
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

    def __init__(self, image_dir, labels, image_names, transform=None):
        self.image_dir = image_dir
        self.labels = labels
        self.image_names = image_names
        self.transform = transform

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        img_name = self.image_names[idx] + '.png'
        img_path = os.path.join(self.image_dir, img_name)

        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        label = self.labels[idx]

        return image, label


class XrayDataModule(pl.LightningDataModule):
    """Pytorch lightning data module

    Parameters
    ----------
    pl.LightningDataModule : pl.LightningDataModule
        base class from pytorch lightning
    """

    def __init__(
            self, data, image_dir,
            transform=None, val_split=0.2,
    ):
        super().__init__()
        self.data = data
        self.image_dir = image_dir
        self.transform = transform
        self.val_split = val_split
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def setup(self, stage: str):
        if stage == 'fit':
            # Split into train and validation randomly at the patient level
            unique_ids = self.data['Patient_ID'].unique()
            random.shuffle(unique_ids)
            cutoff = round(len(unique_ids) * self.val_split)
            valid_patient_ids = unique_ids[0:cutoff]
            train_patient_ids = unique_ids[cutoff:]

            # Keep rows where the filepath to the image exists
            train_data = self.data[
                self.data['Patient_ID'].isin(
                    train_patient_ids,
                )
            ].dropna(
                subset=['HAQ', 'UID'],
            ).reset_index(drop=True)
            valid_data = self.data[
                self.data['Patient_ID'].isin(
                    valid_patient_ids,
                )
            ].dropna(
                subset=['HAQ', 'UID'],
            ).reset_index(drop=True)

            train_data = train_data[[
                os.path.isfile(
                    str(self.image_dir) + '/' + str(i) + '.png',
                ) for i in train_data['UID']
            ]]
            valid_data = valid_data[[
                os.path.isfile(
                    str(self.image_dir) + '/' + str(i) + '.png',
                ) for i in valid_data['UID']
            ]]

            train_image_names = list(train_data['UID'])
            train_labels = list(train_data['HAQ'])
            valid_image_names = list(valid_data['UID'])
            valid_labels = list(valid_data['HAQ'])

            self.train_dataset = XrayDataset(
                image_dir=self.image_dir,
                labels=train_labels,
                image_names=train_image_names,
                transform=self.transform,
            )
            self.val_dataset = XrayDataset(
                image_dir=self.image_dir,
                labels=valid_labels,
                image_names=valid_image_names,
                transform=self.transform,
            )
        if stage == 'test':
            # use data from the test dataframe here

            test_labels = None
            test_image_names = None

            self.test_dataset = XrayDataset(
                image_dir=self.image_dir,
                labels=test_labels,
                image_names=test_image_names,
                transform=self.transform,
            )

    def train_dataloader(self):
        """Training data loader

        Returns
        -------
        DataLoader
        """
        return DataLoader(
            self.train_dataset,
            batch_size=8,
            shuffle=True,
            num_workers=2,
            pin_memory=True,
        )

    def val_dataloader(self):
        """Validation data loader

        Returns
        -------
        DataLoader
        """
        return DataLoader(
            self.val_dataset,
            batch_size=8,
            shuffle=False,
            num_workers=2,
            pin_memory=True,
        )

    def test_dataloader(self):
        """Test Dataloader

        Returns
        -------
        DataLoader
        """
        return DataLoader(
            self.test_dataset,
            batch_size=8,
            shuffle=False,
            num_workers=2,
            pin_memory=True,
        )


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

    def forward(self, inputs):  # pylint: disable=arguments-differ
        """Forward pass

        Parameters
        ----------
        inputs : torch.Tensor
            input image

        Returns
        -------
        torch.Tensor
            prediction
        """
        return self.model(inputs)

    def training_step(self, batch):  # pylint: disable=arguments-differ
        """Takes training step, calculates loss per batch

        Parameters
        ----------
        batch : torch.Tensor
            batch of image data
        Returns
        -------
        float
            loss value
        """
        inputs, targets = batch
        inputs = inputs.float()
        targets = targets.float()
        outputs = self(inputs).squeeze(dim=1)
        loss = nn.MSELoss()(outputs, targets)
        return loss

    # def validation_step(self, batch):

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)


def train_model(
        dataset_path: Path,
        checkpoint_path: Path,
        which_label: str,
        seed: int,
        image_size: int,
):
    """Train lightning model

    Instantiates data, model and trainer classes then fits models

    Parameters
    ----------
    dataset_path : Path
        Path to the image and csv data
    checkpoint_path : Path
        Path to save model training files
    which_label : str
        Which functional outcome to train to
        (i.e. HAQ, HAQ-change, pain, SF36)
    seed : int
        random seed for everything pl related
    max_epochs : int
        num training epochs
    image_size : int
        size to resize image to
        (note this is a square so we only take one number)
    lr: float
        learning rate
    """
    # Function for setting the seed
    pl.seed_everything(seed)

    # Ensure that all operations are deterministc for reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Create checkpoint path if it doesn't exist yet
    os.makedirs(checkpoint_path, exist_ok=True)

    # Define any transformations
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])

    # NEED TO DROP IF GREATERN THAN MAX VALUE FOR HAQ FIRST
    # NEED A GENERALISABLE SOLUTION DEPENDENT ON OUTCOME CHOSEN TO TRAIN TO

    labels_df = pd.read_csv(
        dataset_path / 'dataframes' /
        'xrays_train.csv',
    ).dropna(subset=[which_label])

    data_module = XrayDataModule(
        data=labels_df,
        transform=transform,
        image_dir=dataset_path / 'xray_images',
    )
    data_module.setup(stage='fit')
    train_loader = data_module.train_dataloader()
    val_loader = data_module.val_dataloader()

    torch.set_float32_matmul_precision('medium')

    model = DenseNetLightning()
    trainer = pl.Trainer(
        deterministic=True,
        max_epochs=100, log_every_n_steps=50,
    )

    trainer.fit(model, train_loader, val_loader)
