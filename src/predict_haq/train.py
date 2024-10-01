"""Splits train valid data and trains model."""
from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import roc_utils
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision.models import densenet161
from torchvision.transforms import v2


class XrayDataset(Dataset):
    """Defines dataset for xray data

    Parameters
    ----------
    Dataset : Dataset
        base dataset class from pytorch
    """

    def __init__(self, image_dir, labels, image_names, transform):
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
            self, data, image_dir, outcome, seed,
            transform=None, val_split=0.2, num_workers=2,

    ):
        super().__init__()
        self.data = data
        self.image_dir = image_dir
        self.outcome = outcome
        self.seed = seed
        self.transform = transform
        self.val_split = val_split
        self.num_workers = num_workers
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def setup(self):
        """Setup training, validation and testdata

        Parameters
        ----------
        stage : str
            Fitting, testing or prediction stage
        """
        # Split into train and validation randomly at the patient level
        unique_ids = self.data['Patient_ID'].unique()
        random.seed(self.seed)
        random.shuffle(unique_ids)
        cutoff = round(len(unique_ids) * self.val_split)
        valid_patient_ids = unique_ids[0:cutoff]
        train_patient_ids = unique_ids[cutoff:]

        train_data = self.data[
            self.data['Patient_ID'].isin(
                train_patient_ids,
            )
        ]

        valid_data = self.data[
            self.data['Patient_ID'].isin(
                valid_patient_ids,
            )
        ]

        train_image_names = list(train_data['UID'])
        valid_image_names = list(valid_data['UID'])

        train_labels = np.array(list(train_data[self.outcome]))
        valid_labels = np.array(list(valid_data[self.outcome]))

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

        # This can be changed to the test data once get to that point
        # Test on validation data until finished model development
        self.test_dataset = XrayDataset(
            image_dir=self.image_dir,
            labels=valid_labels,
            image_names=valid_image_names,
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
            batch_size=4,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=False,
            drop_last=True,
        )

    def val_dataloader(self):
        """Validation data loader

        Returns
        -------
        DataLoader
        """
        return DataLoader(
            self.val_dataset,
            batch_size=4,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=False,
            drop_last=True,

        )

    def test_dataloader(self):
        """Test Dataloader

        Returns
        -------
        DataLoader
        """
        return DataLoader(
            self.test_dataset,
            batch_size=4,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=False,
            drop_last=True,
        )


class DenseNetLightning(pl.LightningModule):
    """Defines DenseNet Model

    Parameters
    ----------
    pl : _type_
        _description_
    """

    def __init__(
        self, out_features=1, learning_rate=1e-4, outcome='HAQ',
        handsorfeet=None, seed=123,
    ):
        super().__init__()
        self.model = densenet161()  # weights=DenseNet161_Weights.IMAGENET1K_V1
        self.model.classifier = nn.Linear(
            self.model.classifier.in_features,
            out_features,
        )
        self.learning_rate = learning_rate
        self.outcome = outcome
        self.handsorfeet = handsorfeet
        self.seed = seed
        self.test_preds = []
        self.test_true = []

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
        ----------git
        batch : torch.Tensor
            batch of image data
        Returns
        -------
        float
            loss value
        """
        inputs, targets = batch
        inputs = inputs.float()
        targets = targets.float()/18
        outputs = self(inputs).squeeze(dim=1)
        loss = nn.MSELoss()(outputs, targets)
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch):  # pylint: disable=arguments-differ
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
        rmse = torch.sqrt(loss)

        self.log('val_MSE (loss)', loss)
        self.log('val_RMSE', rmse)

    def test_step(self, batch):
        """Takes test step, calculates loss per batch

        Logs the MSE and RMSE

        Parameters
        ----------
        batch : ?
            batch of data
        """
        inputs, targets = batch
        inputs = inputs.float()
        targets = targets.float()
        outputs = self(inputs).squeeze(dim=1)
        loss = nn.MSELoss()(outputs, targets)
        rmse = torch.sqrt(loss)

        self.log('test_MSE', loss)
        self.log('test_RMSE', rmse)

        self.test_preds.append(outputs)
        self.test_true.append(targets)

    def on_test_epoch_end(self):
        """Aggregate outputs from the entire tes tdataset
        Returns
        -------
        Mean AUC
            Returns the mean auc from bootstrapped AUCs
        """
        # Optionally, aggregate outputs from the entire test dataset
        preds = torch.cat(self.test_preds)
        targets = torch.cat(self.test_true)
        avg_mse = nn.MSELoss()(preds, targets)

        targets_bin = [0 if i == 0 else 1 for i in targets.cpu()]

        preds = [np.array(i.cpu()) for i in preds]
        targets = [np.array(i.cpu()) for i in targets]

        rocs = roc_utils.compute_roc_bootstrap(
            X=preds, y=targets_bin, pos_label=1,
            n_bootstrap=10000,
            random_state=42,
            return_mean=False,
        )
        roc_mean = roc_utils.compute_mean_roc(rocs)
        auc_mean = round(float(roc_mean['auc_mean']), 3)
        auc95_ci = [round(float(i), 3) for i in roc_mean['auc95_ci'][0]]

        dataframe = pd.DataFrame({'Preds': preds, 'Targets': targets})
        filename = str(self.handsorfeet) + '_' + self.outcome + \
            '_' + str(self.seed) + '_AI.csv'
        dataframe.to_csv(Path('figures/') / filename)
        # Log the averaged metrics
        self.log('avg_test_MSE', avg_mse)
        self.log('mean_AUC', auc_mean)
        self.log('95_ci_lower', auc95_ci[0])
        self.log('95_ci_upper', auc95_ci[1])

        return roc_mean, auc95_ci[0], auc95_ci[1]

    def configure_optimizers(self):
        """Configure optimizer

        Returns
        -------
        torch.optim
            Optimizer
        """
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)


def train_model(
        image_path: Path,
        data: pd.DataFrame,
        outcome_train: str,
        handsorfeet: str,
        checkpoint_path: Path,
        outcome: str,
        seed: int,
        image_size: int,
        max_epochs: int,
        learning_rate: int,
):
    """Train lightning model

    Instantiates data, model and trainer classes then fits models

    Parameters
    ----------
    dataset_path : Path
        Path to the image and csv data
    checkpoint_path : Path
        Path to save model training files
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
    transform = v2.Compose([
        v2.Resize((image_size+100, image_size+100)),
        v2.RandomResizedCrop(size=(image_size, image_size)),
        v2.RandomHorizontalFlip(p=0.5),
        v2.ColorJitter(brightness=(0.4, 0.6), contrast=(0.4, 0.6)),
        v2.RandomRotation(degrees=(-10, 10)),
        v2.RandomInvert(p=0.5),
        v2.ToTensor(),
    ])

    data_module = XrayDataModule(
        data=data,
        outcome=outcome,
        seed=seed,
        transform=transform,
        image_dir=image_path,
    )
    data_module.setup()
    train_loader = data_module.train_dataloader()
    val_loader = data_module.val_dataloader()
    test_loader = data_module.test_dataloader()

    torch.set_float32_matmul_precision('medium')

    model = DenseNetLightning(
        out_features=1,
        learning_rate=learning_rate,
        outcome=outcome_train,
        handsorfeet=handsorfeet,
        seed=seed,
    )

    tb_logger = pl.loggers.TensorBoardLogger(
        save_dir='lightning_logs',  # Change this to your desired directory
        # Name of the experiment, can include subfolders
        name=(f'''LOG_{outcome}
              _SEED_{str(seed)}f
              _IMSIZE_{str(image_size)}
              _EPOCHS_{str(max_epochs)}
              _LR_{str(learning_rate)}'''),
        version=1,                # Version of the experiment
    )

    trainer = pl.Trainer(
        deterministic=True,
        max_epochs=max_epochs,
        logger=tb_logger,
        accelerator='gpu',
    )

    trainer.fit(model, train_loader, val_loader)
    auc = trainer.test(model, test_loader)
    filename = str(outcome_train) + '_results.csv'
    path_df = Path(checkpoint_path / filename)

    params_dict = {
        'Outcome': outcome, 'Hands or feet': handsorfeet, 'Image path': image_path, 'Seed': seed,
        'Imsize': image_size, 'Epochs': max_epochs, 'LR': learning_rate, 'AUC': auc,
    }

    data_row = pd.DataFrame(
        params_dict,
        index=[0],
    ).reset_index(drop=True)
    # if df exists
    if os.path.isfile(path_df):
        results_df = pd.read_csv(
            path_df, index_col=False,
        ).reset_index(drop=True)
        results_df = pd.concat([results_df, data_row], ignore_index=True)
    else:
        results_df = data_row

    results_df.to_csv(path_df, index=False)
