"""Train model.

Trains deep learning model with PytorchLightning to predict
function in rheumatoid arthritis
"""
from __future__ import annotations

import argparse
from pathlib import Path

from predict_haq.train import train_model


def arg_parse_train():
    """Parse args

    Parses args of the path to data and path to where checkpoints
    ought to be saved

    Returns
    -------
    Args
    """
    parser = argparse.ArgumentParser(
        description='''Predict functional outcomes in
                     rheumatoid arthritis using AI''',
    )
    parser.add_argument('--datasetpath', help='path to input data')
    parser.add_argument('--checkpointpath', help='path to input data')
    args = parser.parse_args()
    return args


def main():
    """Takes input arg paths and train models based on parameter
    """
    args = arg_parse_train()
    dataset_path = Path(args.datasetpath)
    checkpoint_path = Path(args.checkpointpath)
    train_model(
        dataset_path=dataset_path,
        checkpoint_path=checkpoint_path,
        which_label='HAQ',
        seed=42,
        image_size=128,
    )


if __name__ == '__main__':
    main()
