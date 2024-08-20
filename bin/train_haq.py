from __future__ import annotations

import argparse
from pathlib import Path

from predict_haq.train import train_model


def arg_parse_train():
    parser = argparse.ArgumentParser(
        description='''Predict functional outcomes in
                     rheumatoid arthritis using AI''',
    )
    parser.add_argument('--datasetpath', help='path to input data')
    parser.add_argument('--checkpointpath', help='path to input data')
    args = parser.parse_args()
    return args


def main():
    args = arg_parse_train()
    dataset_path = Path(args.datasetpath)
    checkpoint_path = Path(args.checkpointpath)
    train_model(
        DATASET_PATH=dataset_path,
        CHECKPOINT_PATH=checkpoint_path,
        which_label='HAQ',
        seed=42,
        max_epochs=100,
        batch_size=4,
        image_size=128,
    )


if __name__ == '__main__':
    main()
