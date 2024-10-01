"""Train model.

Trains deep learning model with PytorchLightning to predict
function in rheumatoid arthritis
"""
from __future__ import annotations

import argparse
from pathlib import Path

from predict_haq.preprocessing import process_dataframe
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
    parser.add_argument('--csvpath', help='path to input csv data')
    parser.add_argument('--imagepath', help='path to input image data')
    parser.add_argument('--checkpointpath', help='path to input data')
    parser.add_argument(
        '--seed', help='random seed for everything', default=134,
    )
    parser.add_argument(
        '--image_size', help='image size (lenght of side of square)',
        default=50,
    )
    parser.add_argument(
        '--max_epochs', help='number of training epochs', default=1,
    )
    parser.add_argument(
        '--learning_rate',
        help='learning rate for training', default=1e-4,
    )
    parser.add_argument(
        '--outcome',
        help='outcome to train to, contemporaenous vs future HAQ',
        default='HAQ',
    )

    parser.add_argument(
        '--handsorfeet',
        help='outcome to train to, contemporaenous vs future HAQ',
        default=None,
    )
    args = parser.parse_args()
    return args


def main():
    """Takes input arg paths and train models based on parameter
    """
    args = arg_parse_train()
    image_path = Path(args.imagepath)
    checkpoint_path = Path(args.checkpointpath)
    csv_path = Path(args.csvpath)

    if args.outcome == 'HAQ':
        outcome = 'HAQ'
        df = process_dataframe(
            csv_path / 'train_data_HAQ.csv', image_path, outcome, args.handsorfeet,
        )
    if args.outcome == 'Future_HAQ':
        outcome = 'HAQ'
        df = process_dataframe(
            csv_path / 'train_data_future_HAQ.csv', image_path, outcome, args.handsorfeet,
        )

    print(f'SEED: {args.seed}')
    print(f'IMAGE SIZE: {args.image_size}')
    print(f'EPOCHS: {args.max_epochs}')
    print(f'LR: {args.learning_rate}')
    print(f'INPUT DATA PATH: {image_path}')
    print(f'OUTCOME: {args.outcome}, {args.handsorfeet}')
    print(f'>>>>>>>>> Training to outcome: {args.outcome} <<<<<<<<<<')
    print()

    train_model(
        image_path=image_path,
        data=df,
        outcome_train=args.outcome,
        handsorfeet=args.handsorfeet,
        outcome=outcome,
        checkpoint_path=checkpoint_path,
        seed=int(args.seed),
        image_size=int(args.image_size),
        max_epochs=int(args.max_epochs),
        learning_rate=float(args.learning_rate),
    )


if __name__ == '__main__':
    main()
