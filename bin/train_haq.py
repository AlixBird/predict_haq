"""Train model.

Trains deep learning model with PytorchLightning to predict
function in rheumatoid arthritis
"""
from __future__ import annotations

import argparse
from pathlib import Path

from predict_haq.preprocessing import process_dataframe
from predict_haq.results_visualisation import plot_both_rocs
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
    parser.add_argument('--figurepath', help='path to save data for figures')
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
    Produces roc plots of HAQ prediction comparing performance
    of the SvdH score and the AI algorithm
    """
    # Parse args
    args = arg_parse_train()
    # Convert args to path objects
    image_path = Path(args.imagepath)
    figures_path = Path(args.figurepath)
    checkpoint_path = Path(args.checkpointpath)
    csv_path = Path(args.csvpath)

    # Define dataframes depending on outcome selected
    if args.outcome == 'HAQ':
        outcome = 'HAQ'
        df = process_dataframe(
            csv_path / 'train_data_HAQ.csv', image_path, outcome, args.handsorfeet,
        )
        test_df = process_dataframe(
            csv_path / 'test_data_HAQ.csv', image_path, outcome, args.handsorfeet,
        )
    if args.outcome == 'Future_HAQ':
        outcome = 'HAQ'
        df = process_dataframe(
            csv_path / 'train_data_future_HAQ.csv', image_path, outcome, args.handsorfeet,
        )
        test_df = process_dataframe(
            csv_path / 'test_data_future_HAQ.csv', image_path, outcome, args.handsorfeet,
        )

    # Print params selected
    print(f'SEED: {args.seed}')
    print(f'IMAGE SIZE: {args.image_size}')
    print(f'EPOCHS: {args.max_epochs}')
    print(f'LR: {args.learning_rate}')
    print(f'INPUT DATA PATH: {image_path}')
    print(f'OUTCOME: {args.outcome}, {args.handsorfeet}')
    print(f'>>>>>>>>> Training to outcome: {args.outcome} <<<<<<<<<<')
    print()

    # Train model
    train_model(
        image_path=image_path,
        figures_path=figures_path,
        data=df,
        test_data=test_df,
        outcome_train=args.outcome,
        handsorfeet=args.handsorfeet,
        outcome=outcome,
        checkpoint_path=checkpoint_path,
        seed=int(args.seed),
        image_size=int(args.image_size),
        max_epochs=int(args.max_epochs),
        learning_rate=float(args.learning_rate),
    )

    # Save ROC plots to figures folder
    plot_both_rocs(args.handsorfeet, args.outcome, figures_path)


if __name__ == '__main__':
    main()
