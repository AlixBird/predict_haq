"""This is a test"""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd


def process_dataframe(path_to_data: Path, train_or_test: str):
    """Load and process dataframes

    Parameters
    ----------
    path_to_data : Path
        path to data
    train_or_test : str
        whether using train or test csv

    Returns
    -------
    pd.DataFrame
        data with duplicates removed and kept rows with xrays available
    """
    csv_name = 'xray_' + train_or_test + '.csv'
    df = pd.read_csv(path_to_data / 'dataframes' / csv_name)

    # Drop duplicates
    df = df.drop_duplicates(subset=['UID']).reset_index(drop=True)

    # Keep rows that have a corresponding xray image
    df = df[[
        os.path.isfile(path_to_data / 'xray_images' / str(i))
        for i in df['Filenames']
    ]]

    # Still need to remove values above certain range
    return df
