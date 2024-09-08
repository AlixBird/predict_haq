"""This is a test"""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd


def process_dataframe(
    csv_path: Path,
    image_path: str,
    outcome: str,
) -> pd.DataFrame:
    """Load and process dataframes

    Parameters
    ----------
    csv_path : Path
        path to csv
    image_path: Path
        path to images
    train_or_test : str
        whether using train or test csv
    outcome: str
        which outcome, HAQ, pain, SF36

    Returns
    -------
    pd.DataFrame
        data with duplicates removed and kept rows with xrays available
    """
    df = pd.read_csv(csv_path)

    if 'Filenames' not in df:
        df['Filenames'] = df['UID'] + '.png'

    # Drop duplicates
    df = df.drop_duplicates(subset=['Filenames']).reset_index(drop=True)
    # Keep rows that have a corresponding xray image

    df = df[[
        os.path.isfile(str(image_path) + '/' + str(i))
        for i in df['Filenames']
    ]]

    # Drop NAs for outcome
    df = df.dropna(
        subset=[outcome, 'Filenames'],
    ).reset_index(drop=True)

    # Remove values above certain range
    df = df[df[outcome] <= 100]
    # Normalise outcome
    df[outcome] = df[outcome]/16
    df['date_of_visit'] = pd.to_datetime(df['date_of_visit'])
    df = df.sort_values(
        by=['Patient_ID', 'date_of_visit'],
    ).reset_index(drop=True)
    return df
