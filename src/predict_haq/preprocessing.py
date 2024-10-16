"""Preprocess data to remove NaNs, impossible values, select hands or feet data"""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd


def process_dataframe(
    csv_path: Path,
    image_path: str,
    outcome: str,
    handsorfeet: str,
) -> pd.DataFrame:
    """Load and process dataframes

    Parameters
    ----------
    csv_path : Path
        path to csv
    image_path: Path
        path to images
    outcome: str
        which outcome, HAQ, pain, SF36
    handsorfeet: str
        whether to select hands or feet for training

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
    df = df[df[outcome] <= 18]

    df[outcome] = (df[outcome] - df[outcome].mean())/df[outcome].std()

    df['date_of_visit'] = pd.to_datetime(df['date_of_visit'])
    df = df.sort_values(
        by=['Patient_ID', 'date_of_visit'],
    ).reset_index(drop=True)

    # Select rows that are hands or feet xrays
    if handsorfeet:
        if df['Category_hvf'].isna().sum() > 0:
            df = df[df['Baseline_xray_category'] == handsorfeet]
        else:
            df = df[df['Category_hvf'] == handsorfeet]
    return df


def process_dataframe_haq_change(
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
    outcome: str
        which outcome, HAQ, pain, SF36
    handsorfeet: str
        whether to select hands or feet for training

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

    # Remove values above certain range
    df = df[df[outcome] <= 18]

    df[outcome] = (df[outcome] - df[outcome].mean())/df[outcome].std()

    return df
