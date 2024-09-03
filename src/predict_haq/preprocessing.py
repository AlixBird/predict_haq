"""This is a test"""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd


def process_dataframe_future_haq(
    outcome: str,
    csv_path: Path,
    image_path: Path,
) -> pd.DataFrame:
    """Makes future function dataframe.

    Processes full dataframe to get baseline xray alongside future HAQ.

    Parameters
    ----------
    outcome : str
        HAQ
    csv_path : Path
        path to dataframe
    image_path : Path
        path to images

    Returns
    -------
    pd.DataFrame
        Modified dataframe with future HAQ
    """

    df = pd.read_csv(csv_path)

    # Keep rows that have a corresponding xray image
    df = df[df[outcome] <= 16]

    df = df[[
        os.path.isfile(str(image_path) + '/' + str(i))
        for i in df['Filenames']
    ]]

    df = df.dropna(subset=[outcome])

    df['date_of_visit'] = pd.to_datetime(df['date_of_visit'])
    df = df.sort_values(by=['Patient_ID', 'date_of_visit'])

    new_df = pd.DataFrame()

    for _, group in df.groupby('Patient_ID'):
        # group with Filenames for baseline xray
        day_1_data = group.dropna(subset='Filenames').reset_index(drop=True)

        # group with HAQ for future function
        timepoint_2_data = group.dropna(subset=outcome).reset_index(drop=True)

        # If there is data available for the baseline xray
        run_loop = True
        if len(day_1_data) > 0:
            # Get the date of the first xray
            day_1_date = day_1_data['date_of_visit'][0]

            # Dataframe for HAQ to iterate through possible dates
            dates = timepoint_2_data.reset_index(drop=True)

            # Iterate through dates
            diff_days = []
            for _, row in dates.iterrows():

                # Date for each row
                i = row['date_of_visit']

                # Days different between first xray and the row
                diff = (i - day_1_date).days

                diff_days.append(diff)
                # Select first row that is greater than 1 year

                if run_loop:
                    if diff > 365:
                        # Row is the future data
                        new_row = row

                        # Add to this row the difference in days
                        new_row['First_ever_appt'] = diff_days[0]

                        # Addd to this row the Filename from baseline
                        new_row['baselinexr'] = day_1_data['Filenames'][0]

                        # Add baseline HAQ to the row
                        new_row['Baseline_HAQ'] = day_1_data['HAQ'][0]

                        # add to this row the days different
                        new_row['Days_since_first_xray'] = diff

                        new_df = pd.concat(
                            [new_df, new_row], axis=1, ignore_index=True,
                        )
                        run_loop = False

    new_df = new_df.transpose()
    new_df = new_df.dropna(subset=['HAQ', 'Baseline_HAQ'])
    new_df['Change_HAQ'] = (new_df['HAQ'] - new_df['Baseline_HAQ'])/16
    new_df = new_df[new_df['Days_since_first_xray'].between(365, 730)]
    new_df = new_df.drop_duplicates(subset='baselinexr')

    # Drop duplicates
    new_df = new_df.drop_duplicates(
        subset=['Filenames'],
    ).reset_index(drop=True)

    return new_df


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
