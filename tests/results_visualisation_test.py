from __future__ import annotations

from pathlib import Path

import pandas as pd
from predict_haq.results_visualisation import binarise_outcome
from predict_haq.results_visualisation import get_filename
from predict_haq.results_visualisation import get_roc_ci
from predict_haq.results_visualisation import get_title
from predict_haq.results_visualisation import prep_df_results
from predict_haq.results_visualisation import round_list


def get_dummy_data():
    Patient_IDs = [1, 1, 2, 3, 3, 3, 4, 5, 6, 6]
    HAQ = [0.25, 0, 0, 0.125, 0.75, 2, 0, 0.125, 1.125, 0]
    scores = [5, 3, 20, 21, 102, 0, 1, 3, 0, 164]
    df = pd.DataFrame({'Patient_IDs': Patient_IDs, 'HAQ': HAQ, 'scores': scores})
    df = pd.concat([df]*10)
    return df


def test_binarise_outcome():
    df = get_dummy_data()
    Y = binarise_outcome(df['HAQ'], 0)
    assert Y == [1, 0, 0, 1, 1, 1, 0, 1, 1, 0]*10


def test_round_list():
    lst = [0.542, 0.651]
    rounded_lst = round_list(lst, 1)
    assert rounded_lst == [0.5, 0.7]


def test_get_roc_ci():
    df = get_dummy_data()

    dict_results = get_roc_ci(
        dataframe=df,
        x_name='scores',
        y_name='HAQ',
        legend_name='legend',
        plt_color='blue',
        thresh=0,
        title='Title',
        plot=False,
        n_bootstraps=1000,
        random_state=123,
    )

    assert 0.35 < dict_results['AUC mean'] < 0.45


def test_get_filename():
    svdh_filename, ai_filename = get_filename('hands', '')

    assert svdh_filename == 'hands_SVDH_HAQ.csv'
    assert ai_filename == 'hands_HAQ_AI.csv'


def test_get_title():
    title = get_title('HAQ_change')

    assert title == '1-2 year HAQ change'


def test_prep_df_results():
    dict_results_ai = {
        'Result': 'Test_result_ai',
        'AUC mean': 0.5,
        '95% CI': '[0.4, 0.6]',
    }
    dict_results_svdh = {
        'Result': 'Test_result_svdh',
        'AUC mean': 0.4,
        '95% CI': '[0.45, 0.55]',
    }
    results_df = prep_df_results(dict_results_ai, dict_results_svdh, Path('Pathtonowhere/'))

    assert len(results_df) == 2
    assert 'Test_result_ai' in results_df['Result'].values
    assert 'Test_result_svdh' in results_df['Result'].values
    assert results_df[results_df['Result'] == 'Test_result_ai']['AUC mean'].values[0] == 0.5
    assert results_df[results_df['Result'] == 'Test_result_svdh']['AUC mean'].values[0] == 0.4
