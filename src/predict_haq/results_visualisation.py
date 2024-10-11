from __future__ import annotations

import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import roc_utils
plt.style.use('ggplot')


def binarise_outcome(data, thresh):
    return [1 if i > thresh else thresh for i in data]


def round_list(lst, k):
    return [round(float(i), k) for i in lst]


def plot_roc_ci(
    dataframe: pd.DataFrame,
    x_name: str,
    y_name: str,
    legend_name: str,
    plt_color: str,
    thresh: float,
    title: str,
    plot: bool,
):
    # Convert outcome to binary
    Y = binarise_outcome(dataframe[y_name], thresh)

    # Bootstrap rocs
    rocs = roc_utils.compute_roc_bootstrap(
        X=dataframe[x_name], y=Y, pos_label=1,
        n_bootstrap=10000,
        random_state=12,
        return_mean=False,
    )

    if plot:
        ret_mean = roc_utils.plot_mean_roc(
            rocs, show_ci=False, show_ti=True,
            color=plt_color, auto_flip=False, label=legend_name,
        )
    else:
        ret_mean = roc_utils.compute_mean_roc(rocs, auto_flip=False)

    dict_results = {
        'Result': title + legend_name,
        'AUC mean': round(ret_mean['auc_mean'], 4),
        '95% CI': str(round_list(ret_mean['auc95_ti'][0], 4)),
    }

    return dict_results


def plot_both_rocs(
        handsorfeet: str,
        outcome: str,
        figures_path: Path,
        thresh: float,
        plot: bool = False,
):
    """_summary_

    Parameters
    ----------
    handsorfeet : str
        Input data type
    outcome : str
        Whether training to contemporaneous or future HAQ
    figures_path : Path
        Path to where figures ares aved
    """
    # Set filename based on if Contemporaneous or future HAQ
    if 'Future' in outcome or 'change' in outcome:
        svdh_filename = handsorfeet+'_SVDH_Future_HAQ.csv'
        ai_filename = handsorfeet+'_Future_HAQ_AI.csv'
    else:
        svdh_filename = handsorfeet+'_SVDH_HAQ.csv'
        ai_filename = handsorfeet+'_HAQ_AI.csv'

    # Read in results csvs
    human = pd.read_csv(figures_path / svdh_filename)
    ai = pd.read_csv(figures_path / ai_filename)

    # Save figure
    if outcome == 'HAQ':
        title = 'Contemporaneous HAQ'
    if outcome == 'Future_HAQ':
        title = '1-2 year HAQ'
    if outcome == 'HAQ_change':
        title = '1-2 year HAQ change'

    # Plot ROCs for both
    dict_results_ai = plot_roc_ci(
        ai,
        'Preds',
        'Targets',
        f'AI (n xrays={len(ai)}, pts={str(ai["Patient_ID"].nunique())})',
        plt_color='#377eb8',
        thresh=thresh,
        title=title,
        plot=plot,
    )

    dict_results_svdh = plot_roc_ci(
        human,
        'final_score',
        'HAQ',
        f'SvdH scores (n xrays={len(human)}, pts={str(human["Patient_ID"].nunique())})',
        plt_color='#ff7f00',
        thresh=thresh,
        title=title,
        plot=plot,
    )

    if plot:
        plt.legend(fontsize=7, loc='lower right')
        plt.title(f'{title} - {handsorfeet}')
        pngname = handsorfeet+outcome+'.png'
        plt.savefig(figures_path / pngname, dpi=300)

    data_rows = pd.DataFrame.from_dict([dict_results_ai, dict_results_svdh])
    path_df = figures_path / 'auc_results.csv'
    if os.path.isfile(path_df):
        results_df = pd.read_csv(
            path_df, index_col=False,
        ).reset_index(drop=True)

        results_df = pd.concat([results_df, data_rows], ignore_index=True)
    else:
        results_df = data_rows
    results_df = results_df.drop_duplicates(subset='Result', keep='last')

    results_df.to_csv(path_df, index=False)
