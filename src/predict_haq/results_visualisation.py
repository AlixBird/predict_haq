from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import pandas as pd
from roc_utils import compute_mean_roc
from roc_utils import compute_roc_bootstrap
from roc_utils import plot_mean_roc

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
plt.style.use('ggplot')


def binarise_outcome(data: list, thresh: float) -> list:
    """Binarises list at a specified threshold

    Parameters
    ----------
    data : list
        list of floats
    thresh : float
        threshold to dichotomise at

    Returns
    -------
    list
        list of dichotomised data
    """
    return [1 if i > thresh else 0 for i in data]


def round_list(lst: list, k: int) -> list:
    """Round a list of floats

    Parameters
    ----------
    lst : list
        list
    k : int
        how many decimals to round to

    Returns
    -------
    list
        list of rounded values
    """
    return [round(float(i), k) for i in lst]


def get_roc_ci(
    dataframe: pd.DataFrame,
    x_name: str,
    y_name: str,
    legend_name: str,
    plt_color: str,
    thresh: float,
    title: str,
    plot: bool,
    n_bootstraps: int,
    random_state: int,
) -> dict:
    """Get ROCs, mean AUC and confidence intervals

    Parameters
    ----------
    dataframe : pd.DataFrame
        data of predicted values and actual HAQ
    x_name : str
        name of column for x data
    y_name : str
        name of column for y data
    legend_name : str
        legend description for plot
    plt_color : str
        color for plot
    thresh : float
        threshold to dichotomise outcome
    title : str
        plot title
    plot : bool
        whether to save plot to file
    n_bootstraps : int
        number of bootstraps to calc 95% CI
    random_state : int
        seed

    Returns
    -------
    dict
        dictionary of mean auc and 95% CI
    """
    # Convert outcome to binary
    Y = binarise_outcome(dataframe[y_name], thresh)

    # Bootstrap rocs
    rocs = compute_roc_bootstrap(
        X=dataframe[x_name], y=Y, pos_label=1,
        n_bootstrap=n_bootstraps,
        random_state=random_state,
        return_mean=False,
    )

    if plot:
        ret_mean = plot_mean_roc(
            rocs, show_ci=False, show_ti=True,
            color=plt_color, auto_flip=False, label=legend_name,
        )
    else:
        ret_mean = compute_mean_roc(rocs, auto_flip=False)

    dict_results = {
        'Result': title + legend_name,
        'AUC mean': round(ret_mean['auc_mean'], 4),
        '95% CI': str(round_list(ret_mean['auc95_ti'][0], 4)),
    }

    return dict_results


def get_filename(handsorfeet: str, outcome: str) -> Tuple[str, str]:
    """Gets csv filename depending on input parameters

    Parameters
    ----------
    handsorfeet : str
        whether hands or feet xrays
    outcome : str
        Contemporaneous haq, future or change

    Returns
    -------
    Tuple[str,str]
        Returns filename for svdh data, and ai data
    """
    if 'Future' in outcome or 'change' in outcome:
        svdh_filename = handsorfeet+'_SVDH_Future_HAQ.csv'
        ai_filename = handsorfeet+'_Future_HAQ_AI.csv'
    else:
        svdh_filename = handsorfeet+'_SVDH_HAQ.csv'
        ai_filename = handsorfeet+'_HAQ_AI.csv'
    return svdh_filename, ai_filename


def get_title(outcome: str) -> str:
    """Get plot title

    Parameters
    ----------
    outcome : str
        Contemporaneous, future or change

    Returns
    -------
    str
        Title for plot
    """
    if outcome == 'HAQ':
        title = 'Contemporaneous HAQ'
    if outcome == 'Future_HAQ':
        title = '1-2 year HAQ'
    if outcome == 'HAQ_change':
        title = '1-2 year HAQ change'
    return title


def plot_rocs(hands_or_feet: str, outcome: str, figures_path: Path, title: str):
    """Saves plot to file

    Parameters
    ----------
    hands_or_feet : str
        Whether hands or feet, for plot title
    outcome : str
        Whether future, change or contemp, for plot title
    figures_path : Path
        _description_
    title : str
        _description_
    """
    plt.legend(fontsize=7, loc='lower right')
    plt.title(f'{title} - {hands_or_feet}')
    pngname = hands_or_feet+outcome+'.png'
    plt.savefig(figures_path / pngname, dpi=300)
    plt.close()


def prep_df_results(dict_results_ai: dict, dict_results_svdh: dict, path_df: Path) -> pd.DataFrame:
    """Combines results dicts into a dataframe

    Parameters
    ----------
    dict_results_ai : dict
        Results from AI
    dict_results_svdh : dict
        Results from human svdh
    path_df : Path
        Path to data

    Returns
    -------
    pd.DataFrame
        resulting dataframe of mean AUC, 95% CIs
    """
    # Save results as dataframe
    data_rows = pd.DataFrame.from_dict([dict_results_ai, dict_results_svdh])

    # Check if csv already exists
    # If so append the new data
    if os.path.isfile(path_df):
        results_df = pd.read_csv(
            path_df, index_col=False,
        ).reset_index(drop=True)
        results_df = pd.concat([results_df, data_rows], ignore_index=True)
    else:
        # If csv doesn't exist create new dataframe
        results_df = data_rows
    # Keep latest result if there are multiple of same result
    results_df = results_df.drop_duplicates(subset='Result', keep='last')
    return results_df


def plot_ai_vs_human_rocs(
        hands_or_feet: str,
        outcome: str,
        figures_path: Path,
        thresh: float,
        plot: bool = False,
        n_bootstraps: int = 10000,
        random_state: int = 2808,
):
    """Plots ROCs and 95% CIs comparing AI vs human performance at
    function prediction as measured by the HAQ

    Parameters
    ----------
    hands_or_feet : str
        Hand or feet xrays
    outcome : str
        Contemporaneous haq, future or change
    figures_path : Path
        Path to where figures are saved
    thresh : float
        Dichotomisting threshold for ROC construction
    plot : bool, optional
        Whether to save plot to file, by default False
    n_bootstraps : int, optional
        Num of bootstraps when calculating 95%CIs, by default 10000
    random_state : int, optional
        Seed, by default 2808
    """
    # Set csv filename to read in based on if Contemporaneous or future HAQ
    #  and hands or feet xrays
    logger.info(f'Processing {hands_or_feet} with outcome {outcome}')
    svdh_filename, ai_filename = get_filename(hands_or_feet, outcome)

    # Read in csvs
    human = pd.read_csv(figures_path / svdh_filename)
    ai = pd.read_csv(figures_path / ai_filename)

    # Get plot title
    title = get_title(outcome)

    # Plot ROC for AI
    dict_results_ai = get_roc_ci(
        ai,
        'Preds',
        'Targets',
        f'AI (n xrays={len(ai)}, pts={str(ai["Patient_ID"].nunique())})',
        plt_color='#377eb8',
        thresh=thresh,
        title=title,
        plot=plot,
        n_bootstraps=n_bootstraps,
        random_state=random_state,
    )

    # Plot ROC for svdh
    dict_results_svdh = get_roc_ci(
        human,
        'final_score',
        'HAQ',
        f'SvdH scores (n xrays={len(human)}, pts={str(human["Patient_ID"].nunique())})',
        plt_color='#ff7f00',
        thresh=thresh,
        title=title,
        plot=plot,
        n_bootstraps=n_bootstraps,
        random_state=random_state,
    )

    # Save ROCs as plots in folder specified by figures_path
    if plot:
        plot_rocs(hands_or_feet, outcome, figures_path, title)

    # Save results in dataframe
    results_df = prep_df_results(
        dict_results_ai, dict_results_svdh,
        figures_path / 'auc_results.csv',
    )
    results_df.to_csv(figures_path / 'auc_results.csv', index=False)
