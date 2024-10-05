from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import roc_utils
plt.style.use('ggplot')


def plot_roc_ci(
    dataframe: pd.DataFrame,
    x_name: str,
    y_name: str,
    legend_name: str,
    plt_color: str,
):
    # Convert outcome to binary
    Y = [1 if i > 0 else 0 for i in dataframe[y_name]]

    # Bootstrap rocs
    rocs = roc_utils.compute_roc_bootstrap(
        X=dataframe[x_name], y=Y, pos_label=1,
        n_bootstrap=10000,
        random_state=12,
        return_mean=False,
    )

    # Plot mean roc with 95% CI
    roc_utils.plot_mean_roc(
        rocs, show_ci=True, show_ti=True,
        color=plt_color, auto_flip=False, label=legend_name,
    )


def plot_both_rocs(handsorfeet: str, outcome: str, figures_path: Path):
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
    if 'Future' in outcome:
        svdh_filename = handsorfeet+'_svdh_haq_Future_roc.csv'
        ai_filename = handsorfeet+'_Future_HAQ_AI.csv'
    else:
        svdh_filename = handsorfeet+'_svdh_haq_roc.csv'
        ai_filename = handsorfeet+'_HAQ_AI.csv'

    # Read in results csvs
    human = pd.read_csv(figures_path / svdh_filename)
    ai = pd.read_csv(figures_path / ai_filename)

    # Plot ROCs for both
    plot_roc_ci(ai, 'Preds', 'Targets', 'AI performance', plt_color='red')
    plot_roc_ci(human, 'final_score', 'HAQ', 'Human performance', plt_color='blue')

    # Save figure
    plt.title(f'{outcome} {handsorfeet}')
    pngname = handsorfeet+outcome+'.png'
    plt.savefig(figures_path / pngname, dpi=300)
