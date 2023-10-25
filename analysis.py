"""Generate plots and analysis from the output of ``chatgpt_query.py`.

The output of this script is the following set of PNG files written to the current working directory,
* 
"""

from copy import deepcopy
from itertools import product
import json
from pathlib import Path
import re
from typing import Any, Callable, Optional
import zipfile

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, wasserstein_distance
import seaborn as sns
from seaborn._statistics import EstimateAggregator

from parse_ces import get_ces_data
from utils import QUERY_OUTPUT_DIR, PROMPT_OPTIONS, START_TIME, PromptOptionsType, setup_logging

LOGGER = setup_logging('LOGS/analysis')

# Pointer to the output of a run of `chatgpt_query.py`
QUERY_TIMESTAMP = '23_04_08_20-28-41'
QUERY_RESULT_JSON = QUERY_OUTPUT_DIR / f'gpt_api_output_{QUERY_TIMESTAMP}.jsonl'

PROMPT_OPTIONS.pop('issue')

PLOT_DIR = Path('PLOTS')

sns.color_palette("Set2")

# -----------------------------
# Function defs
# -----------------------------

def mean_absolute_percentage_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate the mean absolute percent error; output is a percentage.
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true))
    return mape * 100

def process_json_line(source: str) -> list[dict]:
    """Process one line of a json response file and return a list of response dictionaries.
    """
    responses = []
    id_str = source['response']['id']
    params = source['params']
    query_prompt = source['query'][1]['content']
    for choice in source['response']['choices']:
        response_msg = choice['message']['content']
        score = get_score_response(response_msg)
        cindx = choice['index']
        responses.append({'id': id_str, 'prompt': query_prompt, 'response': response_msg, 'score': score, 'choice_index': cindx})
        responses[-1].update(params)
    return responses

def load_gpt_data(path: Path) -> pd.DataFrame:
    """Load a json file generated by `openai_api_query.py`, which may also be zipped
    """
    responses = []
    if path.suffix == '.zip':
        with zipfile.ZipFile(path, 'r') as z:
            for filename in z.namelist():  
                with z.open(filename) as f:  
                    for line in f:
                        for resp in process_json_line(json.loads(line)):
                            responses.append(resp)
    else:
        with open(path, 'r') as f:
            for line in f:
                for resp in process_json_line(json.loads(line)):
                    responses.append(resp)

    return pd.DataFrame(responses)

REGEXP = r'[^0-9]*Position( score)?:?\s*?[ :]?"?\(?(\d)\)?'
REGEXP_PROG = re.compile(REGEXP, re.IGNORECASE)

def get_score_response(msg: str) -> int:
    """Extract a score scale response from a query response message.
    
    This looks for a pattern matching the following at the start of the `msg` string,

    > Support level: 4
    Or 
    > Support level 4
    """
    return int(REGEXP_PROG.match(msg).groups()[-1])

def get_plot_path(filename: str) -> Path:
    """Get a filename in the plot directory.
    """
    outfile = PLOT_DIR / START_TIME / filename
    outfile.parent.mkdir(parents=True, exist_ok=True)
    return outfile

def extract_singleton_from_df(df: pd.DataFrame, col: str) -> str:
    """Extract a constant value from a DataFrame (`df`) column (`col`), validating that there are no 
    other non-null values in the column.
    """
    vals = df[col].dropna().unique()
    assert len(vals) == 1, f'The DataFrame has zero or multiple values for {col=}'
    return vals[0]

def modify_xtick_label(g: sns.FacetGrid, new_label: str, old_value: Optional[float | int]=None, 
    old_label: Optional[str]=None):
    """Modify a matplotlib `ax` xaxis in place to replace `old_label` with `new_label` for one 
    tick.
    """
    ax = g.axes.flatten()[-1]
    labels = [x.get_text() for x in ax.get_xticklabels()]
    if old_label is not None:
        assert old_label in labels, f'Label {old_label} does not exist on plot'
        pos = labels.index(old_label)
    elif old_value is not None:
        values = ax.get_xticks()
        assert old_value in values, f'Label {old_value} does not exist on plot'
        pos = list(values).index(old_value)
    else:
        raise ValueError("Must specify either an old_label or old_value")
    labels[pos] = new_label
    g.set_xticklabels(labels, rotation=45, ha='right')

def bar_plot(data_df: pd.DataFrame, demos: PromptOptionsType=PROMPT_OPTIONS,
    filename_prefix: Optional[str]=None, plot_func: Callable=sns.barplot, 
    low_label_col: Optional[str]='low_level', high_label_col: Optional[str]='high_level', 
    cardinality_col: Optional[int]='cardinality', title: Optional[str]=None,
    timestamp: str=QUERY_TIMESTAMP,
    **kwargs):
    """Make a simple bar plot of data from `chatgpt_query.py`
    
    `kwargs` are passed to `plot_func`
    """
    if filename_prefix is None:
        filename_prefix = ''
    else:
        filename_prefix += '_'
    outfile = get_plot_path(f'{filename_prefix}bar_plot_{timestamp}')
    
    # Avoid overwriting anything
    copy_df = data_df.copy()
    keys = list(demos.keys())[::-1]
    assert len(keys) > 1, 'Expected at least two query facets'
    copy_df['label'] = copy_df.apply(lambda x: '; '.join(x[keys[2:]]), axis=1)
    
    # Make the plot
    g = sns.FacetGrid(copy_df, col=keys[0], row=keys[1], margin_titles=True, hue=keys[-1] if len(keys)>2 else None,
        row_order=demos[keys[1]], col_order=demos[keys[0]], height=2, aspect=1)
    g.set_ylabels('')
    # Shorten column labels
    g.set_titles(col_template="{col_name}")
    bar_order = list(product(*[demos[key] for i, key in enumerate(keys) if i > 1]))
    bar_order = ['; '.join(b) for b in bar_order]
    g.map_dataframe(plot_func, y="label", x="score", orient='h', order=bar_order, errorbar=('ci', 95), **kwargs)
    g.map(plt.grid, axis='x', zorder=-10, color='0.5')
    plt.gca().set_xlim(1)
    if low_label_col is not None and low_label_col in copy_df:
        tick_label = f'1: {extract_singleton_from_df(copy_df, low_label_col)}'
        plt.gca().xaxis.set_major_locator(plt.FixedLocator(range(1, int(plt.gca().get_xlim()[1]) + 1)))
        modify_xtick_label(g, tick_label, old_value=1)
    if high_label_col is not None and cardinality_col is not None and \
        high_label_col in copy_df and cardinality_col in copy_df:
        cardinality = int(extract_singleton_from_df(copy_df, cardinality_col))
        plt.gca().xaxis.set_major_locator(plt.FixedLocator(range(1, cardinality + 1)))
        tick_label = f'{cardinality}: {extract_singleton_from_df(copy_df, high_label_col)}'
        modify_xtick_label(g, tick_label, old_value=cardinality)
    if title is None:
        title = data_df.name
    g.fig.suptitle(title, wrap=True)
    plt.tight_layout()
    dual_savefig(outfile)
    plt.close()

def dual_savefig(outfile: Path | str):
    """Save the current matplotlib figure to both png and pdf files.
    """
    if isinstance(outfile, str):
        outfile = Path(outfile)
    plt.savefig(outfile.with_suffix('.png'), dpi=300)
    plt.savefig(outfile.with_suffix('.pdf'))

## NOTE - using these functions is super slow, so I have reduced n_boot to 1000 from default of 10k
bootstrap_func = EstimateAggregator('mean', ('ci', 95), n_boot=1000)
def bootstrap_ci(x: pd.Series, return_max: bool) -> pd.DataFrame:
    """Returned bootstrapped 95% confidence interval values for a pandas Series.
    Useful to estimate errors within a pd.GroupBy.
    
    The output represents the difference between the lower 95% CI and the mean if `return_max` is False,
    otherwise the difference from the upper CI.
    """
    boot_result = bootstrap_func(pd.DataFrame({'val': x.values}), 'val')
    if return_max:
        return boot_result['valmax'] - boot_result['val']
    return boot_result['val'] - boot_result['valmin']

def bootstrap_ci_min(x: pd.Series) -> pd.Series: 
    """Return lower 95% CI interval."""
    return bootstrap_ci(x, return_max=False)

def bootstrap_ci_max(x: pd.Series) -> pd.Series: 
    """Return upper 95% CI interval."""
    return bootstrap_ci(x, return_max=True)

def scatter_plot(df: pd.DataFrame, group_col: str, x_group: str, y_group: str, val_col: str, cats: list[str],
    hue_col: str, title: Optional[str]=None, filename_prefix: Optional[str]=None, all_only: bool=False, 
    timestamp: str=QUERY_TIMESTAMP, **kwargs):
    """Compare average results across two groups from column `group_col`, `x_group` and `y_group`, for some value column `val_col`
    in a scatterplot, after aggregating over the column `cats`.
    """
    if filename_prefix is None:
        filename_prefix = ''
    else:
        filename_prefix += '_'
    outfile = get_plot_path(f'{filename_prefix}scatter_plot_{timestamp}')
    
    # Decide parameters of plot
    hue_vals = df[hue_col].unique()
    if all_only is True:
        cats_all = ['ALL']
        n_cols=1
    elif len(cats) == 1:
        cats_all = deepcopy(cats)
        n_cols = 1
    else:
        cats_all = ['ALL'] + cats
        n_cols = len(cats_all)
    
    # Make plot
    agg_funcs = [np.mean, bootstrap_ci_min, bootstrap_ci_max, np.min, np.max]
    if n_cols == 1 and len(hue_vals) == 1:
        # Make singular plot
        fig, ax = plt.subplots(1, 1, figsize=(3, 3))
        axs = np.array([[ax]])
    elif n_cols == 1:
        # Make 1D plots horizontal
        fig, axs = plt.subplots(1, len(hue_vals), figsize=(2 * len(hue_vals), 2))
        axs = np.array([axs]).T
    else:
        fig, axs = plt.subplots(len(hue_vals), n_cols, figsize=(2 * (n_cols), 2 * len(hue_vals)))
    for j, cat_j in enumerate(cats_all):
        if j == 0 or cat_j == 'ALL':
            dfg = df.groupby(cats + [group_col, hue_col])[val_col].agg(agg_funcs)
        else:
            dfg = df.groupby([cat_j, group_col, hue_col])[val_col].agg(agg_funcs)
        axs[0, j].set_title(cat_j)
        for i, hue_val in enumerate(hue_vals):
            color = plt.cm.Paired(i / len(hue_vals))
            x = dfg.xs((hue_val, x_group), level=(hue_col, group_col))['mean']
            y = dfg.xs((hue_val, y_group), level=(hue_col, group_col))['mean']
            x_err_bootstrap_ci = dfg.xs((hue_val, x_group), level=(hue_col, group_col))[['bootstrap_ci_min', 'bootstrap_ci_max']].T
            y_err_bootstrap_ci = dfg.xs((hue_val, y_group), level=(hue_col, group_col))[['bootstrap_ci_min', 'bootstrap_ci_max']].T
            # 45 degree line
            min_x_y = [dfg.xs((hue_val, x_group), level=(hue_col, group_col))['amin'].min() * 0.9] * 2
            max_x_y = [dfg.xs((hue_val, x_group), level=(hue_col, group_col))['amax'].max() * 1.1] * 2
            axs[i, j].plot([min_x_y, max_x_y], [min_x_y, max_x_y], ls='dashed', color='0.5')
            # Errorbar points
            axs[i, j].errorbar(x, y, x_err_bootstrap_ci, y_err_bootstrap_ci, elinewidth=1, c=color, label=hue_val, lw=0, marker='o', **kwargs)
            # Correlation
            if len(x) > 2:
                corr = pearsonr(x, y)[0]
                mape = mean_absolute_percentage_error(x, y)
                axs[i, j].text(0.01, 0.99, f'$\\rho={100 * corr:0.1f}\%$\nMAPE={mape: 0.1f}%', horizontalalignment='left', verticalalignment='top', 
                    transform=axs[i, j].transAxes, fontsize=7,  backgroundcolor=(1, 1, 1, 0.7))
            if j==0: 
                axs[i, j].legend(loc='lower right', fontsize=6)
    if title is not None:
        plt.suptitle(title, wrap=True)
    fig.supxlabel(x_group)
    fig.supylabel(y_group)
    plt.tight_layout(rect=[0.05, 0.05, 1, 1])
    dual_savefig(outfile)
    plt.close()

def wrap_title(ax, width: int=15):
    """Apply word wrap to the title of a maptloltib ax.
    """
    title = ax.get_title()
    ax.set_title('\n'.join(title.split('|')), fontsize=7)

def hist_plot(df: pd.DataFrame, row: Optional[str]=None, col: Optional[str]=None, filename_prefix: Optional[str]=None, 
    timestamp: str=QUERY_TIMESTAMP, **kwargs):
    """Make a hist plot comparing results across datasets.
    
    `kwargs` are passed to `sns.displot`.
    
    TODO add a distributional similarity measure, e.g. https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.chisquare.html
    or https://vasishth.github.io/bayescogsci/book/ch-MPT.html#modeling-multiple-categorical-responses
    """
    if filename_prefix is None:
        filename_prefix = ''
    else:
        filename_prefix += '_'
    outfile = get_plot_path(f'{filename_prefix}hist_plot_{timestamp}')
    
    g = sns.displot(df, x='score', hue='dataset', row=row, col=col, multiple='layer', stat='percent', common_norm=False, 
        height=2, facet_kws=dict(legend_out=True), discrete=True, **kwargs)
    g.map_dataframe(annotate_nemd, 'dataset')
    
    g.set(xlim=(df['score'].min() - 0.25, df['score'].max() + 0.75))
    for ax in g.axes.flatten():
        wrap_title(ax)
    plt.tight_layout()
    dual_savefig(outfile)
    plt.close()

def normalized_emd(x: np.ndarray, y: np.ndarray, norm: Optional[float]=None) -> float:
    """A normalized Wasserstein / earth mover's distance function for a scale corresponding to the 
    range of the input data.
    """
    if norm is None:
        norm = max(max(x), max(y)) - min(min(x), min(y))
    emd = wasserstein_distance(x, y)
    return emd / norm

def annotate_nemd(dataset_col: str, color: Any, data: pd.DataFrame):
    """Annotate a plot with the normalized Earth mover's distance of two distributions
    """
    assert data[dataset_col].nunique() == 2, 'nemd function is only defined for comparing two datasets'
    x_data, y_data = data.groupby(dataset_col)['score'].agg(list)
    cardinality = data['cardinality'].max()
    nemd = normalized_emd(x_data, y_data, norm=cardinality)
    ax = plt.gca()
    ax.text(0.05, 0.8, f'NEMD:\n{nemd:.2f}', transform=ax.transAxes, backgroundcolor=(1, 1, 1, 0.7))

# -----------------------------
# Main logic
# -----------------------------

if __name__ == '__main__':
    gpt_data_df = load_gpt_data(Path(QUERY_RESULT_JSON))
    # Plot the GPT data alone
    gpt_data_df.groupby('issue').apply(bar_plot)
    
    ces_data_df = get_ces_data()
    # Plot the CES data alone
    ces_data_df.groupby('issue').apply(bar_plot)
