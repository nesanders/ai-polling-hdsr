"""This script wraps together several other scripts to accomplish the following,

* Parse CES data to extract data on a set of policies
* Query the OpenAI API to generate chat responses that simulate artificial polling of demographics across those policies
* Analyze the resulting data, comparing the GPT results to the real people from the CES data

This script has a CLI. Use `python query_and_analyze_ces.py --help` for documentation.

See the definition of `QUERY_CONFIG_SCHEMA` for guidance on the json config file format and options.
"""

from copy import deepcopy
import json
import jsonschema
from pathlib import Path
from typing import Optional

import click
import pandas as pd
import seaborn as sns

from analysis import bar_plot, hist_plot, load_gpt_data, scatter_plot
from openai_api_query import _get_model_params, OUTPUT_DIR, PromptOptionsType, main as query_main
from parse_ces import CES_DEMOS, RetainPoliciesType, get_ces_data
from utils import START_TIME, invert_dict, setup_logging

LOGGER = setup_logging('LOGS/query_and_analyze_ces')

QUERY_CONFIG_SCHEMA = {
    "type" : "object",
    "properties": {
        "BASE_PROMPT" : {
            "type" : "string",
            "description": "The prompt text for the GPT queries, with python "
                           "formattable string placeholders matching the demographic "
                           "fields from PROMPT_OPTIONS.",
        },
        "DEMOS": {
            "type": "array",
            "items": {
                "type": "string"
            },
            "description": "List of demos; must be a subset of `CES_DEMOS.keys()`"
        },
        "CES_POLICIES": {
            "type": "object",
            "description": "Policies to pull from the CES data and query GPT for.",
            "additionalProperties": {
                "type": "object",
                "items": {
                    "type": "object",
                    "code": {
                        "type": "string",
                        "description": "Code number for a question, used for lookups in CES data."
                    },
                    "prompt": {
                        "type": "string",
                        "description": "Prompt text used in CES survey."
                    }
                }
            }
        },
        "QUERY_REPEATS": {
            "type": "integer",
            "description": "How many prompt completions to run for each demographic bin; by default 1."
        },
        "N_PER_QUERY": {
            "type": "integer",
            "description": "How many prompt completions to run per query; by default 10."
        },
        "PREVIOUS_QUERY_FILE": {
            "type": "string",
            "description": "Optional; if specified, load a previous GPT query output from this file "
                "rather than re-querying. If this points to a partially-downloaded file, then any "
                "prompts not yet queried for will be completed (it will pick up querying where it "
                "left off)."
        },
    },
    "REQUIRED": [
        "BASE_PROMPT",
        "DEMOS",
        "CES_POLICIES"
    ]
}

def normalize_levels(source: str) -> str:
    """Normalize policy issue response levels from CES data by making certain transformations.
    """
    source = source.strip().lower()
    if source == 'selected':
        output = 'strongly agree'
    elif source == 'not selected':
        output = 'strongly disagree'
    else:
        output = source
    return output

IssueMapType = dict[str, str | dict[str, str]]

def get_prompt_options(demos: list[str], ces_df: pd.DataFrame, ces_policies: RetainPoliciesType
) -> tuple[PromptOptionsType, IssueMapType]:
    """Extract a set of prompt options from the demographic data of a CES DataFrame.
    
    Returns a tuple of three dicts; the first has the full prompt, suitable for GPT querying,
    the second is a dictionary mapping the full prompts to their shortened names, and the third
    is a mapping of issues to their low/high response levels
    """
    prompt_options = {}
    # Lookup demos
    for demo in demos:
        if ces_df[demo].dtype == 'category':
            prompt_options[demo] = ces_df[demo].cat.categories.to_list()
        else:
            prompt_options[demo] = ces_df[demo].unique().tolist()
    # Construct issues
    issue_map = {ces_policies[issue]['prompt']: issue for issue in ces_policies}
    # Lookup response values
    issues_dicts = []
    for issue in ces_policies:
        an_issue = {}
        an_issue['issue'] = ces_policies[issue]['prompt']
        an_issue['cardinality'] = len(ces_df[issue].cat.categories)
        an_issue['low_level'] = normalize_levels(ces_df[issue].cat.categories[0])
        an_issue['high_level'] = normalize_levels(ces_df[issue].cat.categories[-1])
        issues_dicts.append(an_issue)
    prompt_options['issue'] = issues_dicts
    return prompt_options, issue_map

def load_query_config(config_path: Path) -> tuple[str, list[str], RetainPoliciesType, int, int, Optional[str]]:
    """Parse the json config file and return the base_query, ces_demos, ces_policies, query_repeats, and previous_query_file
    """
    with open(config_path, 'r') as f:
        config = json.load(f)
    jsonschema.validate(instance=config, schema=QUERY_CONFIG_SCHEMA)
    for key in config['DEMOS']:
        assert key in CES_DEMOS, "You requested a demo in `DEMO` that doesn't exist in CES data."
        assert f'{{{key}}}' in config['BASE_PROMPT'], "You requested demo {demo}, but it does not exist in the base prompt"
    if 'PREVIOUS_QUERY_FILE' not in config:
        config['PREVIOUS_QUERY_FILE'] = None
    else:
        config['PREVIOUS_QUERY_FILE'] = Path(config['PREVIOUS_QUERY_FILE'])
    if 'N_PER_QUERY' not in config:
        config["N_PER_QUERY"] = 10
    if 'QUERY_REPEATS' not in config:
        config["QUERY_REPEATS"] = 1
    
    return (config['BASE_PROMPT'], config['DEMOS'], config['CES_POLICIES'], config['N_PER_QUERY'], config['QUERY_REPEATS'], 
        config['PREVIOUS_QUERY_FILE'])

def get_data(
    base_prompt: str, 
    demos: list[str], 
    ces_policies: RetainPoliciesType, 
    n_per_query: int, 
    query_repeats: int, 
    previous_query_file: Optional[Path | str]
) -> tuple[pd.DataFrame, pd.DataFrame, IssueMapType, dict[str, str], PromptOptionsType]:
    """Load CES and AI polling data, either from API queries or a previously saved file.
    """
    LOGGER.info('Load the data')
    ces_df = get_ces_data(demos=demos, policies=ces_policies)
    prompt_options, issue_map = get_prompt_options(demos, ces_df, ces_policies)
    issue_map_invert = invert_dict(issue_map)
    
    outfile, _, _ = query_main(base_prompt, prompt_options, _get_model_params(n_per_query), query_repeats=query_repeats, previous_query_file=previous_query_file)
    gpt_df = load_gpt_data(outfile)
    
    return outfile, ces_df, gpt_df, issue_map, issue_map_invert, prompt_options

def transform_data(gpt_df: pd.DataFrame, ces_df: pd.DataFrame, demos: list[str], issue_map: IssueMapType, prompt_options: PromptOptionsType
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, list[str]], dict[str, list[str]]]:
    """Transform loaded polling data to prepare for plotting.
    """
    LOGGER.info('Transform the data')
    gpt_df_plot = gpt_df.copy()
    gpt_df_plot['issue'] = gpt_df_plot['issue'].map(issue_map)
    ces_df_plot = ces_df.copy()
    for issue in issue_map.values():
        # Note - we add 1 to the codes so they start at 1, consistent with the prompt template
        ces_df_plot[issue] = ces_df_plot[issue].cat.codes.values + 1
    ces_df_plot = pd.melt(ces_df_plot, id_vars=demos, value_vars=list(issue_map.values()), 
                          var_name='issue', value_name='score')
    demos_options = {
        demo: prompt_options[demo] for demo in prompt_options 
        if demo not in ['issue', 'cardinality', 'low_level', 'high_level']
    }
    
    combined_df = pd.concat({'gpt': gpt_df_plot, 'ces': ces_df_plot}, names=['dataset']).reset_index()
    # We want to put 'dataset' first so it will show up as the bottom-level comparator in bar charts
    demos_options_with_dataset = {}
    demos_options_with_dataset['dataset'] = ['gpt', 'ces']
    for key, val in demos_options.items():
        demos_options_with_dataset[key] = val
    
    
    return ces_df_plot, gpt_df_plot, combined_df, demos_options, demos_options_with_dataset

@click.command()
@click.option(
    '-c', '--query_config', 
    required=True, 
    type=click.Path(), 
    help='Path to a json config file containing the query config matching QUERY_CONFIG_SCHEMA', 
    prompt=True)
def main(query_config: Path):
    """Parse CES data, query 
    """
    LOGGER.info('Begin query_and_analyze')
    base_prompt, demos, ces_policies, n_per_query, query_repeats, previous_query_file = load_query_config(query_config)
    outfile, ces_df, gpt_df, issue_map, issue_map_invert, prompt_options = get_data(
        base_prompt, demos, ces_policies, n_per_query, query_repeats, previous_query_file)
    out_csv = (OUTPUT_DIR / outfile.name).with_suffix('.csv.gz')
    gpt_df.to_csv(out_csv)
    
    ces_df_plot, gpt_df_plot, combined_df, demos_options, demos_options_with_dataset = transform_data(
        gpt_df, ces_df, demos, issue_map, prompt_options)
    
    LOGGER.info('Make the plots')
    for issue in issue_map.values():
        LOGGER.info(f'Plot the datasets alone for {issue}')
        bar_plot(gpt_df_plot[gpt_df_plot['issue'] == issue], demos=demos_options, filename_prefix=f'gpt_{issue}',
            title=issue_map_invert[issue], plot_func=sns.pointplot, timestamp=START_TIME)
        bar_plot(ces_df_plot[ces_df_plot['issue'] == issue], demos=demos_options, filename_prefix=f'ces_{issue}',
            title=issue_map_invert[issue], plot_func=sns.pointplot, timestamp=START_TIME)
        
        LOGGER.info(f'Plot the datasets together for {issue}')
        bar_plot(combined_df[combined_df['issue'] == issue], demos=demos_options_with_dataset, filename_prefix=f'combined_{issue}',
            title=issue_map_invert[issue], plot_func=sns.pointplot, timestamp=START_TIME)
        
    scatter_plot(combined_df, group_col='dataset', x_group='ces', y_group='gpt', val_col='score', filename_prefix='combined', 
        hue_col='issue', cats=list(demos_options.keys()), timestamp=START_TIME)
    for issue in issue_map.values():
        scatter_plot(combined_df[combined_df['issue'] == issue], group_col='dataset', x_group='ces', y_group='gpt', val_col='score', 
            filename_prefix=f'combined_ALL_{issue}', hue_col='issue', cats=list(demos_options.keys()), all_only=True, 
            timestamp=START_TIME)
    for cat in list(demos_options.keys()):
        scatter_plot(combined_df, group_col='dataset', x_group='ces', y_group='gpt', val_col='score', filename_prefix=f'combined_{cat}', 
            hue_col='issue', cats=[cat], timestamp=START_TIME)
        for issue in issue_map.values():
            scatter_plot(combined_df[combined_df['issue'] == issue], group_col='dataset', x_group='ces', y_group='gpt', val_col='score', 
                filename_prefix=f'combined_ALL_{issue}_{cat}', hue_col='issue', cats=[cat], all_only=True, timestamp=START_TIME)
    
    hist_plot(combined_df, row='ideology', col='issue', filename_prefix='byissue_byideology', row_order=demos_options['ideology'], 
        timestamp=START_TIME)
    hist_plot(combined_df, row='gender', col='issue', filename_prefix='byissue_bygender', timestamp=START_TIME)
    for issue in issue_map.values():
        hist_plot(combined_df[combined_df['issue'] == issue], row=None, col='ideology', filename_prefix=issue,
            col_order=demos_options['ideology'], timestamp=START_TIME)
    
    LOGGER.info('End query_and_analyze')

if __name__ == '__main__':
    main()
