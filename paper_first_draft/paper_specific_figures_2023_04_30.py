"""This is a one-off analysis script for generating custom plots used in the first paper submission from
data previously generated by `query_and_analyze.py`.
"""

import sys
# Add repo to path
sys.path.append('../')

from query_and_analyze_ces import get_data, load_query_config, transform_data
from utils import setup_logging

LOGGER = setup_logging('LOGS/paper_specific_figures_2023_04_30')

# Path to previously generated datafile
QUERY_CONFIG = '../configs/full_config_2023_04_22.json'
GPT_DATA_FILE = 'gpt_api_output_2023_04_24_22-42-26.jsonl.zip'

def main():
    LOGGER.info('Begin paper_specific_figures_2023_04_30')
    base_prompt, demos, ces_policies, n_per_query, query_repeats, previous_query_file = load_query_config(QUERY_CONFIG)
    outfile, ces_df, gpt_df, issue_map, issue_map_invert, prompt_options = get_data(
        base_prompt, demos, ces_policies, n_per_query, query_repeats, GPT_DATA_FILE)
    
    ces_df_plot, gpt_df_plot, combined_df, demos_options, demos_options_with_dataset = transform_data(
        gpt_df, ces_df, demos, issue_map, prompt_options)
    
    LOGGER.info('Make the plots')

    
    LOGGER.info('End paper_specific_figures_2023_04_30')
    breakpoint()

if __name__ == '__main__':
    main()