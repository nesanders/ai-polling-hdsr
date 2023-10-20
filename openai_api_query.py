"""Use the OpenAI API to automate querying GPT models many times across different prompt constructions.

Expects the user's OpenAPI API secret key to be either set as the environment variable `OPENAI_API_KEY` 
or stored in plaintext to the local file `SECRET_openai.txt`.

Output will be saved to `gpt_output_{START_TIME}.csv`
"""

from datetime import datetime
from itertools import product
import json
import os
from pathlib import Path
import shutil
from typing import Any, Optional

import numpy as np
import openai
import pandas as pd
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)

from analysis import load_gpt_data
from utils import (QUERY_OUTPUT_DIR as OUTPUT_DIR, PROMPT_OPTIONS, START_TIME, 
    PromptOptionsType, setup_logging)

LOGGER = setup_logging('LOGS/openai_api_query')

# -----------------------------
# Configurable global variables
# -----------------------------

BASE_PROMPT: str = 'Please write a 1 paragraph letter to the editor from the perspective of a {age} year old {gender} who is politically {ideology} and lives in {state}, US expressing a clear point of view on the policy of, "{issue}". Express their level of support for the policy on a 5 point scale. Before the letter, summarize their support with a "Support level:" statement followed by a single number representing the person\'s level of support on a 5 point scale, where 5 represents the highest level of support for this policy and 1 represents strong disagreement with this policy.'

# The system message for the OpenAI API request
# For details, see https://platform.openai.com/docs/guides/chat/introduction
SYSTEM_MESSAGE: dict[str, str] = {"role": "system", "content": "You are a helpful assistant."}


# -----------------------------
# Non-configurable global variables
# -----------------------------

QueryJobType = list[dict[str, str]]
QueryJobListType = list[dict[str, dict[str, str] | QueryJobType]]

# -----------------------------
# Function defs
# -----------------------------

def _get_model_params(query_repeats: int=10) -> dict[str, str]:
    """Model parameter dictionary to pass to OpenAI API
    For details, see https://platform.openai.com/docs/guides/chat/introduction
    """
    return {'model': 'gpt-3.5-turbo-0301', 'n': query_repeats}

def product_dict(params: dict[Any, list[Any]]) -> list[dict[Any, Any]]:
    """Apply itertools.product to a dictionary, unpacking a dict of lists into a list of dicts.
    """
    keys, values = zip(*params.items())
    output = []
    for bundle in product(*values):
        output.append(dict(zip(keys, bundle)))
    return output

def get_prompt_dicts(query_string: str, query_params: list[dict]) -> QueryJobListType:
    """Generate a list of queries, where each item records the input set of `'params'` as a dictionary 
    and has a query formatted for use with `create_chat_completion` as a `'prompt'`.
    """
    jobs = []
    for params in query_params:
        query = query_string.format(**params)
        prompt = [SYSTEM_MESSAGE, {"role": "user", "content": query + "\n"}]
        jobs.append({'params': params, 'prompt': prompt})
    return jobs

def get_secret(secret_file: Path=Path('SECRET_openai.txt')) -> str:
    """Load an API secret from a file.
    """
    secret = os.getenv("OPENAI_API_KEY")
    if secret is None:
        if not secret_file.exists():
            raise ValueError(f'{secret_file=} does not exist')
        with open(secret_file, 'r') as f:
            secret = f.read().strip()
    return secret

@retry(wait=wait_random_exponential(min=20, max=300), stop=stop_after_attempt(6))
def create_chat_completion(
    params: dict[str, str],
    messages: QueryJobType,
    outfile: Path,
    model_params: Optional[dict]=None
) -> openai.openai_object.OpenAIObject:
    """Function for getting a chat completion.
    
    Appends output to `outfile`.
    """
    if model_params is None:
        model_params = _get_model_params()
    LOGGER.info('Begin query')
    response = openai.ChatCompletion.create(messages=messages, **model_params)
    response_message = response.choices[0].message.to_dict()
    with open(outfile, mode='a') as f:
        f.write(json.dumps({'params': params, 'query': messages, 'response': response}) + '\n')
    return response

def query_all_chat_completions(
    messages: QueryJobListType,
    outfile: Path,
    model_params: Optional[dict]=None
) -> list[openai.openai_object.OpenAIObject]:
    """Run create_chat_completion many times to generate all chat completions.
    """
    if model_params is None:
        model_params = _get_model_params()
    return [create_chat_completion(msg['params'], msg['prompt'], outfile) for msg in messages]

def flatten_dict(source: dict) -> dict:
    """Take a dictionary which may have hierarchical elements (i.e. a dict in dict) and flatten it by one level.
    """
    out_dict = {}
    for key, item in source.items():
        if isinstance(item, dict):
            for subkey in item:
                out_dict[subkey] = item[subkey]
        else:
            out_dict[key] = item
    return out_dict

PromptListType = list[dict[str, dict|str]]
def get_incomplete_prompts(previous_data: pd.DataFrame, prompts: PromptListType, n_per_query: int
) -> PromptListType:
    """Return a subset of `prompts` that are not found in `previous_data`, so they may 
    be used to continue a cached query.
    """
    output = []
    
    # Construct a set of series corresponding to the prompts, with choice_index added
    prompts_df = pd.DataFrame([p['params'] for p in prompts])
    col_order = list(prompts_df.columns)
    prompts_df['prompt_index'] = prompts_df.groupby(col_order).cumcount()
    
    # Extract a set of Series corresponding to all previous prompts
    previous_prompts_df = previous_data.copy()
    previous_prompts_df['prompt_index'] = previous_prompts_df.groupby(col_order + ['choice_index']).cumcount()
    col_order += ['prompt_index']
    # Drop the rows that vary only by choice_index
    previous_prompts_df = previous_prompts_df[col_order].drop_duplicates()
    
    # Identify the overlap
    df_merge = pd.merge(prompts_df, previous_prompts_df, on=col_order, how='left', 
        indicator='Exist')
    # Return only those prompts not in the previous query
    for prompt_i, prompt in enumerate(prompts):
        if df_merge['Exist'].iloc[prompt_i] != 'both':
            output.append(prompt)
    return output

# -----------------------------
# Main logic
# -----------------------------

def main(base_query: str=BASE_PROMPT, query_options: PromptOptionsType=PROMPT_OPTIONS, model_params: Optional[dict]=None,
    query_repeats: int=1, previous_query_file: Optional[Path]=None
) -> tuple[Path, Path, list[openai.openai_object.OpenAIObject]]:
    """Run OpenAI queries using the `base_query` string and `query_options` demographic substitutions
    and save responses to a timestamped promptfile and output file.
    
    If `previous_query_file` is specified, it will be loaded and checked to see if
    it contains all expected completions; if so, it will be passed directly as `outfile`.
    If not, it will be copied and appended to until all queries are complete.
    
    Returns
    -------
    outfile: Path
        json file path (or zipped json) containing query outputs
    promptfile: Path
        json file path containing query prompts
    responses: list[openai.openai_object.OpenAIObject]
        OpenAI responses correspondong to the outfile
    """
    promptfile = Path(f'PROMPTS/gpt_prompts_{START_TIME}.jsonl')
    promptfile.parent.mkdir(exist_ok=True, parents=True)
    outfile = OUTPUT_DIR / f'gpt_api_output_{START_TIME}.jsonl'
    outfile.parent.mkdir(exist_ok=True, parents=True)
    
    openai.api_key = get_secret()
    
    # NOTE - we use flatten_dict here so that the user may input a dictionary as query_options, i.e. a set of parameter
    # values that go together and don't get expanded by the product function, which then get flattened before query time.
    flat_prompts = map(flatten_dict, product_dict(query_options))
    prompts = get_prompt_dicts(base_query, flat_prompts)
    if query_repeats > 1:
        prompts *= query_repeats
    with open(promptfile, 'w') as f:
        json.dump(prompts, f)
    
    if previous_query_file is not None:
        LOGGER.info(f'Loading previous data file {previous_query_file=}')
        previous_data = load_gpt_data(previous_query_file)
        remaining_prompts = get_incomplete_prompts(previous_data, prompts, model_params['n'])
        if len(remaining_prompts) == 0:
            LOGGER.info('Previous query determined to be complete; no further querying will be done')
            return previous_query_file, promptfile, None
        else:
            LOGGER.info('Previous query determined to be incomplete; querying will continue where it left off')
            # Duplicate the old file so we can append to a fresh copy
            shutil.copyfile(previous_query_file, outfile)
            # Subset the prompts to be queried to the ones not already done
            prompts = remaining_prompts
    
    responses = query_all_chat_completions(prompts, outfile, model_params)
    LOGGER.info('Query completed successfully')
    return outfile, promptfile, responses

if __name__ == '__main__':
    main()
