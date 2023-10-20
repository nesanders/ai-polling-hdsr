"""Shared utilities for AI polling
"""

from datetime import datetime
import logging
import numpy as np
from pathlib import Path
from scipy.stats import bootstrap
import subprocess

DATE_FORMAT: str = '%Y_%m_%d_%H-%M-%S'
START_TIME = datetime.now().strftime(DATE_FORMAT) 
QUERY_OUTPUT_DIR = Path('QUERY_OUTPUTS/')
PromptOptionsType = dict[str, list[str | int | float | dict[str, str]]]
PROMPT_OPTIONS: PromptOptionsType = {
    'issue': [
        #'cancelling student debt', 
        #'eliminating access to abortion pills', 
        #'transferring funds from police budgets to social services',
        #'increasing the corporate tax rate'
        #'informing the public when sewage is spilled into rivers',
        #'increasing subsidies for veterans\' healthcare',
        #'increasing standards for public school educational achievement',
        #'giving aid to Afghan translators who worked with American troops'
        'eliminating all funding for police agencies',
        'ending public school education after the 8th grade',
        'removing protections against animal abuse',
        'eliminating the US Social Security program',
    ],
    'age': [18, 80],
    'gender': ['man', 'woman'],
    'state': ['Alabama', 'Massachusetts'],
    'ideology': ['liberal', 'liberal leaning', 'conservative leaning', 'conservative']
}


def get_git_hash() -> str:
    return subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()

def setup_logging(logfile_prefix: str | Path, level=logging.INFO) -> logging.Logger:
    """ Setup logging at some prefix `logfile_prefix` with a timestamp and log extension 
    suffixed to it.
    """
    logger = logging.getLogger(logfile_prefix)
    logger.setLevel(level)
    
    logfile = Path(logfile_prefix) / f'{START_TIME}.log'
    logfile.parent.mkdir(exist_ok=True, parents=True)
    file_handler = logging.FileHandler(logfile)
    logger.addHandler(file_handler)
    
    logger.info(f'Script started at {START_TIME}')    
    logger.info(f'Current git hash: {get_git_hash()}')
    return logger

def list_is_unique(source: list) -> bool:
    """Return True if all elements in an input are unique.
    """
    seen = set()
    return not any(itm in seen or seen.add(itm) for itm in source)

def invert_dict(source: dict) -> dict:
    """Invert a key->value dictionary to be a value->key dict.
    """
    assert list_is_unique(list(source.values())), 'Cannot invert a dict with non-unique values'
    return {v: k for k, v in source.items()}

def bootstrap_se(data: np.ndarray, **kwargs) -> dict[str, float]:
    """Return a dict with the bootstrap mean and standard error of a data distribution.
    """
    return bootstrap((data, ), statistic=np.nanmean, **kwargs).standard_error
