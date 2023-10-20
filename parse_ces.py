"""Code to parse data from the Cooperative Election Study Common Content, 2022 data release.

# Usage

If this script is run as `__main__`, it will export a csv of CES data with a few default demographic
and policy issue columns as generated by the `get_ces_data` function.

# Data

The data itself can be downloaded from the following URL and is saved in 'ces_data'

https://doi.org/10.7910/DVN/PR4L8P

Note: we use the stata format (.dta) so that the codebook is integrated directly.
"""

from pathlib import Path
import numpy as np
import pandas as pd

from utils import invert_dict, setup_logging

CES_DATA_DIR = Path(__file__).parent / 'ces_data'
DATA_FILE = CES_DATA_DIR / 'CES22_Common.dta.gz'

DATA_YEAR = 2022

LOGGER = setup_logging('LOGS/parse_ces')

# Demographic fields to use
CES_DEMOS: dict[str, str] = {
    'age': 'birthyr',
    'gender': 'gender4',
    'state': 'inputstate',
    'ideology': 'ideo5',
    'income': 'faminc_new',
    'parent': 'child18',
    'white': 'multrace_1'
}

AGE_BINS = [16, 30, 45, 60, 120]

INCOME_BINS = {
    '<$40k': ['Less than $10,000', '$10,000 - $19,999', '$20,000 - $29,999', '$30,000 - $39,999'],
    '$40-80k': ['$40,000 - $49,999', '$50,000 - $59,999', '$60,000 - $69,999', '$70,000 - $79,999'], 
    '$80-250k': ['$80,000 - $99,999', '$100,000 - $119,999', '$120,000 - $149,999', '$150,000 - $199,999', '$200,000 - $249,999'], 
    '>$250k': ['$250,000 - $349,999', '$350,000 - $499,999', '$500,000 or more'], 
    np.nan: ['Prefer not to say']
}

# Which policy questions cols to subset the DataFrame on
RetainPoliciesType = dict[str, dict[str, str]]
RETAIN_POLICIES: RetainPoliciesType = {
    'medicare_drug_prices': {
        'code': 'CC22_327b',
        'prompt': 'Allow the government to negotiate with drug companies to get '
                  'a lower price on prescription drugs that would apply to both Medicare and '
                  'private insurance. Maximum negotiated price could not exceed 120% of the average '
                  'prices in 6 other countries.'
    },
    'gun_background_checks': {
        'code': 'CC22_330e',
        'prompt': 'Improve background checks to give authorities time to check the '
            'juvenile and mental health records of any prospective gun buyer '
            'under the age of 21'
    },
    'increase_fuel_production': {
        'code': 'CC22_333e',
        'prompt': 'Increase fossil fuel production in the U.S. and boost exports of U.S. '
            'liquefied natural gas'
    },
}



# Placeholder global
CES_DATA = None

def get_ces_data(
    demos: list[str]=list(CES_DEMOS.keys()),
    policies: RetainPoliciesType=RETAIN_POLICIES
) -> pd.DataFrame:
    """Return a DataFrame of CES data for a given set of `demos` and `policies` 
    policy questions.
    """
    if CES_DATA is None:
        LOGGER.info(f'Loading CES data from {DATA_FILE}')
        df = pd.read_stata(DATA_FILE)
        _modify_demos(df)
        return _format_data(df, demos=demos, policies=policies)
    else:
        return CES_DATA

def _remove_not_sure(data: pd.Series):
    """Remove `'Not Sure'` category from a categorical series.
    """
    if data.dtype == 'category' and 'Not sure' in data.cat.categories:
        return data.cat.remove_categories(['Not sure'])
    return data

def _modify_demos(df: pd.DataFrame):
    """Do some reformatting of the CES demos; modifies `df` in place.
    
    Make the following changes to each demo:
    
    * Age: Convert from birth year to categorical binned ages.
    * Gender: Drop non-binary categories
    * Ideology: Remove 'not sure' category
    * Income: Map to a smaller set of categories
    * White: convert to 'white' or 'not white'
    
    Also rename the columns
    """
    LOGGER.info(f'Modifying demographic column formats')
    df.rename(columns=invert_dict(CES_DEMOS), inplace=True)
    # NOTE: pd.cut will return a categorical with an IntervalIndex, which seaborn does not understand.
    # Below, we cast the categorical vector to string labels.
    df['age'] = pd.cut(DATA_YEAR - df['age'], AGE_BINS).astype('str').astype('category')
    df['gender'] = df['gender'].cat.remove_categories(['Other', 'Non-binary'])
    df['ideology'] = _remove_not_sure(df['ideology'])
    income_bin_invert = {val:bin for bin, vals in INCOME_BINS.items() for val in vals}
    df['income'] = df['income'].map(income_bin_invert)
    df['white'] = df['white'].map({'not selected': 'non-white', 'selected': 'white'})

def _format_data(
    df: pd.DataFrame, 
    demos: list[str], 
    policies: RetainPoliciesType
) -> pd.DataFrame:
    """Return only a subset of a DataFrame corresponding to the desired demographic fields (`demos) and policy
    questions (`policies`). The policy columns will be named according to the input `policies` dict.
    
    Rows with any null values in the demos will be dropped. In practice, ~7% of ideology and ~8% of income
    are null and get dropped.
    """
    LOGGER.info(f'Formatting data for output')
    policy_col_map = {pol_dict['code']: pol_name for pol_name, pol_dict in policies.items()}
    df_copy = df.copy()
    df_copy.rename(columns=policy_col_map, inplace=True)
    policy_cols = list(policy_col_map.values())
    for issue in policy_cols:
        df_copy[issue] = _remove_not_sure(df_copy[issue])
    df_copy.dropna(subset=demos, how='any', inplace=True)
    assert len(df_copy) > 0, '_format_data failed; all rows dropped'
    return df_copy[demos + list(policy_cols)]

if __name__ == '__main__':
    df = get_ces_data()
    outfile = CES_DATA_DIR / 'parsed_ces_data.csv.gz'
    LOGGER.info(f'Writing output ti {outfile}')
    df.to_csv(outfile)
