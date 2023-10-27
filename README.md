# Code for "Demonstrations of the Potential of AI-based Political Issue Polling"

This repository contains some code for testing the hypothesis that large language models can be used to generate distributionally-realistic responses to questions about public policy.

This code underlies the results presented in the paper Sanders, N. E., Ulinich, A., & Schneier, B. (2023). [Demonstrations of the Potential of AI-based Political Issue Polling. Harvard Data Science Review, 5(4).](https://hdsr.mitpress.mit.edu/pub/dm2hrtx0/release/1?readingCollection=04d84e0d) ([DOI 10.1162/99608f92.1d3cf75d](https://doi.org/10.1162/99608f92.1d3cf75d) and on arXiv at [arxiv:2307.04781](https://arxiv.org/abs/2307.04781)).

## Usage

Run the automatic AI polling script as follows,

```
python query_and_analyze_ces.py -c 'configs/[my_config].json'
```

To run this, you will need...

* The `conda` environment specified by the `yml` file detailed below.
* A config file, whose schema is specified in `query_and_analyze_ces.py`. Some example configs are provided in the `configs` directory. The most recent of the configs was the one used to produce the version of the analysis that appears in the published work.
* The [CES data](https://doi.org/10.7910/DVN/PR4L8P) downloaded in Stata format; see `parse_ces.py` for details. 
* An OpenAI API secret, expected to be stored in the file `SECRET_openai.txt` or set as the environment variable `OPENAI_API_KEY`.

## Contents

The only user-facing script that need be run is,

* `query_and_analyze_ces.py`: This is the primary entry point for the polling application. It leverages the functionality of all the scripts below.

The following scripts are imported by `query_and_analyze_ces.py` and, while they may have standalone functionality, generally need not be run directly.

* `openai_api_query.py`: This script handles constructing GPT prompts and querying the OpenAI API for completions.
* `parse_ces.py`: This script parses data downloaded from the CES survey for use in generating queries and plotting comaprisons.
* `analysis.py`: This script generates plots and analysis from the output of `openai_api_query.py`.
* `conda_openai_api.yml`: The `conda` environment specification for using the OpenAI API. This env file pins all versions and is fully tested in a Linux environment. Cross-platform users may have more luck using the `conda_openai_api_cross_platform.yml` file, which only pins a minal set of minor versions.
* `utils.py`: Reusable utilities shared by various of the above scripts.

The `configs` issue contains configurations used to execute each of the two versions of the analysis published in the arXiv preprint, and `QUERY_OUTPUTS` contains the cached GPT-3.5 responses associated with each config. The `configs` directly also contains a test configuration suitable for small scale testing.
