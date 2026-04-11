# Probabilistically-Sound Beam Search with Masked Language Models

This repo contains the code necessary to run experiments using the HCB beam search methodology
desribed in our recent preprint, [Probabilistically-Sound Beam Search with Masked Language Models](https://arxiv.org/abs/2402.15020)

## Setup
To install all dependencies in a conda environment, one can first clone this repo, then run:

```
conda env create -f conda_env.yml -n hcb_infilling_env
```

This conda environment will contain all necessary dependencies (and probably also a few extra packages
as well).

## Experiments
To run experiments using our HCB beam search methods, please refer to `scripts/example_experiment.py` for
an example of how to compare HCB beam search to standard beam search on the Brown corpus. Please feel
free to inspect the differences between the methods by beginning at `hcb_infilling/decode.py`.

For questions, don't hesitate to make an issue here. Happy infilling!
