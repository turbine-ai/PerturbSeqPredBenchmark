# Benchmarking foundational cell models for post-perturbation RNAseq prediction

This is the official Github repository of the [Paper](link). This is a fork of the [scGPT](https://www.nature.com/articles/s41592-024-02201-0)  [repository](https://github.com/bowang-lab/scGPT).

## Repository content

- notebooks/
    - _bulk_models.ipynb_: train RF and EN, compares performance to scGPT and "Train Mean"
    - _data_analysis.ipynb_: runs data analysis and generates Figure 2
    - _scgpt_mean.ipynb_: runs the mean model and compares it with scGPT
    - _Tutorial_PerturbationAdamson.ipynb_: trains  scGPT on the Adamson et al. dataset
    - _Tutorial_PerturbationNorman.ipynb_: trains scGPT on the Norman et al. dataset
    - _Tutorial_PerturbationReplogle.ipynb_: trains scGPT on the Replogle et al. dataset

## Reproducibility

To reproduce the results of the paper, please follow the following steps: 
1. Create the conda environment using `make setup`
1. Select the _scgpt_yml_ conda environment as the Python kernel for the notebooks
1. Run the selected notebooks
