# Benchmarking a foundational cell model for post-perturbation RNAseq prediction

This is the official Github repository of the [Benchmarking a foundational cell model for post-perturbation RNAseq prediction](link). This is a fork of the [scGPT](https://www.nature.com/articles/s41592-024-02201-0)  [repository](https://github.com/bowang-lab/scGPT).

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
1. Run `git lfs pull` to download the required data from Git Large File System. If _lfs_ is not installed, pleaser refer to this [guide](https://docs.github.com/en/repositories/working-with-files/managing-large-files/installing-git-large-file-storage)
1. Run `make setup` to create the conda environment, install the ipython kernel and unzip the replogle dataset
1. Select the _scgpt_yml_ conda environment as the Python kernel for the notebooks
1. Run _data_analysis.ipynb_
1. Run the _Tutorial_ notebooks to get the results of scGPT
1. Run _scgpt_mean.ipynb_
1. Run _bulk_models.ipynb_
