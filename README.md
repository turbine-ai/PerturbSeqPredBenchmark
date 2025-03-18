# Benchmarking a foundational cell model for post-perturbation RNAseq prediction

This is the official Github repository of the paper [Benchmarking a foundational cell model for post-perturbation RNAseq prediction](https://www.biorxiv.org/content/10.1101/2024.09.30.615843v1). This is a fork of the [scGPT](https://www.nature.com/articles/s41592-024-02201-0)  [repository](https://github.com/bowang-lab/scGPT).

## Repository content

- notebooks/
    - _bulk_models.ipynb_: trains RF, Elastic Net, KNN Regressor and Train Mean with GO, scGPT, scFoundation and scElmo features (Figure 1 B - E, 2 D)
    - _data_analysis.ipynb_: runs data analysis and generates Figure 2 A - C
    - _Tutorial_PerturbationAdamson.ipynb_: trains  scGPT on the Adamson et al. dataset
    - _Tutorial_PerturbationNorman.ipynb_: trains scGPT on the Norman et al. dataset
    - _Tutorial_PerturbationReplogle.ipynb_: trains scGPT on the Replogle et al. (K562) dataset
    - _Tutorial_PerturbationReplogleRPE1.ipynb_: trains scGPT on the Replogle et al. (RPE1) dataset
    - _embedding_eval.ipynb_: embedding analysis

- scFoundation training entry points at _scFoundation/GEARS_:
    - _train_adamson.py_: trains on the Adamson et al. dataset
    - _train_norman.py_: trains on the Norman et al. dataset
    - _train_replogle_rp1.py_: trains on the Replogle et al. (K562) dataset
    - _train_replogle.py_: trains on the Replogle et al. (RPE1) dataset

## Reproducibility

To reproduce the results of the paper, please follow the following steps: 
1. Run `git lfs pull` to download the required data from Git Large File System. If _lfs_ is not installed, pleaser refer to this [guide](https://docs.github.com/en/repositories/working-with-files/managing-large-files/installing-git-large-file-storage)
1. Run `make setup` to create the conda environment, install the ipython kernel and unzip the replogle dataset
1. Run scGPT trainings
    1. Select the _scgpt_yml_ conda environment as the Python kernel for the notebooks
    1. Run the _Tutorial_ notebooks to get the results of scGPT
1. Run scFoundation trainings
    1. Create the conda environment for scFoundation by running `conda create env -f scFoundation/conda.yaml`
    1. Run `conda activate scfoundation`
    1. Start the trainings via the entry points

1. Run _data_analysis.ipynb_
1. Run _bulk_models.ipynb_
