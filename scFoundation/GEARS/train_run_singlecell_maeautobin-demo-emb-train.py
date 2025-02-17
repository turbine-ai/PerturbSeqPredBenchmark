import argparse
from dataclasses import dataclass
from datetime import datetime
import os

import os
import time
import argparse
import pandas as pd
import scanpy as sc
from os.path import join as pjoin

from gears import PertData, GEARS

def create_result_dir(args):
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    result_dir = os.path.join(
        args.workdir, "results", args.data_name, str(args.train_gene_set_size),
        f"baseline_{args.split}_seed_{args.seed}_hidden_{args.hidden_size}_epochs_{args.epochs}_batch_{args.batch_size}_accmu_{args.accumulation_steps}_mode_{args.mode}_highres_{args.highres}_lr_{args.lr}",
        timestamp
    )
    os.makedirs(result_dir, exist_ok=True)
    return result_dir


def main(args):
    # args = parser.parse_args()

    # get data
    pert_data = PertData(args.data_dir)
    # load dataset in paper: norman, adamson, dixit.
    # try:
    #     if args.data_name in ['norman', 'adamson', 'dixit']:
    #         pert_data.load(data_name = args.data_name)
    #     else:
    #         print('load data')
    #         pert_data.load(data_path = pjoin(args.data_dir, args.data_name))
    # except:
    adata = sc.read_h5ad(pjoin(args.data_dir, args.data_name+'.h5ad'))
    adata.uns['log1p'] = {}
    adata.uns['log1p']['base'] = None
    pert_data.new_data_process(dataset_name=args.data_name, adata=adata)
        
    print(adata)
    # specify data split
    pert_data.prepare_split(split = args.split, seed = args.seed, train_gene_set_size=args.train_gene_set_size)
    # get dataloader with batch size
    pert_data.get_dataloader(batch_size = args.batch_size, test_batch_size = args.test_batch_size)

    # set up and train a model
    gears_model = GEARS(pert_data, device = args.device)
    gears_model.model_initialize(hidden_size = args.hidden_size, 
                                 model_type = args.model_type,
                                 bin_set=args.bin_set,
                                 load_path=args.singlecell_model_path,
                                 finetune_method=args.finetune_method,
                                 accumulation_steps=args.accumulation_steps,
                                 mode=args.mode,
                                 highres=args.highres)
    gears_model.train(epochs = args.epochs, result_dir=args.result_dir,lr=args.lr)

    # save model
    gears_model.save_model(args.result_dir)

    # save params
    param_pd = pd.DataFrame(vars(args), index=['params']).T
    param_pd.to_csv(f'{args.result_dir}/params.csv')

@dataclass
class RunArgs:
    device: str = "cuda"
    data_dir: str = "./data/"
    data_name: str = "demo"
    split: str = "simulation"
    seed: int = 1
    epochs: int = 1
    batch_size: int = 2
    accumulation_steps: int = 1
    test_batch_size: int = 2
    hidden_size: int = 512
    train_gene_set_size: float = 0.75
    mode: str = "v1"
    highres: int = 0
    lr: float = 0.0002
    model_type: str | None = "maeautobin"
    bin_set: str | None = "autobin_resolution_append"
    finetune_method: str | None = "frozen"
    singlecell_model_path: str | None = "../model/models/models.ckpt"
    workdir: str = "./"
    result_dir: str | None = None


if __name__ == "__main__":
    args = RunArgs()

    args.result_dir = create_result_dir(args)
    main(args)
    
