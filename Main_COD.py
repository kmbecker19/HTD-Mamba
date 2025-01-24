'''
Main_COD.py

A script for easily editing the training and evaluation parameters for
HTD-Mamba.
'''
import time
import argparse

from CL_Train import train, eval, select_best
import os
import pprint
from typing import Union
from Utils import time_func

os.environ['CUDA_LAUNCH_BLOCKING'] = '0'
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

modelConfig = {
        "state": "select_best",  # train or select_best, eval
        "epoch": 200,
        "band": 201,
        "batch_size": 80,
        "seed": 1,
        "channel": 16,
        "lr": 1e-4,
        "multiplier": 2.,
        "epision": 10,
        "grad_clip": 1.,
        "device": "cuda:0",  ### MAKE SURE YOU HAVE A GPU !!!
        "training_load_weight": None,
        "save_dir": "./models/",
        "test_load_weight": "ckpt_180_.pt",
        "patch_size": 11,
        "m": 30,
        "state_size":16,
        "layer": 1,
        "delta": 0.1,
        "dataset": "Subj_015",
        "path": "datasets/subj_015.mat"
}


@time_func
def main(model_config: dict):
    if model_config["state"] == "train":
        train(model_config)
    elif model_config["state"] == "select_best":
        select_best(model_config)
    elif model_config["state"] == "eval":
        eval(model_config)
    else:
        raise ValueError("Invalid state. Choose `train`, `select_best`, or `eval`.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="A script for easily editing the training and evaluation parameters for HTD-Mamba."
    )

    # Add command line arguments for each parameter in modelConfig
    for key in modelConfig.keys():
        parser.add_argument(f'--{key}', type=type(modelConfig[key]), 
                            default=modelConfig[key], 
                            help=f"Parameter <{key.upper()}> for model. Default: {modelConfig[key]}")

    args = parser.parse_args()
    modelConfig.update(vars(args))

    main(modelConfig)