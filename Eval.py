import os
import time
from typing import Dict
import torch
import numpy as np
from ts_generation import ts_generation
from Data_Patch import Data
import matplotlib.pyplot as plt
from CL_Model import SpectralGroupAttention
from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm import tqdm
from Utils import checkFile, standard, file_must_exist, time_func
from Scheduler import GradualWarmupScheduler
import torch.nn.functional as F
import scipy.io as sio
from sklearn import metrics
import random

from CL_Train import (seed_torch, paintTrend, normalize, 
                      cosin_similarity, transpose, info_nce_loss)
import pickle
import argparse
import pprint

modelConfig = {
        "state": "eval",  # train or select_best, eval
        "epoch": 200,
        "band": 189,
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
        "dataset": "Sandiego",
        "data_path": "datasets/Sandiego.mat",
        "target_path": None,
        "ts_type": 7
}


@file_must_exist
def load_ts_data(filename):
    if filename.endswith(".mat"):
        mat = sio.loadmat(filename)
    elif filename.endswith(".pkl"):
        with open(filename, 'rb') as f:
            mat = pickle.load(f)
    else:
        raise ValueError(f"The file extension of {filename} is not supported.")
    
    if type(mat) is dict and 'data' in mat:
        data = mat['data']
    else:
        raise ValueError(f"Contents of file {filename} must be a dictionary with a 'data' key.")

    target_spectrum = np.expand_dims(standard(data), axis=-1)
    return target_spectrum


@time_func
def eval(modelConfig: Dict):
    start = time.perf_counter()
    seed_torch(modelConfig['seed'])
    device = torch.device(modelConfig["device"])
    path = modelConfig["save_dir"] + '/' + modelConfig['dataset'] + '/'
    with torch.no_grad():
        # Load image data
        mat = sio.loadmat(modelConfig["data_path"])
        data = mat['data']
        map = mat['map']
        data = standard(data)
        # Load target_spectrum if target_path is specified, generate if not
        if modelConfig["target_path"] is not None:
            target_spectrum = load_ts_data(modelConfig["target_path"])
        else:
            target_spectrum = ts_generation(data, map, modelConfig["ts_type"])
        h, w, c = data.shape
        numpixel = h * w
        data_matrix = np.reshape(data, [-1, c], order='F')
        model = SpectralGroupAttention(band=modelConfig['band'], group_length=modelConfig['m'],
                                       channel_dim=modelConfig['channel'], state_size=modelConfig['state_size'],
                                       device=device, layer=modelConfig['layer'])
        model = model.to(device)
        ckpt = torch.load(os.path.join(
            path, modelConfig["test_load_weight"]), map_location=device)
        model.load_state_dict(ckpt)
        print("model load weight done.")
        model.eval()

        batch_size = modelConfig['batch_size']
        detection_map = np.zeros([numpixel])
        target_prior = torch.from_numpy(target_spectrum.T)
        target_prior = target_prior.to(device)
        target_prior = torch.unsqueeze(target_prior, dim=1)
        target_features = model(target_prior)
        target_features = target_features.cpu().detach().numpy()

        for i in range(0, numpixel - batch_size, batch_size):
            pixels = data_matrix[i:i + batch_size]
            pixels = torch.from_numpy(pixels)
            pixels = pixels.to(device)
            pixels = torch.unsqueeze(pixels, dim=1)
            features = model(pixels)
            features = features.cpu().detach().numpy()
            detection_map[i:i + batch_size] = cosin_similarity(features, target_features)

        left_num = numpixel % batch_size
        if left_num != 0:
            pixels = data_matrix[-left_num:]
            pixels = torch.from_numpy(pixels)
            pixels = pixels.to(device)
            pixels = torch.unsqueeze(pixels, dim=1)
            features = model(pixels)
            features = features.cpu().detach().numpy()
            detection_map[-left_num:] = cosin_similarity(features, target_features)

        detection_map = np.exp(-1 * (detection_map - 1) ** 2 / modelConfig['delta'])
        detection_map = np.reshape(detection_map, [h, w], order='F')
        detection_map = standard(detection_map)
        detection_map = np.clip(detection_map, 0, 1)
        end = time.perf_counter()
        print('excuting time is %s' % (end - start))
        # save_path = '/home/sdb/experiments/20240504/%s/HTD-Mamba.mat' % modelConfig['dataset']
        # sio.savemat(save_path, {'map': detection_map})
        y_l = np.reshape(map, [-1, 1], order='F')
        y_p = np.reshape(detection_map, [-1, 1], order='F')

        ## calculate the AUC value
        fpr, tpr, threshold = metrics.roc_curve(y_l, y_p, drop_intermediate=False)
        fpr = fpr[1:]
        tpr = tpr[1:]
        threshold = threshold[1:]
        auc1 = round(metrics.auc(fpr, tpr), modelConfig['epision'])
        auc2 = round(metrics.auc(threshold, fpr), modelConfig['epision'])
        auc3 = round(metrics.auc(threshold, tpr), modelConfig['epision'])
        auc4 = round(auc1 + auc3 - auc2, modelConfig['epision'])
        auc5 = round(auc3 / auc2, modelConfig['epision'])
        print('{:.{precision}f}'.format(auc1, precision=modelConfig['epision']))
        print('{:.{precision}f}'.format(auc2, precision=modelConfig['epision']))
        print('{:.{precision}f}'.format(auc3, precision=modelConfig['epision']))
        print('{:.{precision}f}'.format(auc4, precision=modelConfig['epision']))
        print('{:.{precision}f}'.format(auc5, precision=modelConfig['epision']))

        plt.imshow(detection_map)
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="Eval.py", 
                            description="Evaluation script for HTD-Mamba.")
    for key in modelConfig.keys():
        parser.add_argument('--' + key, type=type(modelConfig[key]), required=False,
                            default=modelConfig[key], help=f"(default: {modelConfig[key]})")
    parser.add_argument('--dry_run', '-n', action='store_true', 
                        help="Only print the modelConfig dictionary without running the evaluation")

    args = parser.parse_args()
    modelConfig.update(vars(args))

    if args.dry_run:
        pprint.pp(modelConfig)
    else:
        eval(modelConfig)