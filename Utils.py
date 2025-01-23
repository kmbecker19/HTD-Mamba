import numpy as np
import os
from functools import wraps
import time
import spectral as sp
from pathlib import Path

def time_func(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        print(f'{func.__name__} executed in {end_time - start_time} seconds')
        return result
    return wrapper

def file_must_exist(func):
    @wraps(func)
    def wrapper(filename, *args, **kwargs):
        if not os.path.exists(filename):
            raise FileNotFoundError(f'The file {filename} does not exist.')
        return func(filename, *args, **kwargs)
    return wrapper
    

def load_from_hsi(func):
    '''
    Helper function to automatically load an HSI from a .hdr file with spectralpython.
    '''
    @wraps(func)
    def wrapper(directory, *args, **kwargs):
        if not os.path.isdir(directory):
            raise NotADirectoryError(f'The directory {directory} does not exist.')
        filename = Path(directory).name
        filename_ref = f"REFLECTANCE_{filename}"
        path_reflectance = os.path.join(directory, 'capture', f'{filename_ref}.hdr')
        img_reflectance = sp.envi.open(path_reflectance, Path(path_reflectance).with_suffix('.dat'))
        hsi = img_reflectance.load()
        hsi = np.rot90(hsi, k=-1, axes=(0, 1))
        return func(hsi, *args, **kwargs)
    return wrapper


def standard(X):
    '''
    Standardization
    universal
    :param X:
    :return:
    '''
    min_x = np.min(X)
    max_x = np.max(X)
    if min_x == max_x:
        return np.zeros_like(X)
    return np.float32((X - min_x) / (max_x - min_x))

def checkFile(path):
    '''
    if filepath not exist make it
    :param path:
    :return:
    '''
    if not os.path.exists(path):
        os.makedirs(path)