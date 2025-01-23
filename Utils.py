import numpy as np
import os
from functools import wraps
import time

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