'''
Store_Matdata.py

A script for generating MATLAB-5 style .mat files from .hdr HSI files and JSON polgyon segmentation maps.
'''
import os
from pathlib import Path
from Utils import time_func, file_must_exist, load_from_hsi
import scipy.io as sio
import numpy as np
import argparse
import json
import pickle
from PIL import Image, ImageDraw

IMG_SIZE = (320, 247)


@load_from_hsi
def store_image(image, bands=200, mat_dict=None) -> dict:
    '''
    Take an image from a .hdr file and store it in a dictionary following .mat
    file format.
    '''
    if mat_dict is not None:
        result = mat_dict
    else:
        result = {
            'map': None,
            'data': None
        }
    _, _, n_bands = image.shape
    b_start = (n_bands // 2) - (bands // 2)
    b_end = (n_bands // 2) + (bands // 2)
    result['data'] = image[:, :, b_start:b_end]
    return result


def store_map_from_json(map_path, target='PVC', mat_dict=None) -> dict:
    '''
    Load a segmentation from a JSON file and store it in a dictionary following .mat
    file format.
    '''
    if mat_dict is not None:
        result = mat_dict
    else:
        result = {
            'map': None,
            'data': None
        }
    with open(map_path, 'r') as f:
        map_data = json.load(f)

    # Create binary segmentation map from the polygon points given in the JSON file
    # TODO: Potentially edit this code to work for multiple target instances
    for object in map_data['objects']:
        if object['category'] == target:
            polygon = [(round(x), round(y)) for [x, y] in object['segmentation']]
            img = Image.new('L', IMG_SIZE, 0)
            ImageDraw.Draw(img).polygon(polygon, outline=1, fill=1)
            result['map'] = np.array(img, dtype='uint8')
            break
    return result
            
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='A program for generating MATLAB-5 style .mat files from .hdr HSI files and JSON polgyon segmentation maps.'
    )

    # Define the command line arguments
    arguments = {
        'map_file': 'map.json',
        'output_file': 'output.mat',
        'bands': 200,
    }
    parser.add_argument('data_dir', type=str, help='The path to the HSI data to store as a MATLAB file.')
    for key in arguments.keys():
        parser.add_argument(f'-{key[0]}', f'--{key}',
                            type=type(arguments[key]), 
                            default=arguments[key],
                            metavar='',
                            help=f'Set {key} to value. (default: {arguments[key]})')

    parser.add_argument('-t', '--target', type=str, choices=['PVC', 'Metal', 'Vest'], metavar='', default='PVC',
                        help='Set the target category for segmentation. (choices: PVC, Metal, Vest)')
    
    args = parser.parse_args()
    
    # Load the HSI data
    mat_data = store_image(args.data_dir, args.bands)

    # Load the map data
    mat_data = store_map_from_json(args.map_file, args.target, mat_dict=mat_data)

    # Save the results to a .mat file
    print(f"Saving data to file '{args.output_file}'...")
    sio.savemat(args.output_file, mat_data, do_compression=True)
    print(f"Data saved to '{args.output_file}'")