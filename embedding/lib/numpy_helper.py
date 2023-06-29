"""
Numpy helper functions

author: Kyle Eskridge
date: 6/28/2023
GitHub: bkesk
"""
import os
import json

import numpy as np

LIB_PATH='/mnt/home/beskridge/software/embeddingAFQMC/embedding/lib/einsum_paths'

def save_einsum_path(path,fname="path.json"):
    '''
    Save einsum path to 'fname' within the embeddingAFQMC
    library. This will overwrite existing an existing path
    saved under the same name!
    '''
    with open(os.path.join(LIB_PATH,fname),'w') as f:
        f.write(json.dumps(path))


def load_einsum_path(fname="path.json"):
    '''
    Attempt to load a known optimized tensor contraction path
    from the library. If the path does not exist in the library,
    None is returned
    '''
    try:
        with open(os.path.join(LIB_PATH,fname),'r') as f:
            path = json.loads(f.read())
            return path
    except FileNotFoundError:
        return None

def einsum_optimized(contraction_str,*arg,fname='default.json'):
    """
    Wrapper for numpy.einsum that uses an optimized contraction path
    if possible.

    arg : is used to capture the tensors to be contracted
    kwargs are forwarde

    This function will attempt the following (in order) until one is successful:
    1. load an optimized path from embeddingAFQMC's internal library
    2. recompute the optimized path - save to internal library

    """
    print('Attempting to load pre-optimized path', flush=True)
    path = load_einsum_path(fname)

    if path is None:
        print('Could not find preoptimized path - optimizing einsum path', flush=True)
        path,path_str = np.einsum_path(contraction_str,*arg,optimize='greedy')
        print("     optimal path: ", path)
        print(path_str)

        print('Saving optimized path for future use as ', fname, flush=True)
        save_einsum_path(path=path,fname=fname)

    print('contracting tensor along optimal path', flush=True)
    return np.einsum(contraction_str,*arg,optimize=path)
