import logging
import h5py as h5

from .infile_yaml import get_settings

def read_orbitals(orb_string, orb_name='/scf/mo_coeff'):
    '''
    read orbitals from an HDF5 file. Orbitals should be stored based
    on PySCF's conventions: '/scf/mo_coeff' by default.
    '''
    
    temp = orb_string.split(':')
    fname = temp[0]
    if len(temp) == 2:
        orb_name = temp[1]

    try:
        with h5.File(fname,'r') as f:
            orbitals = f[orb_name][...].copy()
        return orbitals
    except Exception as e:
        logging.error(f'Unable to read orbitals: {e}')
        return None


def orbitals_from_input(fname):
    '''
    read orbitals based on settings
    '''
    return read_orbitals(get_settings(fname, block_name='orbital_basis'))

