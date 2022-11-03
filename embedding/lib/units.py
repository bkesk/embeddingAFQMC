'''
Helper Physical constants / converters.

Most internal quantities are expressed in atomic units
'''

from collections.abc import Iterable

ANGSTROM = 1.889725989 # Bohr

def Ang2Bohr(pos: Iterable) -> tuple:
    '''
    Returns the position given in 'pos'(specified in Angstrom) in units Bohr
    '''
    return tuple( p*ANGSTROM for p in pos )


def Bohr2Ang(pos: Iterable) -> tuple:
    '''
    Returns the position given in 'pos'(specified in Angstrom) in units Bohr
    '''
    return tuple( p/ANGSTROM for p in pos )
