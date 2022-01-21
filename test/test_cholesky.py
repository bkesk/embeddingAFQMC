import numpy as np
import h5py as h5
import embedding.cholesky as ch

import pytest

# Q: do I need my own class, or won't a PySCF SCF object suffice?
class TwoBody:
    '''
    Generic Two-Body Tensor test class:

    Attributes:
      - data (numpy.array with dtype float or complex): array containing two-body tensor elements 
      - tol (float) : numerical tolerance for equivalence
    '''
    def __init__(self, data=None, tol=1.0E-8):
        self.data = data
        self.tol = tol

    def __eq__(self, other):
       assert isinstance(other, TwoBody)
       assert self.data.shape == other.data.shape
       return np.isclose(self.data, other.data)

@pytest.fixture
def model_system():
    from pyscf import gto, scf

    atoms = '''He  0.0  0.0  0.0'''

    mol = gto.M(atom=atoms,
                basis='ccpvdz',
                spin=0, 
                verbose=4,
                parse_arg=False) 

    mf = scf.ROHF(mol)
    E = mf.kernel()

    return mf

def test_cholesky(model_system):
    from embedding.cholesky.integrals.gto import GTOIntegralGenerator

    mf = model_system

    integrals = GTOIntegralGenerator(mf.mol)
    ncv, twoBody = ch.cholesky(integrals, tol=1.0E-6)

    # need the "correct" values of ncv and twoBody -> store them in an hdf5 file?
    assert ncv > 0

def test_cholesky_input():
    '''
    test that the cholesky function rejects 'None' integral generator
    '''
    with pytest.raises(TypeError):
        ch.cholesky()

def test_cholesky_input():
    '''
    test that the cholesky function rejects random matrix as integral generator
    '''
    mat = np.random.rand(4,4)
    with pytest.raises(TypeError):
        ch.cholesky(mat)

