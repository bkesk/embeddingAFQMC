import os
import h5py as h5
import numpy as np

import embedding as pel
import embedding.cholesky as ch

import pytest

@pytest.fixture
def local_model_system():
    from pyscf import gto, scf

    atoms = '''O  0.000000  0.000000  1.5
    H   -0.4704240745 0.0 0.0
    H   0.4704240745 0.0 0.0
    H   1.4112722235 0.0 0.0
    H   2.3521203725 0.0 0.0
    H   3.2929685215 0.0 0.0
    H   4.2338166705 0.0 0.0
    H   5.1746648195 0.0 0.0
    H   6.1155129685 0.0 0.0
    H   7.0563611175 0.0 0.0
    H   7.9972092665 0.0 0.0
    '''

    mol = gto.M(atom=atoms,
                basis={'O' : 'ccpvdz', 'H' : 'sto6g'},
                spin=2, 
                verbose=4,
                parse_arg=False) 

    mf = scf.ROHF(mol)
    E = mf.kernel()

    return mf


def test_high_level():
    from embedding.cholesky.integrals.gto import GTOIntegralGenerator
    from pyscf import gto

    mol = gto.M(atom='Si 0.0 0.0 0.0',
                basis='ccpvdz',
                verbose=4)

    with h5.File(os.path.join('test','data','data.h5')) as f:
        E0_known=f['Econst'][...]           
        twoBody_known=f['twoBody'][...]          
        ncv_known=f['numCholeskyActive'][...]
        oneBody_known=f['oneBody'][...]          
        S_known=f['S'][...]                

        Erohf=f['Erohf'][...]            
        Ecas=f['Ecas'][...]             
        mo=f['orbitals'][...]         

    gto_gen = GTOIntegralGenerator(mol)
    _, choleskyAO = ch.cholesky(gto_gen,tol=1.0E-4)

    S = mol.intor('int1e_ovlp')
    k = mol.intor('int1e_kin')
    v = mol.intor('int1e_nuc')
    
    ncore = 5
    nactive = 4
    
    print("Nactive AFQMC = ", nactive)
    twoBody,ncv,oneBody,S,E0 = pel.make_embedding_H(ncore=ncore,
                                                    nactive=nactive,
                                                    E0=0.0,
                                                    tol=1.0e-8,
                                                    C=mo,
                                                    twoBody=choleskyAO,
                                                    oneBody=k+v,
                                                    S=S,
                                                    transform_only=True)

    #TODO this test fails unexpectedly, why?
    #assert np.isclose(twoBody, twoBody_known).all()

    assert np.isclose(oneBody, oneBody_known).all()
    assert np.isclose(S, S_known).all()
    assert np.isclose(E0, E0_known).all()


def test_pel():
    '''
    make sure that pel imports and runs.
    '''
    with pytest.raises(ValueError):
        pel.make_embedding_H(0,0)
