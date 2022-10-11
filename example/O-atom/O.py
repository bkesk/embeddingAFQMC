import os
import numpy as np
import h5py as h5

from pyscf import gto,scf

# my libs
from embedding.cholesky.simple_cholesky import cholesky
from embedding.cholesky.integrals.gto import GTOIntegralGenerator

import embedding.cholesky as ch
import embedding as pel

if "USE_GAFQMC" in os.environ:
    import embedding.io.gafqmc as gio

use_afqmclab=False
if "AFQMCLAB_DIR" in os.environ:
    from afqmclab.pyscfTools.model import writeModel
    from afqmclab.pyscfTools.rhf import writeSD2is
    use_afqmclab=True
    
#### Normal PySCF ####
mol = gto.M(atom=[['O',(0.000,0.000,0.000)]],
            basis='ccpvdz',
            spin=2,
            verbose=4)

mf = scf.ROHF(mol)
mf.chkfile = 'ROHF.chk'
mf.kernel()
            
mo = mf.mo_coeff

#### Make H ####
S = mol.intor('int1e_ovlp')
k = mol.intor('int1e_kin')
v = mol.intor('int1e_nuc')

ncore = 1
M = S.shape[0]
Mactive = M - ncore

# example: for larger systems, it can be useful to save matrix gto-basis integrals
#    in case more than one choice of 'ncore' is used for the same the system
#run_chol=True
#if run_chol:
#    verb_back = mol.verbose
#    mol.verbose = 4
#
#    gto_gen = GTOIntegralGenerator(mol)
#    numcholesky, choleskyAO = cholesky(gto_gen,tol=1.0E-6)
#    M = choleskyAO.shape[-1]
#    mol.verbose = verb_back
#
#    f = h5.File('twoBodyAO','w')
#    f.create_dataset('twoBodyAO',data=choleskyAO)
#    f.close()

#f = h5.File('twoBodyAO','r')
#choleskyAO = f['twoBodyAO'][...]
#f.close()

# make two-body integral Cholesky vectors
verb_back = mol.verbose
mol.verbose = 4 # change verbosity, 'cholesky' is very verbose for verbose >= 5!
gto_gen = GTOIntegralGenerator(mol)
numcholesky,choleskyAO = cholesky(gto_gen,tol=1.0E-6)
M = choleskyAO.shape[-1]
mol.verbose = verb_back

twoBody,numCholeskyActive,oneBody,S,E_const = pel.make_embedding_H(nfc=ncore,
                                                                   nactive=Mactive,
                                                                   Enuc=0.0,
                                                                   tol=1.0e-8,
                                                                   C=mo,
                                                                   twoBody=choleskyAO,
                                                                   oneBody=k+v,
                                                                   S=S,
                                                                   transform_only=True)

orbs = np.eye(Mactive)

# write GAFQMC output
if "USE_GAFQMC" in os.environ:
    gio.write_orbs(orbs, Mactive, "O.eigen_gms")
    gio.save_oneBody_gms(Mactive, oneBody, S)
    gio.save_cholesky(numCholeskyActive, Mactive, twoBody)

# for comparison, also save in AFQMCLAB format, if installed.
if use_afqmclab:
    nElec = (mol.nelec[0] - ncore,
             mol.nelec[1] - ncore)
    writeModel(nElec, oneBody, twoBody)
    writeSD2is(orbs, filename='phi.dat', nElec=nElec)
