import os
import numpy as np

from pyscf import gto,scf

from embedding import make_embedding_H, get_one_body, get_two_body


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
S, oneBodyAO = get_one_body(mol)

ncore = 1
M = S.shape[0]
Mactive = M - ncore

# make two-body integral Cholesky vectors
choleskyAO = get_two_body(mol)


twoBody,numCholeskyActive,oneBody,S,E_const = make_embedding_H(ncore=ncore,
                                                               nactive=Mactive,
                                                               E0=0.0,
                                                               tol=1.0e-8,
                                                               C=mo,
                                                               twoBody=choleskyAO,
                                                               oneBody=oneBodyAO,
                                                               S=S,
                                                               transform_only=True)

orbs = np.eye(Mactive)

#### Export to (available) AFQMC code(s) ####

# write GAFQMC output
if "USE_GAFQMC" in os.environ:
    import embedding.io.gafqmc as gio

    gio.write_orbs(orbs, Mactive, "O.eigen_gms")
    gio.save_oneBody_gms(Mactive, oneBody, S)
    gio.save_cholesky(numCholeskyActive, Mactive, twoBody)

# write AFQMCLab output
if "AFQMCLAB_DIR" in os.environ:
    from afqmclab.pyscfTools.model import writeModel
    from afqmclab.pyscfTools.rhf import writeSD2is

    nElec = (mol.nelec[0] - ncore,
             mol.nelec[1] - ncore)
    writeModel(nElec, oneBody, twoBody)
    writeSD2is(orbs, filename='phi.dat', nElec=nElec)
