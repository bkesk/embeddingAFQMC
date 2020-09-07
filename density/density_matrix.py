import numpy as np
import h5py as h5

from pyscf import gto

import pyqmc.matrices.gms as gms

def transform_dm(P, D, S):
    '''
    P_{mu nu} - density matrix in AO basis
    D_{i j} - coefficient matrix of orbitals represented in AO basis
    S_{mu nu} - overlap matrix of AO basis


    P'_ij = [D^dag S P S D]_ij 
    '''
    DdagS = np.matmul(D.conj().T,S)
    SD = np.matmul(S,D)
    Pnew = np.matmul(DdagS, P) 
    Pnew = np.matmul(Pnew, SD) 
    return Pnew

def make_rdm1(C, O):
    '''
    UNTESTED


    seems to have an error: when given CMO orbitsl (or LMO), comptued incorrect density
    '''
    print("[!] density_matrix.make_rdm1: Warning: not fully tested!")
    dm = np.zeros((C.shape[0], C.shape[0]))
    dm = np.einsum('ui,j->uj', C,O)
    print("DM = {}".format(dm))
    dm = np.matmul(dm,C.conj().T)
    print("DM = {}".format(dm))
    return dm

def make_Cbar(C,D,S):
    '''
    UNTESTED

    computes Cbar = [D^dag S C] where,

    C - coefficient matrix specifying Canonical orbitals (in GTO basis)
    D - coefficient matrix specifying Localized orbitals (in GTO basis)
    S - overlap matrix of GTO basis

    note: in reality C,D, and S just need to be in the same basis, not necesarily the GTO basis
    '''

    print("[!] density_matrix.make_Cbar: Warning: not fully tested!")
    
    Cbar = np.zeros((D.shape[1],C.shape[1]))
    Cbar = np.matmul(D.conj().T,S)
    Cbar = np.matmul(Cbar,C)
    return Cbar

def cross_ovlp(mol1,mol2):
    atm3, bas3, env3 = gto.conc_env(mol1._atm, mol1._bas, mol1._env,
                        mol2._atm, mol2._bas, mol2._env)
    cross_ovlp = gto.moleintor.getints('int1e_ovlp_sph', atm3, bas3, env3)
    return cross_ovlp

def cross_gto_rdm1(dm,cross_ovlp,debug=True):
    '''
    Project dm to a different basis

    Inputs cross_ovlp is the overlap matrix of Basis 1 (dm basis) concatenated with Basis 2 (new basis),
       the cross overlap matrix, Sbar has sectors:
        1.  Sbar11 - overlap matrix for Basis 1
        2.  Sbar22 - overlap matrix for Basis 2
        3.  Sbar12 - cross-basis overlap between 1 to 2
        3.  Sbar21 - cross-basis overlap between 2 to 1

    '''
    if debug:
        print(f' [ ] dm = {dm} \n    -- shape ={dm.shape} ')
    M1 = dm.shape[-1]
    Sbar21 = cross_ovlp[M1:,:M1]
    Sbar12 = cross_ovlp[:M1,M1:]
    if debug:
        
        print(f' [ ] shape of Sbar21 is {Sbar21.shape}')
        print(f' [ ] shape of Sbar12 is {Sbar12.shape}')
    P = np.matmul(Sbar21,dm)
    P = np.matmul(P,Sbar12)
    if debug:
        print(f' [ ] shape of P is {P.shape}')
        for d in P.shape[0]:
            print('trace of P is: {np.trace(d)}')
    return P

def check_symmetric(M, delta=1.0E-6, verb=False):
    #Test = M
    #Test[abs(Test) < delta] = 0.0
    

    max_dev = np.amax(np.abs(M - M.T))
    if verb:
        print("check_symmetric: max deviation is {}".format(max_dev))

    if max_dev < delta:
        return True
    else:
        return False

if __name__ == "__main__":
    print("[+] : testing ...")
    C = np.eye(10)
    print("C:\n",C)
    O = np.zeros(C.shape[0])
    for i in range(3):
        O[i] = 1.0
    for i in range(2):
        O[i+3] = 0.5
    print("O:\n",O)

    print(make_rdm1(C,O))
    print("Trace(dm) = {}".format(np.trace(make_rdm1(C,O))))
