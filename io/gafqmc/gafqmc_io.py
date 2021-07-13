import numpy as np

import pyqmc.matrices.gms as gms # written by Wirawan Purwanto
from V2b_inspect import load_V2b_dump, save_V2b_dump, sym_2d_unpack, sym_2d_pack # written by Wirawan Purwanto


def write_orbs(C, M, output, restricted=True): # simple wrapper function to interface with Wirawan's GAMESS format library
    '''
    \psi_i = \sum^M_mu C_{mu i} G_\mu  where G_\mu are GTOs and \psi is molecular orbital.
    
    C - matrix containing the molecular orbitals
    M - # GTO basis functions
    output - filename to save orbitals to (strongly recomended: end with '.eigen_gms')
    '''

    print("[+] Writing orbitals in \'GAMESS\' format to {}".format(output))
    O = gms.EigenGms()
    O.nbasis = M
    if restricted:
        O.alpha = C
    else:
        O.alpha = C[0]
        O.beta = C[1]
    O.write(output)
    print("[+] Write complete")

def save_cholesky(Ncv, M, CVlist, outfile="V2b_AO_cholesky.mat", verb=False):
    '''
    This function converts from a list of square-matrix Cholesky vectors, to a numpy
    array containing the lower diagonal (LD) form of each CV with shape = (Ncv, M*(M+1)/2). This is
    the format that the GAFQMC code will expect.
    '''
    # Q: Does this work with CVlist = [CV1, CV2, ... ,CV_Ncv] w. CV.shape = M*M (old method)
    #               and with CVlist = np.array() w/ shape (Ncv,M,M) (new method)
    #               it looks like it does, we have a 'reshape into the new shape
    #M = int(np.sqrt(2*CV_LD.shape[1]))
    CVarray = np.empty((Ncv, M*(M+1)//2))
    if verb:
        print (CVarray.shape)
    for i in range(Ncv):
        # convert from a 1-D vector, to a MxM matrix rep.
        # insert factor of 1/sqrt(2)        
        Lmat = CVlist[i].reshape((M,M))*(1/np.sqrt(2)) #TODO: remove reshape one np.array version fully implemented!
        if verb:
            print (i, Lmat.shape)
        sym_2d_pack(Lmat,CVarray[i])
    if verb:
        print ("Lmat shape = ", Lmat.shape)
    save_V2b_dump(CVarray.T, outfile, dbg=1)


def load_choleskyList_3_IndFormat(infile="V2b_AO_cholesky.mat",verb=False,is_complex=True):
    '''
    This function converts from a list of square-matrix Cholesky vectors, to a numpy
    array containing the lower diagonal (LD) form of each CV with shape = (Ncv, M*(M+1)/2). This is
    the format that the GAFQMC code will expect.
    '''
    if verb:
        print ("loading cholesky vectors")
    CV_LD = load_V2b_dump(infile).T
    M = int(np.sqrt(2*CV_LD.shape[1]))
    Ncv = CV_LD.shape[0]
    if verb:
        print ("M = ", M, ", Ncv = ", Ncv)
    CVarray = np.empty((Ncv, M, M))
    if is_complex:
        CVarrayDag = np.empty((Ncv, M, M))
    for i in range(Ncv):
        if verb:
            print ("vector ", i, flush=True)
        # convert from a 1-D vector, to a MxM matrix rep.
        # insert factor of sqrt(2) (pyscf and GAFQMC use different conventions concerning including/exclding the factor of 1/2 in the matrix elements        
        Lmat = sym_2d_unpack(CV_LD[i])*np.sqrt(2) #CVlist[i].reshape(M,M)*(1/np.sqrt(2)))
        if verb: 
            print (Lmat.shape)
        CVarray[i] = Lmat
        if is_complex:
            CVarrayDag[i] = Lmat.conj().T
    if is_complex:
        return M, Ncv, CVarray, CVarrayDag
    else:
        return M, Ncv, CVarray

def save_oneBody_gms(M, K, S, outfile='one_body_gms'):
    '''
    This function saves the overlap matrix (S) and the one-body Hamiltonian matrix elements (K)
    in the format used by GAFQMC ("GAMESS" format, one_body_gms).
    M is the number of basis set functions
    '''
    O = gms.OneBodyGms()
    O.nbasis = M
    O.S = S
    O.H1 = K
    O.write(outfile)

def load_oneBody_gms(infile='one_body_gms'):
    '''
    This function saves the overlap matrix (S) and the one-body Hamiltonian matrix elements (K)
    in the format used by GAFQMC ("GAMESS" format, one_body_gms).
    M is the number of basis set functions
    '''
    O = gms.OneBodyGms(infile)
    return O.nbasis, O.H1, O.S
