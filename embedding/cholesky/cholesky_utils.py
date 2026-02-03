"""
Cholesky Utilities

author: Kyle Eskridge

"""
import numpy as np

from embedding.lib.numpy_helper import einsum_optimized

def _ao2mo_cholesky_matmal(C,choleskyVecAO,verb=False):
    '''
    Transforms the GTO basis Cholesky vectors to the MO basis
    
    Inputs:
       C - coefficient matrix which specifies the desired MOs in terms of the GTO basis funcs
             (index conventions: C_{mu i} mu - GTO index, i - MO index)
       choleskyVecAO - numpy array containing the Cholesky vectors represented in the GTO basis

    index convention for CVs: choleskyVecAO[gamma, mu, nu]
                with gamma - Cholesky vector index
                     mu,nu - GTO indices 
           * similar for MO basis mu,nu -> i,l

    Returns:
       chleskyVecMO - numpy array containing the Cholesky vectros represented in the MO basis
    '''
    print('[+] transforming Cholesky vectors to MO basis',flush=True)
    ncv = choleskyVecAO.shape[0]
    MA = C.shape[1]
    nGTO, nactive = C.shape
    Cdag = C.conj().T # for readability below!
    choleskyVecMO = np.zeros((ncv,MA,MA))
    for i in np.arange(ncv):
        if verb:
            print(f'transforming vector {i}')
            if i % 100 == 0:
                print('',end='',flush=True)
        temp = np.matmul(Cdag,choleskyVecAO[i,:,:])
        choleskyVecMO[i,:,:] = np.matmul(temp,C)
    return choleskyVecMO


def _ao2mo_cholesky_einsum(C,choleskyVecAO,verb=False):
    '''
    Transforms the GTO basis Cholesky vectors to the MO basis
    
    Inputs:
       C - coefficient matrix which specifies the desired MOs in terms of the GTO basis funcs
             (index conventions: C_{mu i} mu - GTO index, i - MO index)
       choleskyVecAO - numpy array containing the Cholesky vectors represented in the GTO basis

    index convention for CVs: choleskyVecAO[gamma, mu, nu]
                with gamma - Cholesky vector index
                     mu,nu - GTO indices 
           * similar for MO basis mu,nu -> i,l

    Returns:
       chleskyVecMO - numpy array containing the Cholesky vectros represented in the MO basis
    '''

    print('[+] transforming Cholesky vectors to MO basis',flush=True)
    Cdag = C.conj().T
    return einsum_optimized('im,gmn,nj->gij',Cdag,choleskyVecAO,C,fname='ao2mo_cholesky_path.json') 

ao2mo_cholesky = _ao2mo_cholesky_matmal

def get_embedding_constant(C, Alist, AdagList, debug=False, is_complex=True):
    '''
    Computes the embedding constant from MO basis Cholesky vectors
    
    NOTES:- make cuts before calling in C, Alist, AdagList
          - no need for C, its assumed that Alist / AdagList as in correct basis
    
    Inputs:
    C - array containing just the inactive orbitals
    Alist, AdagList - restricted to frozen orbitals
    '''
    if is_complex:
        print('[+] computing <Vd> ...',flush=True)
        Vd = einsum_optimized('gii,gjj->',Alist,AdagList,fname='get_embedding_constant_Vd.json')

        print('[+] computing <Vx> ...',flush=True)
        Vx = einsum_optimized('gij,gji->',Alist,AdagList,fname='get_embedding_constant_Vx.json')

    else:
        print('[+] computing <Vd> ...',flush=True)
        Vd = einsum_optimized('gii,gjj->',Alist,Alist,fname='get_embedding_constant_Vd.json')

        print('[+] computing <Vx> ...',flush=True)
        Vx = einsum_optimized('gij,gji->',Alist,Alist,fname='get_embedding_constant_Vx.json')
    
    return 2*Vd - Vx 

def get_embedding_potential(nfc, C, Alist, AdagList, debug=False,is_complex=True):
    '''
    Computes the embedding potential from MO basis Cholesky vectors
    
    NOTES: make cuts before calling in C, Alist, AdagList
    '''
    
    if is_complex:
        G_core = np.eye(nfc,dtype='complex128')
        # compute the direct term as G_{I L} * V_{I j k L} -> Pyscf (Chemists') notation, want (IL|jk) mo integrals
        print('[+] computing Vd ...',flush=True)
        Vd = einsum_optimized('il,gil,gjk->jk',G_core,Alist[:,:nfc,:nfc],AdagList[:, nfc:, nfc:],fname='get_embedding_potential_Vd.json')

        # compute the exchange term as G_{I L} * V_{i J k L} -> Pyscf (Chemists') notation, want (iL|Jk) mo integrals
        print('[+] computing Vx ...')
        Vx = einsum_optimized('jl,gil,gjk->ik',G_core,Alist[:,nfc:,:nfc],AdagList[:,:nfc,nfc:],fname='get_embedding_potential_Vx.json')
    else:
        G_core = np.eye(nfc)
        # compute the direct term as G_{I L} * V_{I j k L} -> Pyscf (Chemists') notation, want (IL|jk) mo integrals
        print('[+] computing Vd ...')
        Vd = einsum_optimized('il,gil,gjk->jk',G_core,Alist[:,:nfc,:nfc],Alist[:, nfc:, nfc:],fname='get_embedding_potential_Vd.json')

        # compute the exchange term as G_{I L} * V_{i J k L} -> Pyscf (Chemists') notation, want (iL|Jk) mo integrals
        print('[+] computing Vx ...')
        Vx = einsum_optimized('jl,gil,gjk->ik',G_core,Alist[:,nfc:,:nfc],Alist[:,:nfc,nfc:],fname='get_embedding_potential_Vx.json')

    return 2*Vd - Vx
