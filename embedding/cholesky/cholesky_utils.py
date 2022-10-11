import numpy as np

def ao2mo_cholesky(C,choleskyVecAO,verb=False):
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
    ncv = choleskyVecAO.shape[0]
    MA = C.shape[1]
    nGTO, nactive = C.shape
    Cdag = C.conj().T # for readability below!
    choleskyVecMO = np.zeros((ncv,MA,MA))
    for i in range(ncv):
        if verb:
            print(f'transforming vector {i}')
            if i % 100 == 0:
                print('',end='',flush=True)
        temp = np.matmul(Cdag,choleskyVecAO[i,:,:])
        choleskyVecMO[i,:,:] = np.matmul(temp,C)
    return choleskyVecMO

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
        Vd = np.einsum('gii,gjj->',Alist,AdagList,optimize='greedy')
        Vx = np.einsum('gij,gji->',Alist,AdagList,optimize='greedy')
    else:
        Vd = np.einsum('gii,gjj->',Alist,Alist,optimize='greedy')
        Vx = np.einsum('gij,gji->',Alist,Alist,optimize='greedy')
    
    return 2*Vd - Vx 

def get_embedding_potential(nfc, C, Alist, AdagList, debug=False,is_complex=True):
    '''
    Computes the embedding potential from MO basis Cholesky vectors
    
    NOTES: make cuts before calling in C, Alist, AdagList
    '''
    
    if is_complex:
        G_core = np.eye(nfc,dtype='complex128')
        # compute the direct term as G_{I L} * V_{I j k L} -> Pyscf (Chemists') notation, want (IL|jk) mo integrals
        print('[+] computing Vd ...')
        Vd = np.einsum('il,gil,gjk->jk',G_core,Alist[:,:nfc,:nfc],AdagList[:, nfc:, nfc:],optimize='greedy')

        # compute the exchange term as G_{I L} * V_{i J k L} -> Pyscf (Chemists') notation, want (iL|Jk) mo integrals
        print('[+] computing Vx ...')
        Vx = np.einsum('jl,gil,gjk->ik',G_core,Alist[:,nfc:,:nfc],AdagList[:,:nfc,nfc:],optimize='greedy')
    else:
        G_core = np.eye(nfc)
        # compute the direct term as G_{I L} * V_{I j k L} -> Pyscf (Chemists') notation, want (IL|jk) mo integrals
        print('[+] computing Vd ...')
        Vd = np.einsum('il,gil,gjk->jk',G_core,Alist[:,:nfc,:nfc],Alist[:, nfc:, nfc:],optimize='greedy')

        # compute the exchange term as G_{I L} * V_{i J k L} -> Pyscf (Chemists') notation, want (iL|Jk) mo integrals
        print('[+] computing Vx ...')
        Vx = np.einsum('jl,gil,gjk->ik',G_core,Alist[:,nfc:,:nfc],Alist[:,:nfc,nfc:],optimize='greedy')
    
    return 2*Vd - Vx
