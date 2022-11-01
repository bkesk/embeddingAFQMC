import logging

import numpy as np

import embedding.cholesky as ch


def ao2mo_mat(C, mat):
    '''
    Transforms the GTO basis matrix to the MO basis
    
    Inputs:
       C - coefficient matrix which specifies the desired MOs in terms of the GTO basis funcs
             (index conventions: C_{mu i} mu - GTO index, i - MO index)
       mat - numpy matrix (num GTO x num GTO) represented in the GTO basis

    Returns:
       matMO - matrix in MO basis
    '''

    matMO = np.matmul(C.conj().T,mat)
    matMO = np.matmul(matMO,C)
    return matMO


def make_embedding_H(ncore=0,nactive=None,E0=0.0,tol=1.0e-6,C=None,twoBody=None,oneBody=None,S=None,transform_only=False):
    '''
    high level function to produce the embedding / downfolding Hamiltonian
   
    Required Inputs:
      - ncore (integer) : number of orbitals to freeze
      - nactive (integer): number of orbitals to treat as active in the embedding Hamiltonian
      - E0 (float) : the nuclear repulsion energy (in E_{Hartree}) plus any other constant energy terms
      - C (numpy.Array, float of complex) 
      - oneBody (numpy.Array, float of complex) 
      - twoBody (numpy.Array, float of complex) 
      - S (numpy.Array) 

    Optional Inputs:
      - transform_only (Boolean) : if False, a secondary Cholesky decomposition will be performed on the active space 
                                    two-body integrals. This is meant to reduce the overhead of 
      - tol (float) : tolerance for performing Cholesky decomposition on the active space two-body intergrals. 
                           IMPORTANT: tol should be greater than the tolerance of the original integrals. i.e. for
                           Cholesky vector inputs, tol should be greater than the original cholesky threshold.
 
    Returns:
        - twoBodyActive (numpy.Array shape (NcvActive,nactive,nactive) )
        - NcvActive (int)
        - oneBody_active (numpy.Array shape (nactive,nactive) )
        - S_active (numpy.Array shape (nactive,nactive) )
        - E_const (float) : contstant energy term (E0 + frozen orbital contribution)
    '''
    if C is None:
        raise ValueError("\"make_embedding_Hb\" requires C as input")
    
    if nactive is None:
        nactive = S.shape[0] - ncore

    is_complex = np.iscomplexobj(C)

    C = C[:,:ncore+nactive]
    
    Alist = ch.ao2mo_cholesky(C,twoBody)
    Ncv = twoBody.shape[0]

    if transform_only:
        print('Only transforming from GTO to orthonormal basis with no additional Cholesky decomposition', flush=True)
        NcvActive = Ncv
        twoBodyActive = Alist[:,ncore:,ncore:]
    else:
        from embedding.cholesky.integrals.factored import FactoredIntegralGenerator

        print(f'Performing Cholesky decomposition within the active space num. frozen occupied={ncore}, num. of active orbitals = {nactive}', flush=True)
        V = FactoredIntegralGenerator(Alist[:,ncore:,ncore:])
        NcvActive, twoBodyActive = ch.cholesky(integral_generator=V,tol=tol)
        del(V)

    print('Computing one-body embedding terms', flush=True)
    
    S_MO = ao2mo_mat(C,S)
    oneBody_MO = ao2mo_mat(C,oneBody)

    S_active = S_MO[ncore:,ncore:]
    oneBody_active = oneBody_MO[ncore:,ncore:]

    print(f'shape of oneBody_active is {oneBody_active.shape}')
    if ncore > 0:
        oneBody_active+=ch.get_embedding_potential(ncore, C, Alist, AdagList=None,is_complex=is_complex)
    
    print('Computing constant energy', flush=True)
    if ncore > 0:
        E_K = 2*np.einsum('ii->',oneBody_MO[:ncore,:ncore])
        E_V = ch.get_embedding_constant(C,Alist[:,:ncore,:ncore],AdagList=None,is_complex=is_complex)
    else:
        E_K=0.0
        E_V=0.0
    E_const = E0 + E_K + E_V
    print(f'E_0 = E_0 (input) + E_K + E_V = {E_const} with:\n  - E_0 (input) = {E0}\n  - E_K = {E_K}\n  - E_V = {E_V}')

    return twoBodyActive,NcvActive,oneBody_active,S_active,E_const


def get_one_body(mol):
    '''
    get one-body Hamiltonian, and basis overlap matrix.
    '''
    #TODO: if ecp, add to one-body Hamiltonian
    S = mol.intor('int1e_ovlp')
    k = mol.intor('int1e_kin')
    v = mol.intor('int1e_nuc')

    return S, k+v


def get_two_body(mol, tol=1.0E-6):
    '''
    Get Cholesky-decomposed two-body Coulomb interaction tensor in GTO basis.
    '''
    from embedding.cholesky import cholesky
    from embedding.cholesky.integrals.gto import GTOIntegralGenerator

    if mol.verbose > 4:
        logging.warn("mol has high verbosity: Cholesky output will be very large")

    gto_gen = GTOIntegralGenerator(mol)
    numcholesky,choleskyAO = cholesky(gto_gen,tol=tol)

    return choleskyAO
