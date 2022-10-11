import numpy as np

from .base import IntegralGenerator

class FactoredIntegralGenerator(IntegralGenerator):
    '''
    Class to generate Integrals based on a generic factored form of the Electron repulsion Integrals:
    
    V_{a b} =Sum_{j}^J  A^g_a * (A^g)^dag_b

    where V is the coloumb matrix, a,b are orbital pair indices, and the set of A vectors may be Cholesky vectors, 3-index Integrals from density fitting (see notes below), eigenvectors of V, or other factorizations of V.

    Notes: in the case of density fitting, one must take care to also factor in the auxiliary basis integrals as follows:
    
    let a=(i,l),b=(j,k) and mu,nu be auxiliary basis indices

    (il|jk) = Sum_{mu,nu} (il|mu)(mu|nu)(mu|jk) 
            = Sum_{mu,nu} B^mu_{il} v_{mu nu} (B^mu)^dag_{jk}
    
    where v_{mu nu} are the coloumb integrals of the aux. basis
    if we now decompse v (cholesky or diagonalization) s.t. 
    v_{mu nu} = Sum_{d} C^d_mu (C^d)^dag_nu then we can express V as:

    V = (a|b) = Sum_{d} A^d_a *(A^d)^dag_b 
        with A^d_a = Sum_mu B^mu_a C^d_mu
    '''
    def __init__(self,vectors,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.vectors = vectors
        self.nbasis = vectors.shape[1]
        
    def get_row(self, index,*args,**kwargs):
        if 'Alist' in kwargs:
            return factored_row(index,A=self.vectors,Adag=self.vectors) - factored_row(index,A=kwargs['Alist'],Adag=kwargs['Alist'])
        else:
            return factored_row(index,A=self.vectors,Adag=self.vectors)

    def get_diagonal(self,*args,**kwargs):
        return factored_diagonal(self.vectors,self.vectors)


def factored_row(index, A, Adag):
    M = A.shape[1]
    i = index // M
    l = index % M
    row = np.einsum('g,gjk',A[:,i,l],Adag)
    return row

def factored_diagonal(A,Adag):
    diag = np.einsum('gil,gil->il', A, Adag)
    return diag
