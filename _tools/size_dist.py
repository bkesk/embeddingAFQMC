import numpy as np

from pyscf import gto

def compute_sigma2(mol, C):
    '''
    Computes the second central moment orbital spread (sigma2) for each orbital in C
    sigma2 = sqrt(mu_2) with
    mu_2 = <(r - <r>)^2>
    
    STATUS:
    - we are computing centroids that agree with ERKALE, but mu_2 / sigma_2 do not agree
       - this confirms that we have the correct convention for C as well
       - also confirms that we are working in Bohr!

    '''
    M,N = C.shape
    
    C_dag = C.conj().T

    rsq = mol.intor('int1e_r2')
    r = mol.intor('int1e_r')
    
    print(f'\n\n{"i":>5} {"mu_2":>12} {"sigma_2":>12} {"Centroid (x,y,z)":>26}')
    for i in range(N):
        # compute <r>
        centroid = np.zeros(3)
        for a in range(3):
            centroid[a] = C_dag[i,:]*r[a]*C[:,i]
        # compute <r^2> - <r>^2
        '''expect_rsq = np.matmaul(C_dag[i,:],rsq)
        expect_rsq = np.matmaul(expect_rqs,C[:,i])
        print(f'expect_rsq = {expect_rsq} with shape {expect_rsq.shape}')
        centroid_sq = np.dot(centroid,centroid)'''
        mu_2 = (C_dag[i,:]*rsq*C[:,i] - np.dot(centroid,centroid))[0,0]
        #mu_2 = expect_rsq + 
        sigma_2 = np.sqrt(mu_2)
        print(f'{i:>5} {mu_2:>12.8e} {sigma_2:>12.8e} ({centroid[0]:>8.6e},{centroid[1]:>8.6e},{centroid[2]:>8.6e})')
        
