import numpy as np
import h5py as h5

def ortho_check(mf,C=None,mol=None,verb=False):
    '''
    Check the orthonormality of the set of orbitals in C (assuming that C is given in terms of
    the AO basis instead of the orthogonalized basis
    '''

    if mol is None:
        mol = mf.mol

    # get C matrix
    if C is None:
        C = np.array(mf.mo_coeff)
    
    # get S matrix (that is, AO overlap matrix)
    Smat = mol.intor('int1e_ovlp')
    S = np.array(Smat)
    if verb:
        print("overlap:\n")
        print(S)
        
    O = np.matmul( np.matmul(C.conj().T,S),C) - np.eye(C.shape[0], C.shape[1])
    
    if verb:
        for i in range(O.shape[0]):
            for j in range(O.shape[1]):
                print(i,j,O[i,j])

    print("Max: ", O.max())

def copy_check(chk_original, chk_copy):
    import subprocess as sp
    sp.call(['cp',chk_original, chk_copy])

def write_to_erkale(mf, checkFile, conv=True): # function used to insert pyscf molecular orbitals into an
                                    # EXISTING ERKALE checkpoint file
    '''
    Write orbitals to an ERKALE checkpoint file;
    To greatly simpify this proces, we begin with an ERKALE checkpoint file from a run using identical:
              - atomic coordinates
              - Gaussian Type orbital basis
    however, the existing checkpoint file does not need to contain converged results. See the sample 
    run in 0.erkale-scf where the max. number of iterations is set to 2.
    
    inputs:

    - mf: a pyscf scf object, which has already been run (and, therefore, contains results)
    - checkpoint: the name of an ERKALE checkpoint file (see above)
    
    '''
    
    f = h5.File(checkFile, 'a')

    if conv:
        f['/Converged'][...] = 1
    
    r = f['/Restricted'][...]


    # note: normally, we have to worry about the relative GTO basis sign conventions
    #       between different quantum chem. codes. However, ERKALE and pyscf happen to
    #       use the same convention.

    if r == 1:
        C = mf.mo_coeff
        MOs = f['/C']
        MOs[...] = C.T # transpose, ERKALE and pyscf use opposite matrix ordering (i.e. 'C' vs. "Fortran')

        E = mf.mo_energy
        f['/E'][...] = E[0]

        # get # alpha electron:
        Na = mf.mol.nelec[0]
        Nb = mf.mol.nelec[1]
        Nset = f['/Nel']
        #Nset[...] = 2*Na # note: this is only true for closed shell, this is part of a hack for open-shells
        Nset[...] = Na + Nb

        #Nset_a = f['/Nel-a']
        #Nset_b = f['/Nel-b']
        f['/Nel-a'][...] = Na # mf.mol.nelec[0]
        f['/Nel-b'][...] = Nb # mf.mol.nelec[1]
        
        #Nset_a[...] = Na
        #Nset_b[...] = Na # this is what we meant to do! We intend to mislead ERKALE

        #O = np.zeros((C.shape[0]))
        #for i in range(Na):
        #    O[i] = 2
        O = mf.mo_occ #mf.mo_occ[0] + mf.mo_occ[1]
        print(O)
        f['/occs'][...] = O

    else:
        def write_ROHF(f, mf):
        # we may need to specify UHF/ROHF
        # the following is based on ROHF
            Rest = f['/Restricted']
            Rest[...] = 1 # this will tell ERKALE to treat as restricted
            
            C = mf.mo_coeff
            try:
                MOs = f['\C']
            except:
                MOs = f.create_dataset("C", C.T.shape)
            MOs[...] = C.T
            
            E = mf.mo_energy
            print(E)
            try:
                eigenE = f['/E']
            except:
                eigenE = f.create_dataset("E", (E.shape[0],1))
            eigenE[...] = np.array(E).T.reshape(E.shape[0],1)

            # get # alpha electron:
            Na = mf.mol.nelec[0]
            Nset = f['/Nel'][...] = 2*Na # note: this is only true for closed shell, this is part of a hack for open-shells

            f['/Nel-a'][...] = Na
            f['/Nel-b'][...] = Na # this is part of a hack
        
            occs = 2*mf.mo_occ[0]
            try:
                OCCs = f['/occs']
            except:
                OCCs = f.create_dataset("occs", occs.shape)
            OCCs[...] = occs
            
        def write_ROHF_ERKFORMAT(f, mf):
        # we may need to specify UHF/ROHF
        # the following is based on ROHF
            #Rest = f['/Restricted']
            #Rest[...] = 1 # this will tell ERKALE to treat as restricted
            
            # Here, we are taking pyscf's ROHF (both spin sectors have the same orbitals)
            # to ERKALE's format (spin sectors have different orbitals)

            C = mf.mo_coeff
            
            f['/Ca'][...] = C.T
            f['/Cb'][...] = C.T
            
            E = mf.mo_energy
            f['/Ea'][...] = np.array(E).T.reshape(E.shape[0],1)
            f['/Eb'][...] = np.array(E).T.reshape(E.shape[0],1)

            
        def write_UHF(f, mf):
            # NOTE: UNTESTED!
            Ca = mf.mo_coeff[0]
            Cb = mf.mo_coeff[1]
            MOs_a = f['/Ca']
            Mos_b = f['/Cb']
            MOs_a[...] = Ca.T
            Mos_b[...] = Cb.T

            Ea = mf.mo_energy[0]
            eigenEa = f['/Ea']
            eigenEa[...] = Ea
            Eb = mf.mo_energy[1]
            eigenEb = f['/Eb']
            eigenEb[...] = Eb
        
        write_ROHF_ERKFORMAT(f, mf)

    f.close()
    return None
