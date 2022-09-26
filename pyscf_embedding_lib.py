import numpy as np
import h5py as h5
from pyscf import lib, gto, scf, ao2mo # importing specific pyscf submodules

# the following are libraries written by us
import pyqmc.matrices.gms as gms # written by Wirawan Purwanto to all read/write to the GAMESS file format for 1-body, and single particle orbitals.
# submodules - use relative path
import embedding.cholesky as ch # written by Kyle Eskridge to implement a Cholesky decomp. in pyscf, along with other support functions.
import embedding.density as dmat

from embedding.cholesky.integrals.factored import FactoredIntegralGenerator
from embedding.cholesky.simple_cholesky import cholesky

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
        
    O = np.matmul( np.matmul(C.conj().T,S),C) - np.eye(C.shape[0], C.shape[1]) # This one seems right
    #O = C*S*C.conj() - np.eye(C.shape[0], C.shape[1])  # I think this is the wrong conv.
    

    if verb:
        for i in range(O.shape[0]):
            for j in range(O.shape[1]):
                print(i,j,O[i,j])

    print("Max: ", O.max())

def copy_check(chk_original, chk_copy):
    import subprocess as sp
    sp.call(['cp',chk_original, chk_copy])

def dm_swap(mf, swapInds):
    '''
    Need a previously-run mf object (preferably 'converged')
    '''
    occ = mf.mo_coeff
    print("Coeffs", occ)

    for pair in swapInds:
        occ[pair[0], :], occ[pair[1], :] = occ[pair[1], :], occ[pair[0], :]
        
    dm = mf.make_rdm1(mo_coeff=occ)
    print("DM : ", dm)
    return dm

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
        #eigenE = f['/E']
        #eigenE[...] = E[0]
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

def make_rdm1_localized_fragments(mf, basis_ename, chk1, chk2, useAlpha=True, verb=False):
    '''
    This funciton constructs a density matrix from molecular fragments within the
    active space of the conbined system.
    
    DEV:
      need:
        - new basis orbitals represented in AO basis
        - orbitals for each subsystem represented in SAME AO basis
    '''

    ## Another point, try to normalize the orbitals in the new basis
    
    # get basis orbitals
    eigen = gms.EigenGms()
    eigen.read(basis_ename, verbose=verb)
    
    if useAlpha:
        C_basis = eigen.alpha.T
    else:
        C_basis = eigen.beta.T
    
    if verb:
        print("[+] trace of C_basis = {}".format(np.trace(C_basis)))

    #C_new = np.zeros(C_basis.shape)

    # get fragment orbitals
    f1 = h5.File(chk1, 'r')
    C1 = f1['/scf/mo_coeff'][...]
    occ1 = f1['/scf/mo_occ'][...]
    M1 = C1.shape[0]
    if verb:
        print("M1: ", M1)
    f1.close()
        
    mol1 = lib.chkfile.load_mol(chk1)

    print(mol1.atom)
    mol1._atom[0] = [0.0, 0.0, 1.5*1.88973] # quick hack to adjust O atom position relative to H chain

    if verb:
        print("[+] orbital occupancy for fragment a: {}".format(occ1))

    f2 = h5.File(chk2, 'r')
    C2 = f2['/scf/mo_coeff'][...]
    occ2 = f2['/scf/mo_occ'][...]
    M2 = C2.shape[0]
    if verb:
        print("M2: ", M2)
    f2.close()

    mol2 = lib.chkfile.load_mol(chk2)

    # retrieving the overlap matrix for the combined system.
    #atm, bas, env = gto.conc_env(mol1._atm, mol1._bas. mol1._env,
    #                             mol2._atm, mol2._bas. mol2._env)
    mol = gto.conc_mol(mol1, mol2)
    S = gto.moleintor.getints('int1e_ovlp_sph', mol._atm, mol._bas, mol._env)

    print("Smatrix = {}".format(S))

    if verb:
        print("[+] orbital occupancy for fragment b: {}".format(occ2))

    #Transform to new basis
    M = M1+M2
    if verb:
        print("M: ", M)
    Ca = np.zeros((M,M))
    Ca[:M1,:M1] = C1
    Ca = C_basis.conj().T*S*Ca
    #Ca = Ca*S*C_basis.conj().T
    if verb:
        print("[+] trace of Ca = {}".format(np.trace(Ca)))

    Cb = np.zeros((M,M))
    Cb[M1:,M1:] = C2
    Cb = C_basis.conj().T*S*Cb
    #Cb = Cb*S*C_basis.conj().T
    if verb:
        print("[+] trace of Cb = {}".format(np.trace(Cb)))
    
    occa = np.zeros((M))
    occa[:M1] = occ1
    occb = np.zeros((M))
    occb[M1:] = occ2
    
    if verb:
        print("[+] occa = {}".format(occa))
        print("[+] occb = {}".format(occb))

    # construct density matrix for each fragment
    dm_a = mf.make_rdm1(mo_coeff=Ca.T, mo_occ=occa)
    if verb:
        print("[+] fragment a DM: ", dm_a)
        print("[+] trace of DM_a alpha = {}".format(np.trace(dm_a[0])))
        print("[+] trace of DM_a beta = {}".format(np.trace(dm_a[1])))

    dm_b = mf.make_rdm1(mo_coeff=Cb.T, mo_occ=occb)
    if verb:
        print("[+] fragment b DM: ", dm_b)
        print("[+] trace of DM_b alpha = {}".format(np.trace(dm_b[0])))
        print("[+] trace of DM_b beta = {}".format(np.trace(dm_b[1])))

    # TEMP for testing!
    '''
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm

    # TEMP for testing!
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(1,4)

    pic_a = ax1.imshow(dm_a[0], cmap=cm.gist_rainbow)
    cbar = fig.colorbar(pic_a, ax=ax1)

    pic_b = ax2.imshow(dm_a[1], cmap=cm.gist_rainbow)
    cbar_b = fig.colorbar(pic_b, ax=ax2)

    pic_a1 = ax3.imshow(dm_b[0], cmap=cm.gist_rainbow)
    cbar_1 = fig.colorbar(pic_a1, ax=ax3)

    pic_b1 = ax4.imshow(dm_b[1], cmap=cm.gist_rainbow)
    cbar_b1 = fig.colorbar(pic_b1, ax=ax4)

    plt.show()
    '''

    def print_mat(C):
        dims = C.shape
        print("mat dims =", dims)
        for i in range(dims[0]):
            for j in range(dims[1]):
                print(i,j,C[i,j])
    #if verb:
    #    print "[+] printing Ca matrix"
    #    print_mat(Ca)
    #    print "[+] printing Cb matrix"
    #    print_mat(Cb)

    dm = dm_a + dm_b # mf.make_rdm1(mo_coeff=C, mo_occ=occ)
    if verb:
        print("DM: ", dm)
        print("[+] make_rdm1_localized_fragments : saving debug info to debug.h5")
        dbg = h5.File("debug.h5")
        try:
            dbg.create_dataset("DM_a", data=dm_a)
            dbg.create_dataset("DM_b", data=dm_b)
            dbg.create_dataset("DM", data=dm)
            dbg.create_dataset("Ca", data=Ca)
            dbg.create_dataset("Cb", data=Cb)
        except:
            dbg["DM_a"][...]=dm_a
            dbg["DM_b"][...]=dm_b
            dbg["DM"][...]=dm
            dbg["Ca"][...]=Ca
            dbg["Cb"][...]=Cb
        dbg.close()

    return dm

def make_rdm1_fragment(mf, chk1, chk2, verb=False):
    f1 = h5.File(chk1, 'r')
    C1 = f1['/scf/mo_coeff'][...]
    occ1 = f1['/scf/mo_occ'][...]
    M1 = C1.shape[0]
    if verb:
        print("M1: ", M1)
    f1.close()

    f2 = h5.File(chk2, 'r')
    C2 = f2['/scf/mo_coeff'][...]
    occ2 = f2['/scf/mo_occ'][...]
    M2 = C2.shape[0]
    if verb:
        print("M2: ", M2)
    f2.close()

    M = M1+M2
    if verb:
        print("M: ", M)
    C = np.zeros((M,M))
    C[:M1,:M1] = C1
    C[M1:,M1:] = C2
    
    occ = np.zeros((M))
    occ[:M1] = occ1
    occ[M1:] = occ2

    def print_mat(C):
        dims = C.shape
        print("mat dims =", dims)
        for i in range(dims[0]):
            for j in range(dims[1]):
                print(i,j,C[i,j])
    if verb:
        print("make_rdm1_fragment: printing C matrix")
        print_mat(C)
        print("make_rdm1_fragment: printing occupancy vector")
        for i,o in enumerate(occ):
            print(f'{i} {o}')

    print(f' C*occ = {C*occ} \n tr(C*occ)={np.trace(C*occ)}')

    dm = np.dot(C.conj().T*occ, C)
    #dm = np.einsum('mi,i,')

    #dm = mf.make_rdm1(mo_coeff=C, mo_occ=occ)
    if verb:
        print("DM: ", dm)
    return np.array(dm)

def make_rdm1_fragment_UHF(mf, chk1, chk2, verb=False):
    '''
    NOT IMPLEMENTED!!! as of March 23, 2020
    '''
    f1 = h5.File(chk1, 'r')
    C1 = f1['/scf/mo_coeff'][...]
    occ1 = f1['/scf/mo_occ'][...]
    M1 = C1.shape[0]
    if verb:
        print("M1: ", M1)
    f1.close()

    f2 = h5.File(chk2, 'r')
    C2 = f2['/scf/mo_coeff'][...]
    occ2 = f2['/scf/mo_occ'][...]
    M2 = C2.shape[0]
    if verb:
        print("M2: ", M2)
    f2.close()

    M = M1+M2
    if verb:
        print("M: ", M)
    C = np.zeros((M,M))
    C[:M1,:M1] = C1
    C[M1:,M1:] = C2
    
    occ = np.zeros((M))
    occ[:M1] = occ1
    occ[M1:] = occ2

    def print_mat(C):
        dims = C.shape
        print("mat dims =", dims)
        for i in range(dims[0]):
            for j in range(dims[1]):
                print(i,j,C[i,j])
    if verb:
        print("printing C matrix")
        print_mat(C)

    dm = mf.make_rdm1(mo_coeff=C, mo_occ=occ)
    if verb:
        print("DM: ", dm)
    return dm


def make_rdm1_pyscf_chk(mf, chk, verb=False):
    f = h5.File(chk, 'r')
    C = f['/scf/mo_coeff'][...]
    occ = f['/scf/mo_occ'][...]
    M = C.shape[0]
    if verb:
        print("M: ", M)
    f.close()

    def print_mat(C):
        dims = C.shape
        print("mat dims =", dims)
        for i in range(dims[0]):
            for j in range(dims[1]):
                print(i,j,C[i,j])
    if verb:
        print("printing C matrix from {}".format(chk))
        print_mat(C)

    dm = mf.make_rdm1(mo_coeff=C, mo_occ=occ)
    if verb:
        print("DM: ", dm)
    return dm

# custom H tools:

def make_rdm1_from_resEigenGms(ename, nelec, mol, verb=False):
    '''
    This function takes in orbitals from an ".eigen_gms" file named 'ename'
    and constructs a reduced density matrix to use as an inital guess in pyscf.

    This function will only use alpha orbitals and will treat the orbitals as restricted
    regardless of the absence or presence of beta orbitals

    inputs:
    - ename - name of eigen_gms file containing the orbitals
    - nelec - a tuple with two entries, the number of alpha, and beta electrons, respectively
    - verb - boolean, turns on/off vebose mode

    '''
    # read alpha orbitals from eigen_gms
    eigen = gms.EigenGms()
    eigen.read(ename, verbose=verb)
    
    C = eigen.alpha    
    
    mf = scf.ROHF(mol)
    mo_occs = np.zeros((C.shape[0]))

    # set occupancies
    for a in range(nelec[0]):
        mo_occs[a] += 1.0
    for b in range(nelec[1]):
        mo_occs[b] += 1.0

    if verb:
        print("occs: ", mo_occs)
        
    dm = mf.make_rdm1(np.array(C), mo_occs)

    if verb:
        print("dm: ", dm)
    return dm

def make_rdm1_from_erkaleOrbs(ename, nelec, mol, verb=False):
    '''
    This function produces the density matrix from ERKALE's ROHF orbitals.
    ERKALE allows the ROHF orbitals to differ between spin sectors (although these
    orbitals are unitarily equivalent to the case in which both spin sectors are the same)

    Pyscf's ROHF implementation assumes that the orbitals are the same.

    Here, we construct the density matrix in the basis of ERKALE alpha orbitals using the UHF
    engine in pyscf. Since the density matrix is invariant under a unitary transformation of the 
    orbitals which preserves the occupied-virtual partioning, we should arrive at the true ROHF 
    density matrix if we use the ERKALE ROHF orbitals.
    '''
    # read alpha and beta orbitals from eigen_gms
    eigen = gms.EigenGms()
    eigen.read(ename, verbose=verb)
    
    Ca = eigen.alpha    
    Cb = eigen.beta
    
    mf_u = scf.UHF(mol)
    mo_u_occs = np.zeros((2, Ca.shape[0]))

    # set occupancies
    for a in range(nelec[0]):
        mo_u_occs[0][a] = 1.0
    
    for b in range(nelec[1]):
        mo_u_occs[1][b] = 1.0
    
    #TEST!!
    #n = np.matmul(Cb.conj().T[:,:nelec[1]],Cb[:nelec[1]])
    #n = np.matmul(Cb[:,:nelec[1]],Cb.conj().T[:nelec[1]])
    #for b in range(Cb.shape[0]):
    #    mo_u_occs[1][b] = n[b,b]

    if verb:
        print("occs: ", mo_u_occs)
        
    dm = mf_u.make_rdm1(mo_coeff=np.array([Ca, Cb]), mo_occ=mo_u_occs)

    def print_mat(C):
        dims = C.shape
        print("mat dims =", dims)
        for i in range(dims[0]):
            for j in range(dims[1]):
                print(C[i,j],end='')
            print("")
    
    if verb:
        print("[+] make_rdm1_from_erkaleOrbs - dm: ")
        print("   dm_Alpha")
        print_mat(dm[0])
        print("   dm_Beta")
        print_mat(dm[1])
    return dm

def custom_jk(mol, dm, *args):
    '''
    This function allows a custom 2-Body Hamiltonian, V, expressed as a factored form,
    V = sum_g A^g (A^g)^dagger (where A are one-body operators)
    with no need to completely recompute the full 2-Body tensor.
    
    Computing J and K from Cholesky vectors, A, (einstein convention implied!):

    J_il = (A g il) (A^dag g jk) * (P kj)
    K_jl =  (A^dag g jk) * (P ki) * (A g il)

    define: (T g ji) = (A^dag g jk) * (P ki)

    then:
    J_il = (A g il) (T g jj)
    K_jl = (A g il) (T g ji)
    '''
   
    All = ch.load_choleskyList_3_IndFormat()
    # currently, we are loading A from memory each time get_jk is called (every HF iteration)
    # it may be possible to give the scf object aribtrary attributes
    # which would allow the following A and Adag matricies to be stored
    A = All[2]
    Adag = All[3]
    
    if mol.verbose > 4:
        print("[+] custom_jk: shape of Adag {}".format(Adag.shape))
        print("[+] custom_jk: shape of dm {}".format(dm.shape))

    T = np.einsum('gjk,ki->gji',Adag,dm[0]) # test! using dm[0] instead of dm
    
    J = np.einsum('gil,gjj->il',A,T)
    K = np.einsum('gil,gji->jl',A,T)
    
    return J, K


def custom_jk_uhf(mol, dm, V2b_file='V2b_AO_cholesky.mat', *args):
    '''
    This function allows a custom 2-Body Hamiltonian, V, expressed as a factored form,
    V = sum_g A^g (A^g)^dagger (where A are one-body operators)
    with no need to completely recompute the full 2-Body tensor.
   
    Computing J and K from Cholesky vectors, A, (einstein convention implied!):

    J_il = (A g il) (A^dag g jk) * (P kj)
    K_jl =  (A^dag g jk) * (P ki) * (A g il)

    define: (T g ji) = (A^dag g jk) * (P ki)

    then:
    J_il = (A g il) (T g jj)
    K_jl = (A g il) (T g ji)

    inputs:
    mol - currently, we are just reading the verbosity setting (TODO update to bool)
    dm - reduces single particle density matrix (should have format dm=[dm_a,dm_b])
    
    '''
    All = ch.load_choleskyList_3_IndFormat(infile=V2b_file)
    # currently, we are loading A from memory each time get_jk is called (every HF iteration)
    # it may be possible to give the scf object aribtrary attributes
    # which would allow the following A and Adag matricies to be stored
    A = All[2]
    Adag = All[3]
    
    if mol.verbose > 4:
        print("[+] custom_jk_uhf: shape of Adag {}".format(Adag.shape))
        print("[+] custom_jk_uhf: shape of dm {}".format(dm.shape))

    T_a = np.einsum('gjk,ki->gji',Adag,dm[0],optimize='greedy') # test! using dm[0] instead of dm
    T_b = np.einsum('gjk,ki->gji',Adag,dm[1],optimize='greedy')

    J_a = np.einsum('gil,gjj->il',A,T_a,optimize='greedy')
    K_a = np.einsum('gil,gji->jl',A,T_a,optimize='greedy')
    
    J_b = np.einsum('gil,gjj->il',A,T_b,optimize='greedy')
    K_b = np.einsum('gil,gji->jl',A,T_b,optimize='greedy')

    J=np.array([J_a,J_b])
    K=np.array([K_a,K_b])

    return J, K


def customH_mf(mf, EnucRep, on_the_fly=True, dm_file=None, N_frozen_occ=None, dm=None, one_body_file='one_body_gms', V2b_file='V2b_AO_cholesky.mat', verb=4):

    # TODO: Known bug - the dm is (sometimes) stored as [dm_a,dm_b] instead of a single matrix
    #                   need to implement a check for this and convert to single matrix

    '''
    This is a wrapper function which sets up, and runs, a mean-field calculation using a 
    custom Hamiltonian. The custom Hamiltonian is given in the format that is used by GAFQMC 
    (i.e. one_body_gms, and V2b_AO_cholesky.mat for one- and two-body terms respectively).
    This is a basis-set agnostic process and only requires that the one- and two-body terms
    be represented in the same basis. If a density matrix (dm) is specified as an initial guess,
    care should be taken to ensure that it is also in the same basis as the one- and two-body terms.
    
    Note: the number of electrons, and spin should be specified already in the pyscf scf object, mf.
    
    sample call:

    >mol = gto.M(spin=0,verbose=5)
    >mol.nelectron = 28 # note: this needs to be the total number of electrons (not N/spin)
    >
    >mf = scf.RHF(mol)
    >
    >EnucRep = 49.4679845596581
    >customH_mf(mf, EnucRep)

    inputs:
    mf - a pyscf scf object (scf.RHF(), scf.UHF(), etc)
    EnucRep - the constant nuclear repulsion energy
    on_the_fly - boolean, determines if teh full two-body tensor is stored, or if need info is
                 computed on the fly
    dm - density matrix use as an initial guess
    one_body_file - the file containing the overlap (S) and one-body Hamiltonian (h1) in GAMESS foramt
    V2b_file - the binary file containing the one-body operators that for a factore rep. of the 
               two-body interaction term.
    verb - sets the verbosity level, passed directly to pyscf functions. Also, for verb > 4 ,                     additionl output is generatd by this function.

    '''

    E=0

    M, h1, S = ch.load_oneBody_gms(one_body_file)

    if verb > 4:
        print("M = ", M)
        print("H1 ", h1)
        print("S ", S)

    # get the eri's:
    #eri_ext = load_V2b_dump("V2b_AO_cholesky.mat").T
    
    #mf = scf.density_fit(scf.RHF(mol))
    #mf = scf.RHF(mol)
    mf.chkfile = "./imported-Hamiltonian.chk"
    mf.get_hcore = lambda *args: h1
    mf.get_ovlp = lambda *args: S
    mf.energy_nuc = lambda *args: EnucRep

    if verb >=5:
        dbg = h5.File("debug.h5",'w')
        try:
            dbg.create_dataset("h1", data=h1)
            dbg.create_dataset("S", data=S)
            dbg.create_dataset("EnucRep", data=EnucRep)
        except:
            dbg["h1"][...]=h1
            dbg["S"][...]=S
            dbg["EnucRep"][...]=EnucRep
        dbg.close()

    if on_the_fly:
        if verb >= 4:
            print("==== computing two-body interactions on the fly ====",flush=True)
            
        def _custom_jk(mol,dm,*args):
            return custom_jk_uhf(mol, dm, V2b_file=V2b_file)

        mf.get_jk = _custom_jk
    else:
        if verb >= 4:
            print("==== computing and storing two-body tensor in memory ====",flush=True)
        Alist = ch.load_choleskyList_GAFQMCformat(infile=V2b_file, verb =(verb>4))
        fERI = ch.factoredERIs_updateable(Alist[2], M, verb=(verb>4))
        eri = fERI.full()
        mf._eri = eri
        print("Shape of mf._eri is {}".format(mf._eri.shape))

    if dm is not None:
        if verb >= 4:
            print("[+] customH_mf : using provided DM as initial guess")

            print("[+] customH_mf : checking integrity of provided density matrix")
            print("  [+] Trace of the density matrix alpha sector = {}".format(np.trace(dm[0])))
            print("  [+] Trace of the density matrix beta sector = {}".format(np.trace(dm[1])),flush=True)

            print("DM provided -> Electronic Energy: {}".format(mf.energy_elec(dm=dm)),flush=True)
        E=mf.kernel(dm)
    elif dm_file is not None and N_frozen_occ is not None:
        if verb >= 4:
            print("[+] customH_mf : using DM from {} with N_frozen_occ = {}".format(dm_file,N_frozen_occ),flush=True)
        dm_full = get_dm_from_h5(dm_file, verb=(verb>=4))
        # need the number of frozen occupied orbitals to get the correct slice from the
        #   full density matrix
        dm = [dm_full[0][N_frozen_occ:M+N_frozen_occ,N_frozen_occ:M+N_frozen_occ],
              dm_full[1][N_frozen_occ:M+N_frozen_occ,N_frozen_occ:M+N_frozen_occ]]
        print("[+] customH_mf : checking integrity of truncated density matrix")
        print("  [+] Trace of the truncated density matrix alpha sector = {}".format(np.trace(dm[0])))
        print("  [+] Trace of the truncated density matrix beta sector = {}".format(np.trace(dm[1])),flush=True)
        E=mf.kernel(dm)
    else:
        if verb >= 4:
            print("[+] customH_mf : no DM provided, using Pyscf default Guess",flush=True)
        E=mf.kernel()

    return E
# Spin consistent basis - prototype:

def make_spin_consistent_basis(nameErkChk, debug=False):
    '''

    NOTE: not finishes or tested! - Kyle: Jan 16th, 2020
    
    Creates the set of 'spin consistent' orbitals (see F. Ma., et. al. in Phys. Rev. Lett. 114, 226401, 2015)
    by diagonalizing the 2M x 2M overlap matrix formed by the concatonated set alpha and beta orbitals
    
    O_(sigma,i)(sigma',j) = < Xi^sigma_i | Xi^sigma'_j >  where | Xi^sigma_i > is a spin-orbital

    The orbitals are read in from an ERKALE checkpoint file, and an MxM coefficient matrix, C is output which
    gives the rep. of the spin consistent orbitals in terms of the GTO basis set functions.
    (i.e. psi_i = Sum_mu C_mu_i * g_mu where g_mu is a GTO basis set function) 

    Input:
    - nameErkChk : name of ERKALE checkpoint file to read Ca, Cb, and S matricies from

    Returns:
    - C : the orbital coefficient matrix, expressed in the GTO basis, for the spin-consistent orbitals
    '''
    
    def comp_overlap(C1, C2, S):
        O = np.zeros(S.shape)
        O = np.matmul(C1.conj().T, S)
        O = np.matmul(O, C2)
    
    f = h5.File(nameErkChk,'r')

    try:
        Ca = f['/Ca'][...]
        Cb = f['/Cb'][...]
    except:
        print("ERKALE checkpoint file does not contain spin-polarized orbitals")
        return None

    S = f['/S'][...]
    M = S.shape[0] # num. basis functions

    O = np.zeros((2*M,2*M))
    
    # first, a test

    O[:M,:M] = comp_overlap(Ca, Ca, S)
    O[M:,M:] = comp_overlap(Cb, Cb, S)
    if debug:    
        dbg = h5.File("debug.h5",'w')
        dbg.create_dataset("Overlap", data=O)
        dbg.close()

    f.close()

def get_dm_from_h5(h5name, verb=False):
    f = h5.File(h5name,'r')
    dma = f['/dm_a'][...]
    dmb = f['/dm_b'][...]
    f.close()
    
    if verb:
        print("[+] get_dm_from_h5 : checking integrity of density matrix in {}".format(h5name))
        print("  [+] Trace of the density matrix alpha sector = {}".format(np.trace(dma)))
        print("  [+] Trace of the density matrix beta sector = {}".format(np.trace(dmb)))

    return [dma,dmb]

def write_rdm1(name,dm,restr=True):
    if restr:
        dma = dm
        dmb = dm
    else:
        dma = dm[0]
        dmb = dm[1]
    f = h5.File(name,'w')
    try:
        f.create_dataset("dm_a", data=dma)
        f.create_dataset("dm_b", data=dmb)
        f.create_dataset("dm", data=dma + dmb)
    except:
        f["dm_a"][...]= dma
        f["dm_b"][...]= dmb
        f["dm"][...]= dma + dmb
    f.close()

def transform_rdm1(mf, dm, gms_name, mol, restr_basis=True):
    '''
    For now, it is assumed that dm = [dma, dmb]
    '''
    LMO = gms.EigenGms()
    LMO.read(gms_name)
    # need to get overlap matrix
    S = ch.get_ovlp(mol)

    if restr_basis:
        D = LMO.alpha
        print("[+] checking orthonormality of localized orbitals: ")
        ortho_check(mf,C=D)
        dm_new = np.array([dmat.transform_dm(dm[0], D, S), dmat.transform_dm(dm[1], D, S)])
    else:
        D = [LMO.alpha, LMO.beta]
        print("[+] checking orthonormality of localized alhpa orbitals: ")
        ortho_check(mf,C=D[0])
        print("[+] checking orthonormality of localized beta orbitals: ")
        ortho_check(mf,C=D[1])
        dm_new = np.array([dmat.transform_dm(dm[0], D[0], S), dmat.transform_dm(dm[1], D[1], S)])

    return dm_new

def check_rdm1(dm):
    print(("[+] Trace dm_a = {}".format(np.trace(dm[0]))))
    print(("[+] Trace dm_b = {}".format(np.trace(dm[1]))))
    print(("[+] Trace dm = {}".format(np.trace(dm[0] + dm[1]))))

    print(("dm_a is symmetric? - {}".format(dmat.check_symmetric(dm[0], verb=True))))
    print(("dm_b is symmetric? - {}".format(dmat.check_symmetric(dm[1], verb=True))))
    print(("dm is symmetric? - {}".format(dmat.check_symmetric(dm[0]+dm[1], verb=True))))

def make_transformed_eigen(C, S, outname=None, restricted=True):
    '''
    Tranforms the coefficient matrix, C, to the MO basis

    psi_i = Sum_mu C_{mu i} g_mu
    
    we are changing basis to:

    psi_i = Sum_j C^bar_{j i} psi_j 
    
    C^bar should be a trivial diagonal matrix for the alpha sectro.
    It can be computed as: C^bar = C^dag S C, where S is the GTO basis overlap matrix
    '''
    
    Cbar = np.zeros(C.shape)
    Cbar = C.conj().T*S*C
    
    if outname is not None:
        write_orbs(Cbar, M=Cbar.shape[0], output=outname, restricted=restricted)

    return Cbar

def transform_wf(Wf, C, S, outname=None, restricted=True):
    '''
    Tranforms the given wavefunction, Wf, to the basis given by coefficient matrix, C, 
        where C is expressed in terms of the GTOs

    Transformation can be computed as: C^bar = C^dag S Wf, where S is the GTO basis overlap matrix
    '''
    
    if restricted:
        #Wf_bar = np.zeros(C.shape) # why?
        Wf_bar = C.conj().T*S*Wf
    else:
        #Wup = np.zeros(C.shape)
        print(f'C.conj().T.shape is {C.conj().T.shape}')
        print(f'S.shape is {S.shape}')
        print(f'Wf[0].shape is {Wf[0].shape}')
        Wup = np.dot(C.conj().T,np.dot(S,Wf[0]))

        #Wdown = np.zeros(C.shape)
        #Wdown = C.conj().T*S*Wf[1]
        Wdown = np.dot(C.conj().T,np.dot(S,Wf[1]))
        Wf_bar = [Wup,Wdown]

    if outname is not None:
        write_orbs(Wf_bar, M=C.shape[0], output=outname, restricted=restricted)

    return Wf_bar


def make_embedding_H(nfc,ntrim,Enuc,tol=1.0e-6,ename='eigen_gms',V2b_source='V2b_AO_cholesky.mat',V1b_source='one_body_gms',transform_only=False,debug=False,mol=None,is_complex=False):
    '''
    high level function to produce the embedding / downfolding Hamiltonian
    saves the results to files

    gets CVs from V2b_source, which chould contain the GTO basis CVs
    similarly for V1b_source

    Inputs:
    
    Outputs:
    > saves the following files
    '''

    #TODO: remove the mol input, depends on some changes in the simple_cholesky module

    # 1. read in orbitals (go ahead and remove the last ntrim orbitals)
    eigen = gms.EigenGms()
    eigen.read(ename,verbose=True)
    if ntrim > 0:
        C = eigen.alpha[:,:-ntrim]
    else:
        C = eigen.alpha
    MActive = C.shape[1] - nfc

    # 2. read CVs from file, and transform to MO basis
    print(f'reading in Cholesky vectors from {V2b_source}', flush=True)
    if is_complex:
        M, Ncv, CVlist, CVdagList = ch.load_choleskyList_3_IndFormat(infile=V2b_source,is_complex=True)
    else:
        M, Ncv, CVlist = ch.load_choleskyList_3_IndFormat(infile=V2b_source,verb=True,is_complex=False)
    # 3. perform CD on transformed CVs - restricted to ACTIVE SPACE
    Alist = ch.ao2mo_cholesky(C,CVlist)
    if is_complex:
        AdagList = ch.ao2mo_cholesky(C,CVdagList)

    # in some cases, we only want to transform CVs to the MO basis
    if transform_only: 
        print('Only transforming from GTO to orthonormal basis with no additional Cholesky decomposition', flush=True)
        NcvActive = Ncv
        CVsActive = Alist[:,nfc:,nfc:]
    else:
        print(f'Performing Cholesky decomposition within the active space num. frozen occupied={nfc}, num. truncated virtual={ntrim}', flush=True)
        V = FactoredIntegralGenerator(Alist[:,nfc:,nfc:])
        #NcvActive, CVsActive = ch.getCholeskyExternal_new(MActive, Alist[:,nfc:,nfc:], AdagList[:,nfc:,nfc:], tol=tol)
        #NcvActive, CVsActive = cholesky(mol=mol,integral_generator=V,tol=tol)
        NcvActive, CVsActive = cholesky(integral_generator=V,tol=tol)
        del(V)

    # 4. save CVs to new file
    print('saving Cholesky vectors',flush=True)
    ch.save_choleskyList_GAFQMCformat(NcvActive, MActive, CVsActive, outfile='V2b_MO_cholesky-active.mat')

    # 5. load K,S from one_body_gms
    #       - transform to MO basis (active space)
    #       - compute embedding potential, add to K_active
    print('Computing one-body embedding terms', flush=True)
    Mfull, K, S = ch.load_oneBody_gms(V1b_source)
    make_transformed_eigen(C[:,nfc:], S, outname='embedding.eigen_gms')
    S_MO = ch.ao2mo_mat(C,S)
    K_MO = ch.ao2mo_mat(C,K)

    S_active = S_MO[nfc:,nfc:]
    K_active = K_MO[nfc:,nfc:]

    print(f'shape of K_active is {K_active.shape}')
    if nfc > 0:
        if is_complex:
            K_active+=ch.get_embedding_potential_CV(nfc, C, Alist, AdagList,is_complex=True)
        else:
            K_active+=ch.get_embedding_potential_CV(nfc, C, Alist, AdagList=None,is_complex=False)

    # 6. save one body terms
    print('Saving one-body embedding terms', flush=True)
    ch.save_oneBody_gms(MActive, K_active, S_active, outfile='one_body_gms-active')
    
    # 7. compute constant Energy
    #       - E_K = trace K over core orbitals
    #       - E_V = embedding constant from V2b
    print('Computing constant energy', flush=True)
    if nfc > 0:
        E_K = 2*np.einsum('ii->',K_MO[:nfc,:nfc])
        
        if is_complex:
            E_V = ch.get_embedding_constant_CV(C,Alist[:,:nfc,:nfc],AdagList[:,:nfc,:nfc],is_complex=is_complex)
        else:
            E_V = ch.get_embedding_constant_CV(C,Alist[:,:nfc,:nfc],AdagList=None,is_complex=False)
    else:
        E_K=0.0
        E_V=0.0

    # 8. print constant energy
    E_const = Enuc + E_K + E_V
    print(f'E_0 = Enuc + E_K + E_V = {E_const} with:\n  - Enuc = {Enuc}\n  - E_K = {E_K}\n  - E_V = {E_V}')

    return E_const

def make_embedding_H_afqmclab(nfc=0,nactive=None,Enuc=0.0,tol=1.0e-6,C=None,twoBody=None,oneBody=None,S=None,transform_only=False):
    '''
    high level function to produce the embedding / downfolding Hamiltonian
    saves the results to files

    gets CVs from V2b_source, which chould contain the GTO basis CVs
    similarly for V1b_source

    Inputs:
    
    Outputs:
    > saves the following files

    #TODO: add option to use a list of active orbital indices - internally, we should always do this. The problem then reduces to generating this list of indices.

    '''

    
    # 1. read in orbitals (go ahead and remove the last ntrim orbitals)
    if C is None:
        print("Currently \"make_embedding_H_afqmclab\" requires C as input")
        return None

    if nactive is None:
        nactive = S.shape[0] - nfc
    
    C = C[:,:nfc+nactive]
    #MActive = C.shape[1] - nfc
    
    # 2. read CVs from file, and transform to MO basis
    Alist = ch.ao2mo_cholesky(C,twoBody)
    Ncv = twoBody.shape[0]

    # in some cases, we only want to transform CVs to the MO basis
    if transform_only:
        print('Only transforming from GTO to orthonormal basis with no additional Cholesky decomposition', flush=True)
        NcvActive = Ncv
        twoBodyActive = Alist[:,nfc:,nfc:]
    else:
        print(f'Performing Cholesky decomposition within the active space num. frozen occupied={nfc}, num. of active orbitals = {nactive}', flush=True)
        V = FactoredIntegralGenerator(Alist[:,nfc:,nfc:])
        NcvActive, twoBodyActive = cholesky(integral_generator=V,tol=tol)
        del(V)

    # 5. load K,S from one_body_gms
    #       - transform to MO basis (active space)
    #       - compute embedding potential, add to K_active
    print('Computing one-body embedding terms', flush=True)
    
    S_MO = ch.ao2mo_mat(C,S)
    oneBody_MO = ch.ao2mo_mat(C,oneBody)

    S_active = S_MO[nfc:,nfc:]
    oneBody_active = oneBody_MO[nfc:,nfc:]

    print(f'shape of oneBody_active is {oneBody_active.shape}')
    if nfc > 0:
        oneBody_active+=ch.get_embedding_potential_CV(nfc, C, Alist, AdagList=None,is_complex=False)
    
    # 7. compute constant Energy
    #       - E_K = trace K over core orbitals
    #       - E_V = embedding constant from V2b
    print('Computing constant energy', flush=True)
    if nfc > 0:
        E_K = 2*np.einsum('ii->',oneBody_MO[:nfc,:nfc])
        E_V = ch.get_embedding_constant_CV(C,Alist[:,:nfc,:nfc],AdagList=None,is_complex=False)
    else:
        E_K=0.0
        E_V=0.0
    E_const = Enuc + E_K + E_V
    print(f'E_0 = Enuc + E_K + E_V = {E_const} with:\n  - Enuc = {Enuc}\n  - E_K = {E_K}\n  - E_V = {E_V}')

    return twoBodyActive,NcvActive,oneBody_active,S_active,E_const


def make_embedding_H_afqmclab_solids(nfc,nactive,Enuc,C=None,tol=1.0e-6,twoBody=None,oneBody=None,S=None,transform_only=True,debug=False,is_complex=False):
    '''
    high level function to produce the embedding / downfolding Hamiltonian
    saves the results to files

    Inputs: ALL in MO basis!
    
    Outputs:
    > saves the following files
    '''

    print("WARNING : make_embedding_H_afqmclab_solids is in Alpha, check output carefully!")

    # 1. read in orbitals (go ahead and remove the last ntrim orbitals)
    if C is None:
        print("Currently \"make_embedding_H_afqmclab\" requires C as input")
        return None
    C = C[:,:nfc+nactive]
    #MActive = C.shape[1] - nfc
    
    # 2. read CVs from file, and transform to MO basis
    # HACK reading in twoBody in mo basis!
    Alist = twoBody[:,:nfc+nactive, :nfc+nactive] #ch.ao2mo_cholesky(C,twoBody)
    AdagList = np.zeros(Alist.shape,dtype='complex128')
    print("AdagList.shape = ", AdagList.shape)
    Ncv = twoBody.shape[0]

    for g in range(Ncv):
        L = Alist[g,:,:]
        AdagList[g,:,:] = L.conj().T

    # in some cases, we only want to transform CVs to the MO basis
    if transform_only:
        print('Only transforming from GTO to orthonormal basis with no additional Cholesky decomposition', flush=True)
        NcvActive = Ncv
        twoBodyActive = Alist[:,nfc:,nfc:]
    else:
        print(f'Performing Cholesky decomposition within the active space num. frozen occupied={nfc}, num. of active orbitals = {nactive}', flush=True)
        V = FactoredIntegralGenerator(Alist[:,nfc:,nfc:])
        NcvActive, twoBodyActive = cholesky(integral_generator=V,tol=tol)
        del(V)

    # 5. load K,S from one_body_gms
    #       - transform to MO basis (active space)
    #       - compute embedding potential, add to K_active
    print('Computing one-body embedding terms', flush=True)
    
    S_MO = ch.ao2mo_mat(C,S)
    oneBody_MO = ch.ao2mo_mat(C,oneBody)

    S_active = S_MO[nfc:,nfc:]
    oneBody_active = oneBody_MO[nfc:,nfc:]

    print(f'shape of oneBody_active is {oneBody_active.shape}')
    if nfc > 0:
        oneBody_active+=ch.get_embedding_potential_CV(nfc, C, Alist, AdagList=AdagList,is_complex=True)
    
    # 7. compute constant Energy
    #       - E_K = trace K over core orbitals
    #       - E_V = embedding constant from V2b
    print('Computing constant energy', flush=True)
    if nfc > 0:
        E_K = 2*np.einsum('ii->',oneBody_MO[:nfc,:nfc])
        E_V = ch.get_embedding_constant_CV(C,Alist[:,:nfc,:nfc],AdagList=AdagList[:,:nfc,:nfc],is_complex=True)
    else:
        E_K=0.0
        E_V=0.0
    E_const = Enuc + E_K + E_V
    print(f'E_0 = Enuc + E_K + E_V = {E_const} with:\n  - Enuc = {Enuc}\n  - E_K = {E_K}\n  - E_V = {E_V}')

    return twoBodyActive,NcvActive,oneBody_active,S_active,E_const

def make_embedding_H_afqmclab_GHF(nfc,nactive,Enuc,tol=1.0e-6,C=None,twoBody=None,oneBody=None,S=None,transform_only=False,debug=False,is_complex=True):
    '''
    high level function to produce the embedding / downfolding Hamiltonian
    saves the results to files

    gets CVs from V2b_source, which chould contain the GTO basis CVs
    similarly for V1b_source

    Inputs:
    
    Outputs:
    > saves the following files
    '''

    
    # 1. read in orbitals (go ahead and remove the last ntrim orbitals)
    if C is None:
        print("Currently \"make_embedding_H_afqmclab\" requires C as input")
        return None
    C = C[:,:nfc+nactive]
    #MActive = C.shape[1] - nfc
    
    # 2. read CVs from file, and transform to MO basis
    Alist = ch.ao2mo_cholesky(C,twoBody)
    
    # Need the complex conjugate as well!! orbitals are complex-valued!
    AdagList = np.zeros(Alist.shape,dtype='complex128')
    for g in range(AdagList.shape[0]):
        AdagList[g] = Alist[g].conj().T

    Ncv = twoBody.shape[0]

    # in some cases, we only want to transform CVs to the MO basis
    if transform_only:
        print('Only transforming from GTO to orthonormal basis with no additional Cholesky decomposition', flush=True)
        NcvActive = Ncv
        twoBodyActive = Alist[:,nfc:,nfc:]
        twoBodyDagActive = AdagList[:,nfc:,nfc:]
    else:
        print(f'Performing Cholesky decomposition within the active space num. frozen occupied={nfc}, num. of active orbitals = {nactive}', flush=True)
        V = FactoredIntegralGenerator(Alist[:,nfc:,nfc:]) # needs update for the case of complex orbitals
        NcvActive, twoBodyActive = cholesky(integral_generator=V,tol=tol)
        del(V)

    # 5. load K,S from one_body_gms
    #       - transform to MO basis (active space)
    #       - compute embedding potential, add to K_active
    print('Computing one-body embedding terms', flush=True)
    
    S_MO = ch.ao2mo_mat(C,S)
    oneBody_MO = ch.ao2mo_mat(C,oneBody)

    S_active = S_MO[nfc:,nfc:]
    oneBody_active = oneBody_MO[nfc:,nfc:]

    print(f'shape of oneBody_active is {oneBody_active.shape}')
    if nfc > 0:
        oneBody_active+=ch.get_embedding_potential_CV_GHF(nfc, C, Alist, AdagList=AdagList,is_complex=False)
    
    # 7. compute constant Energy
    #       - E_K = trace K over core orbitals
    #       - E_V = embedding constant from V2b
    print('Computing constant energy', flush=True)
    if nfc > 0:
        E_K = np.einsum('ii->',oneBody_MO[:nfc,:nfc])
        E_V = 0.5*ch.get_embedding_constant_CV_GHF(C,Alist[:,:nfc,:nfc],AdagList=AdagList[:,:nfc,:nfc],is_complex=False)
        #E_V = ch.get_embedding_constant_CV_GHF(C,Alist[:,:nfc,:nfc],AdagList=AdagList[:,:nfc,:nfc],is_complex=False)
    else:
        E_K=0.0
        E_V=0.0
    E_const = Enuc + E_K + E_V
    print(f'E_0 = Enuc + E_K + E_V = {E_const} with:\n  - Enuc = {Enuc}\n  - E_K = {E_K}\n  - E_V = {E_V}')

                                                                                
    return twoBodyActive,NcvActive,oneBody_active,S_active,E_const


def make_embedding_H_afqmclab_GHF_v2(nfc,nactive,Enuc,tol=1.0e-6,C=None,twoBody=None,oneBody=None,S=None,transform_only=False,debug=False,is_complex=False):
    '''
    high level function to produce the embedding / downfolding Hamiltonian
    saves the results to files

    gets CVs from V2b_source, which chould contain the GTO basis CVs
    similarly for V1b_source

    Inputs:

    nfc = number of frozen SPATIAL orbitals
    nactive = number of active SPATIAL orbitals
    Enuc = constant energy (usually just the nuclear repulsion energy)
    C = Molecular spin-orbital basis
    twoBody = GTO basis Cholesky vectors in GHF form
    oneBody = GTO basis one-body Hamiltonian in GHF form (may contain complex terms such as the SOC operator)
    S = GTO overlap matrix in GHF form

    options:
    transform_only = False : if True, runs a secondary Cholesky decomposition on input CholeskyVecs within the active space
    debug = False " display debugging information
    is_complex = True : use complex arrays to store data? (this is necessary i.e. for SOC)

    Outputs:
 
    '''

    def cut_mat(mat, n, above=True, is_complex=True, m=0):

        '''

        inputs:
        
        mat = GHF-type matrix to cut
        n = index to begin cut
        above = True (bool) is True, return the cut above index n (inclusive)
        is_complex =True (bool) use complex matrix for cut
        m = 0 (optional) upper bound on cut (only effects result if above == True)

        '''

        
        M = mat.shape[0] // 2 - m

        if above:
            size = M - n
        else:
            size = n

        mat_type = mat.dtype
        cut = np.zeros((2*size,2*size),dtype=mat_type)
    
        # need to set 4 sectors!
        if above:
            cut[:size,:size] = mat[n:n+size,n:n+size] # up up 
            cut[:size,size:] = mat[n:n+size,M+n:M+n+size] # up down
            cut[size:,:size] = mat[M+n:M+n+size,n:n+size] # up down
            cut[size:,size:] = mat[M+n:M+n+size,M+n:M+n+size] # up down
        else:
            cut[:size,:size] = mat[:n,:n] # up up 
            cut[:size,size:] = mat[:n,M:M+n] # up down
            cut[size:,:size] = mat[M:M+n,:n] # up down
            cut[size:,size:] = mat[M:M+n,M:M+n] # up down

        return cut

    if is_complex:
        print("make_embedding_H_afqmclab_GHF_v2 does not currently support complex type, using real instead")
        is_complex = False

    # 1. read in orbitals (go ahead and remove the last ntrim orbitals)
    if C is None:
        print("Currently \"make_embedding_H_afqmclab\" requires C as input")
        return None
        
    L = C.shape[0] # total spin-orbitals
    M = L // 2 # total spatial orbitals

    Lcut = L - 2*nfc # number active spin-orbitals
    Mcut = M - nfc # active spatial orbitals

    imsize = nfc+nactive # intermediate size
    if is_complex:
        Cvt = np.zeros((L,2*imsize),dtype='complex128')
    else:
        Cvt = np.zeros((L,2*imsize))
    #C = C[:,:nfc+nactive]
    Cvt[:M,:imsize] = C[:M,:imsize]
    Cvt[M:,imsize:] = C[M:,M:M+imsize]

   # Cvt = cut_mat(C, nfc, above=True, is_complex=False, m=0)

    # 2. read CVs from file, and transform to MO basis
    Alist = ch.ao2mo_cholesky(Cvt,twoBody)
    
    # Need the complex conjugate as well!! orbitals are complex-valued!
    #AdagList = np.zeros(Alist.shape,dtype='complex128')
    #for g in range(AdagList.shape[0]):
    #    AdagList[g] = Alist[g].conj().T

    Ncv = twoBody.shape[0]

    if transform_only == False:
        print("WARNING: secondary Cholesky not tested for GHF formalism. Check that varitaional energy comes out as expected!")

    # in some cases, we only want to transform CVs to the MO basis
    if transform_only:
        print('Only transforming from GTO to orthonormal basis with no additional Cholesky decomposition', flush=True)
        NcvActive = Ncv
        #twoBodyActive = Alist[:,nfc:,nfc:] # wrong! only cuts the up sector core out
        #twoBodyDagActive = AdagList[:,nfc:,nfc:]
        twoBodyActive = np.zeros((NcvActive,Lcut,Lcut))
        for g in range(NcvActive):
            twoBodyActive[g,:,:] = cut_mat(Alist[g,:,:], nfc, above=True)
    else:
        print(f'Performing Cholesky decomposition within the active space num. frozen occupied={nfc}, num. of active orbitals = {nactive}', flush=True)
        V = FactoredIntegralGenerator(Alist[:,nfc:,nfc:]) # needs update for the case of complex orbitals
        NcvActive, twoBodyActive = cholesky(integral_generator=V,tol=tol)
        del(V)

    # 5. load K,S from one_body_gms
    #       - transform to MO basis (active space)
    #       - compute embedding potential, add to K_active
    print('Computing one-body embedding terms', flush=True)
    
    S_MO = ch.ao2mo_mat(Cvt,S)
    oneBody_MO = ch.ao2mo_mat(Cvt,oneBody)

    #S_active = S_MO[nfc:,nfc:]
    #oneBody_active = oneBody_MO[nfc:,nfc:]
    
    S_active = cut_mat(S_MO, nfc, above=True, is_complex=False)
    oneBody_active = cut_mat(oneBody_MO, nfc, above=True, is_complex=True)

    print(f'shape of oneBody_active is {oneBody_active.shape}')
    if nfc > 0:
        oneBody_active+=ch.get_embedding_potential_CV_GHF_v2(nfc, Alist, is_complex=False)
    
    # 7. compute constant Energy
    #       - E_K = trace K over core orbitals
    #       - E_V = embedding constant from V2b
    print('Computing constant energy', flush=True)
    if nfc > 0:
        E_K = 2.0*np.einsum('ii->',oneBody_MO[:nfc,:nfc]) # TODO need to look at other spin-sectors!!
        
        # TODO this is a hack assuming ROHF orbitals!, we should call this twice, once for each sector for UHF!
        E_V = 1.0*ch.get_embedding_constant_CV_GHF_v2(Alist[:,:nfc,:nfc])
    else:
        E_K=0.0
        E_V=0.0
    E_const = Enuc + E_K + E_V
    print(f'E_0 = Enuc + E_K + E_V = {E_const} with:\n  - Enuc = {Enuc}\n  - E_K = {E_K}\n  - E_V = {E_V}')

                                                                                
    return twoBodyActive,NcvActive,oneBody_active,S_active,E_const
