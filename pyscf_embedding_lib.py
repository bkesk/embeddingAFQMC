import numpy as np
import h5py as h5
from pyscf import lib, gto, scf, ao2mo # importing specific pyscf submodules

# the following are libraries written by us
import pyqmc.matrices.gms as gms # written by Wirawan Purwanto to all read/write to the GAMESS file format for 1-body, and single particle orbitals.
# submodules - use relative path
import embedding.cholesky as ch # written by Kyle Eskridge to implement a Cholesky decomp. in pyscf, along with other support functions.
import embedding.density as dmat

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
    #dm = scf.hf.from_chk(mol, chkfile)
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
        print("printing C matrix")
        print_mat(C)

    dm = mf.make_rdm1(mo_coeff=C, mo_occ=occ)
    if verb:
        print("DM: ", dm)
    return dm

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

    T_a = np.einsum('gjk,ki->gji',Adag,dm[0]) # test! using dm[0] instead of dm
    T_b = np.einsum('gjk,ki->gji',Adag,dm[1])

    J_a = np.einsum('gil,gjj->il',A,T_a)
    K_a = np.einsum('gil,gji->jl',A,T_a)
    
    J_b = np.einsum('gil,gjj->il',A,T_b)
    K_b = np.einsum('gil,gji->jl',A,T_b)

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
            print("==== computing two-body interactions on the fly ====")
            
        def _custom_jk(mol,dm,*args):
            custom_jk_uhf(mol, dm, V2b_file=V2b_file, *args)
            
        mf.get_jk = _custom_jk #(mf.mol,dm,V2b_file=V2b_file)
    else:
        if verb >= 4:
            print("==== computing and storing two-body tensor in memory ====")
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
            print("  [+] Trace of the density matrix beta sector = {}".format(np.trace(dm[1])))

            print("DM provided -> Electronic Energy: {}".format(mf.energy_elec(dm=dm)))
        E=mf.kernel(dm)
    elif dm_file is not None and N_frozen_occ is not None:
        if verb >= 4:
            print("[+] customH_mf : using DM from {} with N_frozen_occ = {}".format(dm_file,N_frozen_occ))
        dm_full = get_dm_from_h5(dm_file, verb=(verb>=4))
        # need the number of frozen occupied orbitals to get the correct slice from the
        #   full density matrix
        dm = [dm_full[0][N_frozen_occ:M+N_frozen_occ,N_frozen_occ:M+N_frozen_occ],
              dm_full[1][N_frozen_occ:M+N_frozen_occ,N_frozen_occ:M+N_frozen_occ]]
        print("[+] customH_mf : checking integrity of truncated density matrix")
        print("  [+] Trace of the truncated density matrix alpha sector = {}".format(np.trace(dm[0])))
        print("  [+] Trace of the truncated density matrix beta sector = {}".format(np.trace(dm[1])))
        E=mf.kernel(dm)
    else:
        if verb >= 4:
            print("[+] customH_mf : no DM provided, using Pyscf default Guess")
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
    
    C^bar should be a trivial diagonal matrix in any case I can think of at the present.
    It can be computed as: C^bar = C^dag S C, where S is the GTO basis overlap matrix
    '''
    Cbar = np.zeros(C.shape)
    Cbar = C.conj().T*S*C

    if outname is not None:
        write_orbs(Cbar, M=Cbar.shape[0], output=outname, restricted=restricted)

    return Cbar

def make_embedding_H(nfc,ntrim,Enuc,tol=1.0e-6,ename='eigen_gms',V2b_source='V2b_AO_cholesky.mat',V1b_source='one_body_gms',transform_only=False,debug=False):
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
    eigen = gms.EigenGms()
    eigen.read(ename,verbose=True)
    if ntrim > 0:
        C = eigen.alpha[:,:-ntrim]
    else:
        C = eigen.alpha
    MActive = C.shape[1] - nfc

    # 2. read CVs from file, and transform to MO basis
    M, Ncv, CVlist, CVdagList = ch.load_choleskyList_3_IndFormat(infile=V2b_source)

    # 3. perform CD on transformed CVs - restricted to ACTIVE SPACE
    Alist = ch.ao2mo_cholesky(C,CVlist)
    AdagList = ch.ao2mo_cholesky(C,CVdagList)

    # in some cases, we only want to transform CVs to the MO basis
    if transform_only: 
        print('Only transforming from GTO to orthonormal basis with no additional Cholesky decomposition')
        NcvActive = Ncv
        CVsActive = Alist[:,nfc:,nfc:]
    else:
        NcvActive, CVsActive = ch.getCholeskyExternal_new(MActive, Alist[:,nfc:,nfc:], AdagList[:,nfc:,nfc:], tol=tol)
    
    # 4. save CVs to new file
    ch.save_choleskyList_GAFQMCformat(NcvActive, MActive, CVsActive, outfile='V2b_MO_cholesky-active.mat')

    # 5. load K,S from one_body_gms
    #       - transform to MO basis (active space)
    #       - compute embedding potential, add to K_active
    Mfull, K, S = ch.load_oneBody_gms(V1b_source)
    make_transformed_eigen(C[:,nfc:], S, outname='embedding.eigen_gms')
    S_MO = ch.ao2mo_mat(C,S)
    K_MO = ch.ao2mo_mat(C,K)

    S_active = S_MO[nfc:,nfc:]
    K_active = K_MO[nfc:,nfc:]

    print(f'shape of K_active is {K_active.shape}')
    if nfc > 0:
        K_active+=ch.get_embedding_potential_CV(nfc, C, Alist, AdagList)
    
    # 6. save one body terms
    ch.save_oneBody_gms(MActive, K_active, S_active, outfile='one_body_gms-active')
    
    # 7. compute constant Energy
    #       - E_K = trace K over core orbitals
    #       - E_V = embedding constant from V2b
    if nfc > 0:
        E_K = 2*np.einsum('ii->',K_MO[:nfc,:nfc])
        E_V = ch.get_embedding_constant_CV(C,Alist[:,:nfc,:nfc],AdagList[:,:nfc,:nfc])
    else:
        E_K=0.0
        E_V=0.0

    # 8. print constant energy
    E_const = Enuc + E_K + E_V
    print(f'E_0 = Enuc + E_K + E_V = {E_const} with:\n  - Enuc = {Enuc}\n  - E_K = {E_K}\n  - E_V = {E_V}')

    return E_const
