import numpy as np
import h5py as h5
import sys
import logging

import pyscf
from pyscf  import ao2mo, scf, gto
from pyscf.gto import getints_by_shell

from V2b_inspect import load_V2b_dump, save_V2b_dump, sym_2d_unpack, sym_2d_pack
import pyqmc.matrices.gms as gms



def save_choleskyList_GAFQMCformat(Ncv, M, CVlist, outfile="V2b_AO_cholesky.mat"):
    '''
    This function converts from a list of square-matrix Cholesky vectors, to a numpy
    array containing the lower diagonal (LD) form of each CV with shape = (Ncv, M*(M+1)/2). This is
    the format that the GAFQMC code will expect.
    '''
    #M = int(np.sqrt(2*CV_LD.shape[1]))
    CVarray = np.empty((Ncv, M*(M+1)//2))
    print (CVarray.shape)
    for i in range(Ncv):
        # convert from a 1-D vector, to a MxM matrix rep.
        # insert factor of 1/sqrt(2)        
        Lmat = CVlist[i].reshape(M,M)*(1/np.sqrt(2))
        print (i, Lmat.shape)
        sym_2d_pack(Lmat,CVarray[i])
    print ("Lmat shape = ", Lmat.shape)
    save_V2b_dump(CVarray.T, outfile, dbg=1)

def load_choleskyList_GAFQMCformat(infile="V2b_AO_cholesky.mat", verb=False):
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
    CVarray = np.empty((Ncv, M*M))
    for i in range(Ncv):
        if verb:
            print ("vector ", i)
        # convert from a 1-D vector, to a MxM matrix rep.
        # insert factor of sqrt(2) (pyscf and GAFQMC use different conentions concerning including/exclding the factor of 1/2 in the matrix elements        
        Lmat = sym_2d_unpack(CV_LD[i])*np.sqrt(2)
        if verb:
            print (Lmat.shape)
        CVarray[i] = Lmat.flatten()
    return M, Ncv, CVarray

def load_choleskyList_3_IndFormat(infile="V2b_AO_cholesky.mat",verb=False):
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
    CVarrayDag = np.empty((Ncv, M, M))
    for i in range(Ncv):
        if verb:
            print ("vector ", i)
        # convert from a 1-D vector, to a MxM matrix rep.
        # insert factor of sqrt(2) (pyscf and GAFQMC use different conventions concerning including/exclding the factor of 1/2 in the matrix elements        
        Lmat = sym_2d_unpack(CV_LD[i])*np.sqrt(2) #CVlist[i].reshape(M,M)*(1/np.sqrt(2)))
        if verb: 
            print (Lmat.shape)
        CVarray[i] = Lmat
        CVarrayDag[i] = Lmat.conj().T
    return M, Ncv, CVarray, CVarrayDag

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

def get_ovlp(mol, verb=False):
    try:
        S = mol.intor_symmetric('int1e_ovlp')
    except:
        S = mol.intor_symmetric('cint1e_ovlp_sph') 
    if verb:
        print (S)
    return S

def get_one_body_H(mol, verb=False):
    try:
        Kin = mol.intor_symmetric('int1e_kin')
        Vnuc = mol.intor_symmetric('int1e_nuc')
    except:
        Kin = mol.intor_symmetric('cint1e_kin_sph')
        Vnuc = mol.intor_symmetric('cint1e_nuc_sph')
    if verb:
        print("kinetic ", Kin)
        print("nuc. interaction", Vnuc)
    K = np.add(Kin,Vnuc)
    if verb:
        print(K)
    return K

def save_oneBody_gms(M, K, S, outfile="one_body_gms"):
    O = gms.OneBodyGms()
    O.nbasis = M
    O.S = S
    O.H1 = K
    O.write(outfile)

def map_shellIndex(mol):
    '''
    this function constructs and returns a mapping between the global basis set index, mu, and
    the tuple containing the Shell index, and internal index (i.e. index within the shell), (I,i).
    Uppercase indicies are shell indicies, lowercase are internal, and greek are global
    '''
    nShells = mol.nbas
    index_map = []
    for I in range(nShells):
        shell_info = mol._bas[I]
        l = shell_info[1]
        NsubShell = shell_info[3]
        internalSize = (2*l + 1)*NsubShell #not true for Cartessian, but we want spherical anyways
        for i in range(internalSize):
            index_map.append((I,i))

    return index_map

def V2b_diagonal(mol, intor_name='int2e_sph', verb=None):
    '''
    using upper case indecies for shell index (I,J,K,L, etc.), and lower case for
    orbital index (i,j,k,l, etc.). 

    Here, the 2-body V diagonal is constructed without a complete reconstruction of V.
    However, the pyscf integral library (actually lincint) will only allow us to access integrals 
    shell/shell (not orbital by orbital).

    ***** Imprtant Note: Pyscf seems to use the "chemist's" notation for index ordering
    i.e. the returned integrals are (il|jk) as opposed to the 'physicists' notation <ij|kl>.
    
    '''

    if verb is None:
        verb = mol.verbose

    nbasis  = mol.nao_nr()
    nShells = mol.nbas
    if verb > 5:
        print("# of shells: ", nShells)
   
    Vdiag =np.zeros(nbasis*nbasis)

    index_Map = map_shellIndex(mol)
    if verb > 5:
        print("index map")
        print("length of map:", len(index_Map))
        for entry in index_Map:
            print(entry)
    
    pairIndS = 0
    for I in range(nShells):
        for L in range(nShells):
            J = I
            K = L
            shell_ints = getints_by_shell(intor_name, (I,L,J,K), mol._atm, mol._bas, mol._env)
            if verb > 5:
                print(shell_ints.shape)
            for i in range(shell_ints.shape[0]):
                for l in range(shell_ints.shape[1]):
                    j = i
                    k = l
                    if verb > 5:
                        print("(I,L,J,K) = (%i,%i,%i,%i), (i,l,j,k) = (%i,%i,%i,%i) " % (I,L,J,K,i,l,j,k))
                    i_global = index_Map.index((I,i))
                    l_global = index_Map.index((L,l))
                    Vdiag[i_global*nbasis + l_global] = shell_ints[i,l,j,k]
            if verb > 5:
                print("shell pair index = ", pairIndS)
            pairIndS += 1
        
    return Vdiag

def V2b_row(mol, mu, CVlist=None, intor_name='int2e_sph', verb=None):
    '''
    using upper case indecies for shell index (I,J,K,L, etc.), and lower case for
    orbital index (i,j,k,l, etc.). 

    Here, the 2-body V row is constructed without a complete reconstruction of V.
    However, the pyscf integral library (actually lincint) will only allow us to access integrals 
    shell/shell (not orbital by orbital).

    returns row V_mu_nu of the 2-body interaction tensor.

    ***** Imprtant Note: Pyscf seems to use the 'chemists" notation for index ordering
    i.e. the returned integrals are (il|jk) as opposed to the 'physicists' notation <ij|kl>.
    
    '''    
    if verb is None:
        verb = mol.verbose

    nbasis  = mol.nao_nr()
    nShells = mol.nbas
    if verb > 5:
        print("# of shells: ", nShells)
   
    Vrow =np.zeros(nbasis*nbasis)

    index_Map = map_shellIndex(mol)
    if verb > 5:
        print("index map")
        print("length of map:", len(index_Map))
        for entry in index_Map:
            print(entry)
    
    pairIndS = 0
    
    i_global = mu // nbasis
    l_global = mu % nbasis

    I, i = index_Map[i_global]
    L, l = index_Map[l_global]

    for J in range(nShells):
        for K in range(nShells):
            shell_ints = getints_by_shell(intor_name, (I,L,J,K), mol._atm, mol._bas, mol._env)
            if verb > 5:
                print(shell_ints.shape)
            for j in range(shell_ints.shape[2]):
                for k in range(shell_ints.shape[3]):
                    if verb > 5:
                        print("(I,L,J,K) = (%i,%i,%i,%i), (i,l,j,k) = (%i,%i,%i,%i) " % (I,L,J,K,i,l,j,k))
                    j_global = index_Map.index((J,j))
                    k_global = index_Map.index((K,k))
                    Vrow[j_global*nbasis + k_global] = shell_ints[i,l,j,k]
            if verb > 5:
                print("shell pair index = ", pairIndS)
            pairIndS += 1
        
    # We need to be able to account for the CVs already computed:
    # we are only substracting/ accessing the element in each CV at the relevant mu = (i,l) index.
    if CVlist is not None:
        
        def build_row(CVlist, mu, M):
            '''
            Builds just the mu^th row of the V_{mu nu} tensor, using the Cholesky Vectors.
            '''
            #verb = 7 # TEMP HACK
            if verb > 6:
                print("DEBUG: from 'build_row()': mu =" , mu)
            Vrow = np.zeros((M*M))
            i = mu // M
            l = mu % M
            for g in range(len(CVlist)):
                A = CVlist[g]
                if verb > 6:
                    print("DEBUG: from 'build_row()': A[", g,"] =" , A)
                Amu = A[i*M + l]
                if verb > 6:
                    print ("DEBUG: from 'build_row()': A[", g,"]_mu =" , Amu)
                Adag = A.reshape((M,M)).conj().T
                if verb > 6:
                    print("DEBUG: from 'build_row()': A^dag[", g,"] =" , Adag)
                Vrow += Amu*Adag.reshape((M*M))
            if verb > 6:
                print("DEBUG: from 'build_row()': Vrow =" , Vrow)
            return Vrow#.reshape((M*M))

        Ncv = len(CVlist)
        if verb > 4:
            print("\n   *** V2b_row: debug info ***\n")
            print("Vrow direct from integrals", Vrow)
            Vrow_temp = Vrow.copy()
        Vrow = Vrow - build_row(CVlist, i_global*nbasis + l_global, nbasis)
        if verb > 4:
            print("   Vrow - A*A^dag (i.e. residual matrix row):", Vrow)
            print("   A*A^dag (direct function call to build_row()):", build_row(CVlist, i_global*nbasis + l_global, nbasis))
            print("   delta Vrow: ", Vrow_temp - Vrow)
                             # to rebuild the entire residual matrix. Vfac will compute the row on
                             # the fly, it autimatically applies a factor of -1.
            print("\n   *** *** *** ***\n")
            
    return Vrow

class factoredERIs:
    '''
    author: Kyle Eskridge

    This class contains the two-body Hamiltonian interaction term, Vijkl; 
    however, "under the hood", it stores it in the factored form:
    V_{(i,l)(j,k)} = sum_g A^g_{il} * (A^g_{jk})^dagger
    where A are one-body operators (these could be Cholesky vecotrs, eigenvectors of
    the V supermatrix, etc.)

    This class is not necessarily highly optimized; it is meant to be readable/maintainable.
    It also tends favors lower memory usage over fast execution time, since memory could very
    esily become a limiting factor.

    Notes: the mapping between mu and (i,l) is as follows:
    mu = i*M + l  - inverted as - i = mu / M, l = mu % M
    nu = j*M + k  - inverted as - j = nu / M, k = nu % M

    input arguments:
    Alist - numpy array with 3-dimensions, the first is the vector index, and the other two are basis
            indicies i and l.

    Functions for internal use:
    __getitem__(self, index): returns values of Vijkl:
    
    ex: V = factoredERIs(Alist,M)
    print (V[0,0,0,0]) # this should output V_{0 0 0 0}

    _constitute(self, i,j,k,l): constitutes the V_{ijkl} matrix elements on-the-fly

    functions:
    
    diagonal(): return all diagonal elements of Vijkl supermatrix
    row(mu): return the mu^th row of Vijkl supermatrix

    eri_full: return numpy array containing full Vijkl tensor

    '''
    def __init__(self, Alist, M, verb=False):
        self.Alist = Alist
        self.Nvec = Alist.shape[0]
        self.M = M # TODO: there is probably a way to calculate this
        if verb:
            self.verb=True
            print("Shape of Alist = ", self.Alist.shape)
            print("Nvec = ", self.Nvec)
            print("M = ", self.M)
        else:
            self.verb=False

    def __getitem__(self, key):
        if len(key) == 1:
            return self.row(key)
        else:
            return self._constitute(key[0], key[1], key[2], key[3])

    def _constitute(self,i,j,k,l):
        val = 0
        for g in range(self.Nvec):
            A = self.Alist[g]
            Adag = self.Alist[g].conj().T
            val += (A[i,l])*Adag[j,k]
        return val

    def diagonal(self):
        M = self.M
        diag = np.empty(M*M)
        for mu in range(M*M):
            i = mu // M
            j = i
            l = mu % M
            k = l
            diag[mu] = self._constitute(i,j,k,l)
        return diag

    def row(self, nu):
        M = self.M
        row = np.empty(M*M)
        for mu in range(M*M):
            i = mu // M
            j = nu // M
            l = mu % M
            k = nu % M
            row[mu] = self._constitute(i,j,k,l)
        return row

    def full(self):
        '''
        This is really quite slow. Not recomended.
        '''
        if self.verb:
            print("reconstituting full 2-body potential")
        Msq = self.M*self.M
        V = np.zeros((Msq,Msq))
        for g in range(self.Nvec):
            if self.verb:
                print("vector numbr = ",g)
            A = self.Alist[g].flatten()
            V += np.dot(A[:, None], A[None,:]) # this is equivalent to an outter product of A vecs
        return V

class factoredERIs_updateable:
    '''
    author: Kyle Eskridge

    ***** updated to allow the following form of V (useful for CD): *****

    V_{(i,l)(j,k)} = sum_g^{N_A} A^g_{il} * (A^g_{jk})^dagger - sum_g^{N_B} B^g_{il} * (B^g_{jk})^dagger

    where N_A >= N_B

    more compactly:

    V_{mu nu} = sum A^2 - sum B^2

    where the number of terms in each sum does not need to be the same

    *********************************************************************

    This class contains the two-body Hamiltonian interaction term, Vijkl; 
    however, "under the hood", it stores it in the factored form:
    V_{(i,l)(j,k)} = sum_g A^g_{il} * (A^g_{jk})^dagger
    where A are one-body operators (these could be Cholesky vecotrs, eigenvectors of
    the V supermatrix, etc.)

    This class is not necessarily highly optimized; it is meant to be readable/maintainable.
    It also favors lower memory usage over fast execution time, since memory could very
    esily become a limiting factor.

    Notes: the mapping between mu and (i,l) is as follows:
    mu = i*M + l  - inverted as - i = mu / M, l = mu % M
    nu = j*M + k  - inverted as - j = nu / M, k = nu % M

    input arguments:
    Alist - numpy array with 3-dimensions, the first is the vector index, and the other two are basis
            indicies i and l.

    Functions for internal use:
    __getitem__(self, index): returns values of Vijkl:
    
    ex: V = factoredERIs(Alist,M)
    print (V[0,0,0,0]) # this should output V_{0 0 0 0}

    _constitute(self, i,j,k,l): constitutes the V_{ijkl} matrix elements on-the-fly

    functions:
    
    diagonal(): return all diagonal elements of Vijkl supermatrix
    row(mu): return the mu^th row of Vijkl supermatrix

    eri_full: return numpy array containing full Vijkl tensor

    ToDo: we would like to allow the 'Blist' to grow as needed, 
          currently, it is the same shape as 'Alist'.

    '''
    def __init__(self, Alist, M, verb=False, useB=False):
        self.Alist = Alist
        if useB:
            self.useB = True
            self.Blist = np.zeros(Alist.shape)
            self.Bindex = 0
        else:
            self.useB = False
        self.Nvec = Alist.shape[0]
        self.M = M # TODO: there is probably a way to calculate this
        if verb:
            self.verb=True
            print("Shape of Alist = ", self.Alist.shape)
            print("Nvec = ", self.Nvec)
            print("M = ", self.M)
        else:
            self.verb=False

    def __getitem__(self, key):
        if len(key) == 1:
            return self.row(key)
        else:
            return self._constitute(key[0], key[1], key[2], key[3])

    def _constitute(self,i,j,k,l):
        val = 0
        for g in range(self.Nvec):
            A = self.Alist[g]
            Adag = self.Alist[g].conj().T
            val += (A[i,l])*Adag[j,k]
        if self.useB:
            for g in range(self.Bindex):
                B = self.Blist[g]
                Bdag = self.Blist[g].conj().T
                val -= (B[i,l])*Bdag[j,k]
            
        return val

    def updateBlist(self, B):
        if self.useB:
            self.Blist[self.Bindex] = B
            self.Bindex = self.Bindex + 1
            
    def diagonal(self):
        M = self.M
        diag = np.empty(M*M)
        for mu in range(M*M):
            i = mu // M
            j = i
            l = mu % M
            k = l
            diag[mu] = self._constitute(i,j,k,l)
        return diag

    def row(self, nu): # is this transposed? i.e. is this column and not row?
        M = self.M
        row = np.empty(M*M)
        for mu in range(M*M):
            i = mu // M
            j = nu // M
            l = mu % M
            k = nu % M
            row[mu] = self._constitute(i,j,k,l)
        return row

    #def full(self):
    #    '''
    #    This is really quite slow. Not recomended.
    #    '''
    #    if self.verb:
    #        print("reconstituting full 2-body potential")
    #    Msq = self.M*self.M
    #    V = np.zeros((Msq,Msq))
    #    for g in range(self.Nvec):
    #        if self.verb:
    #            print ("vector numbr = ",g)
    #        A = self.Alist[g].flatten()
    #        V += np.dot(A[:, None], A[None,:]) # this is equivalent to an outter product of A vecs
    #    return V
    def full(self):
        '''
        This is a wrapper for backwards compatibility
        '''
        #return self.full_einsum_2()
        return self.full_einsum()

    def full_einsum(self):
        '''
        reconstitute Vijkl using numpy's einstein summation command
        '''
        A = self.Alist
        Adag = A.conj().T
        #V = np.einsum('gil,gjk->ijkl', A, Adag)
        V = np.dot(Adag, A)
        if self.verb:
            print("reconstituted V : ", V)
            print("reconstituted V shape : ", V.shape)
        return V
        
    def full_einsum_2(self):
        '''
        TODO:
         - remake using the 3-index format A : (gamma,i,l)
        
        reconstitute Vijkl using numpy's einstein summation command
        '''
        debug = False

        if self.verb:
            print("[+] Reconstituting V_ijkl tensor. This may take some time.")
        V = np.zeros((self.M,self.M,self.M,self.M))
        
        if debug:
            import h5py as h5
            dbg = h5.File("factoredERIs-debug.h5","a")
        
        A = self.Alist
        #for g in range(self.Nvec):
        #    if self.verb:
        #        print("  [+] Adding vector {}".format(g))
        #    A = self.Alist[g]
        #    Adag = A.conj().T 
        #    V+= np.einsum('il,kj->ijkl', A, A)
        
        V = np.einsum('gil,gjk->ijkl',A,A)

        #V = np.dot(Adag, A)
        if debug:
            try:
                dbg.create_dataset("A", data=A)
                dbg.create_dataset("Adag", data=Adag)
                dbg.create_dataset("V", data=V)
            except:
                dbg["A"][...] = A
                dbg["Adag"][...] = Adag
                dbg["V"][...] = V
            dbg.close()
        if self.verb:
            print("reconstituted V : ", V)
            print("reconstituted V shape : ", V.shape)
        return V

def dampedPrescreenCond(diag, vmax, delta):
    '''
    NOTE: This has not been completed!!!! 
    
    here, we evaluate the damped presreening condition as per:
    MOLCAS 7: The Next Generation. J. Comput. Chem., 31: 224-247. 2010. doi:10.1002/jcc.21318

    This function will return a prescreened version of the diagonal:

    a diagonal element, diag_(mu), will be set to zero if:

    s* sqrt( diag_(mu) * vmax ) <= delta

    where mu = (i,l) is a pair index, s is a numerical parameter, v is a Cholesky vector, 
    delta is the cholesky threshold and vmax is the maxium value on the diagonal

    TODO: test this, need a system where the possible numerical instbility could be a problem.
    '''

    s = max([delta*1E9, 1.0])
    toScreen = np.less(diag, 0.0) # this is meant to avoid feeding negative numbers to np.sqrt below
    diag[toScreen] = 0.0

    toScreen = np.less_equal(np.sqrt(diag*vmax), delta)
    diag[toScreen] = 0.0
    return diag


def getCholesky(fromInts = True, onTheFly=True, mol=None, CVfile='V2b_AO_cholesky.mat', tol=1e-8, prescreen=True, debug=False):
    '''
    Front-end 'wrapper' for various Cholesky implentations. The combination of settings 'fromInts'
    and 'onTheFly' determine the precise implementation used.

    inputs:
    
    - 'fromInts' : if True, the Cholesky decomposition will be performed by accessing the GTO basis 
    intergrals computed by pyscf. If False, an already factored form of the 2-body interaction term 
    will be used. This is useful for the "re-Cholesky" procedure used after a frozen orbital 
    transformation which has greatly reduced the size of the Hilbert space.

    - 'onTheFly' : if True, V_{mu nu} super-matrix elements/rows will be computed on the fly. If False,
    the entire super-matrix will be constituted and stored in memory.

    - 'mol' : a pyscf molecule object which specifies the system. (only used if fromInts == True)

    - 'CVfile' : a string containing the name of the file which contains the CVs used to form a 
    representation of V_{mu nu}. Defaults to 'V2b_AO_cholesky.mat' since this is the name that GAFQMC
    will expect for Cholesky-decomposed representation.

    - tol: numerical Cholesky decomposition threshold.
    
    - debug: if True, produces significantly more verbose output, for the purpose of debugging.

    RETURNED:
    
    - Ncv:  the  number of Cholesky vectors produced.
    - CV: a numpy array containing the Cholesky vectors. The array is 2D, CV[g, mu] where g is the 
    Cholesky index and mu is the supermatrix pair-index (i.e. mu = (i,l)). CV has dimension Ncv x M*M
    '''
    # need to implement damped prescreening in each of these.
    if fromInts:
        assert mol is not None, "function: getCholesky() requires a pyscf mol object to compute two-body integrals"
        print("Obtaining two-body matrix elements from integrals (pyscf)")
        if onTheFly:
            print("Computing Integrals on the fly")
            Ncv, CVs = getCholesky_OnTheFly(mol=mol, tol=tol, prescreen=prescreen, debug=debug)
        else:
            print("Storing full V_{ijkl} super-matrix in memomry")
            Ncv, CVs = getCholeskyAO(mol=mol, tol=tol, prescreen=prescreen, debug=debug)
    else:
        assert CVfile is not None, "function: getCholesky() requies the name of a file containing a factored representation of the two-body integrals"
        Ncv, CVs = getCholesky_external(CVfile=CVfile, tol=tol, onTheFly=onTheFly)
    return Ncv, CVs

def getCholesky_external(CVfile='V2b_AO_cholesky.mat', tol=1e-8, onTheFly=True):
    M, Ncv, Alist, AdagList = load_choleskyList_3_IndFormat(infile=CVfile)
    if onTheFly:
        print("Computing Rows on the fly")
        Ncv, CVs = getCholeskyExternal(nbasis=M, Alist=Alist, tol=tol)
    else:
        print("Storing full V_{ijkl} super-matrix in memomry")
        Ncv, CVs = getCholeskyExternal_full(nbasis=M, Alist=Alist, tol=tol)
    return Ncv, CVs

def getCholeskyAO(mol=None, tol=1e-8, prescreen=True, debug=False):
    # Hao's code

    nbasis  = mol.nao_nr()
    # be more careful, this is will use a very large amount of memory!
    eri = scf._vhf.int2e_sph(mol._atm,mol._bas,mol._env)
    V   = ao2mo.restore(1, eri, nbasis)
    V   = V.reshape( nbasis*nbasis, nbasis*nbasis )

    if debug:
        Vorig = V.copy()

    choleskyVecAO = []; choleskyNum = 0
    Vdiag = V.diagonal().copy()
    if debug:
        print("Initial Vdiag: ", Vdiag)

    if prescreen: # zero small diagonal matrix elements - see J. Chem. Phys. 118, 9481 (2003)
        imax = np.argmax(Vdiag); vmax = Vdiag[imax]
        toScreen = np.less(Vdiag, tol*tol/vmax)
        Vdiag[toScreen] = 0.0

    while True:
        imax = np.argmax(Vdiag); vmax = Vdiag[imax]
        print( "Inside modified Cholesky {:<9} {:26.18e}.".format(choleskyNum, vmax) )
        if(vmax<tol or choleskyNum==nbasis*nbasis):
            print( "Number of Cholesky fields is {:9}".format(choleskyNum) )
            print('\n')
            break
        else:
            oneVec = V[imax]/np.sqrt(vmax)
            if debug:
                print("\n***debugging info*** \n")
                print("imax: ", imax, " (i,l) ", (imax // nbasis, imax % nbasis))
                print("vmax: ", vmax)
                print("V[imax]", V[imax])
                print("full V[imax]", Vorig[imax])
            choleskyVecAO.append( oneVec )
            choleskyNum+=1
            V -= np.dot(oneVec[:, None], oneVec[None,:])
            Vdiag -= oneVec**2
            if prescreen:
                Vdiag = dampedPrescreenCond(Vdiag, vmax, tol)
            if debug:
                print("oneVec: ", oneVec)
                print("oneVec**2: ", oneVec**2)
                print("Vdiag: ", Vdiag)

    return choleskyNum, choleskyVecAO

def getCholeskyExternal(nbasis, Alist, tol=1e-8):
    # perform a Cholesky decomposition on a factorized representation of V
    # (i.e. V = sum_g A^g * (A^g)^dagger)
    # Alist is a (3-dimentional numpy array) of the one-body operators A^g_{il}.
    
    V = factoredERIs_updateable(Alist,nbasis,verb=True,useB=True)

    choleskyVecAO = []; choleskyNum = 0
    Vdiag = V.diagonal().copy() 
    print(Vdiag)
    while True:
        imax = np.argmax(Vdiag); vmax = Vdiag[imax]
        print( "Inside modified Cholesky {:<9} {:26.18e}.".format(choleskyNum, vmax) )
        if(vmax<tol or choleskyNum==nbasis*nbasis):
            print( "Number of Cholesky fields is {:9}".format(choleskyNum) )
            print('\n')
            break
        else:
            oneVec = V.row(imax)/np.sqrt(vmax)
            choleskyVecAO.append( oneVec )
            choleskyNum+=1
            V.updateBlist(oneVec.reshape(nbasis,nbasis))
            Vdiag -= oneVec**2

    return choleskyNum, choleskyVecAO

def getCholeskyExternal_full(nbasis, Alist, tol=1e-8):
    # perform a Cholesky decomposition on a factorized representation of V
    # (i.e. V = sum_g A^g * (A^g)^dagger)
    # Alist is a (3-dimensional numpy array) of the one-body operators A^g_{il}.

    Vobj = factoredERIs(Alist,nbasis,verb=True)
    V = Vobj.full()

    choleskyVecAO = []; choleskyNum = 0
    Vdiag = V.diagonal().copy()
    while True:
        imax = np.argmax(Vdiag); vmax = Vdiag[imax]
        print( "Inside modified Cholesky {:<9} {:26.18e}.".format(choleskyNum, vmax) )
        if(vmax<tol or choleskyNum==nbasis*nbasis):
            print( "Number of Cholesky fields is {:9}".format(choleskyNum) )
            print('\n')
            break
        else:
            oneVec = V[imax]/np.sqrt(vmax)
            choleskyVecAO.append( oneVec )
            choleskyNum+=1
            V -= np.dot(oneVec[:, None], oneVec[None,:])
            Vdiag -= oneVec**2

    return choleskyNum, choleskyVecAO

def getCholesky_OnTheFly(mol=None, tol=1e-8, prescreen=True, debug=False):

    nbasis  = mol.nao_nr()
        
    choleskyVecAO = []; choleskyNum = 0
    VmaxList = []
    Vdiag = V2b_diagonal(mol).copy() 

    if debug:
        print("Initial Vdiag: ", Vdiag)
        verb = 5
    else:
        verb = 4


    if prescreen: # zero small diagonal matrix elements - see J. Chem. Phys. 118, 9481 (2003)
        imax = np.argmax(Vdiag); vmax = Vdiag[imax]
        toScreen = np.less(Vdiag, tol*tol/vmax)
        Vdiag[toScreen] = 0.0
    
    while True:
        imax = np.argmax(Vdiag); vmax = Vdiag[imax]
        VmaxList.append(vmax)
        print( "Inside modified Cholesky {:<9} {:26.18e}.".format(choleskyNum, vmax) )
        if(vmax<tol or choleskyNum==nbasis*nbasis):
            print( "Number of Cholesky fields is {:9}".format(choleskyNum) )
            print('\n')
            break
        else:
            if debug:
                print("\n*** getCholesky_OnTheFly: debugging info*** \n")
                print("imax = ", imax, " (i,l) ", (imax // nbasis, imax % nbasis))
                print("vmax = ", vmax)
            Vrow = V2b_row(mol, imax, CVlist=choleskyVecAO, verb=verb)
            oneVec = Vrow/np.sqrt(vmax)
            choleskyVecAO.append(oneVec)                
            if debug:
                print("Vrow", Vrow)
            choleskyNum+=1
            Vdiag -= oneVec**2
            if prescreen:
                Vdiag = dampedPrescreenCond(Vdiag, vmax, tol)
            if debug:
                print("oneVec: ", oneVec)
                print("oneVec**2: ", oneVec**2)
                print("Vdiag: ", Vdiag)
                print("\n*** *** *** ***\n")

    return choleskyNum, choleskyVecAO

def getCholesky_OnTheFly_MOBasis(C, mol=None, tol=1e-8, prescreen=True, debug=False):

    '''
    inputs:

    C - coefficient matrix of the MOs in terms of the AO's 
    '''
    print("[+]: Using experimental Cholesky Implementation : CD is performed in the MO basis, computing integrals in AO basis, and transforming as needed")
    
    nbasis  = mol.nao_nr()

    choleskyVecAO = []; choleskyNum = 0
    VmaxList = []
    Vdiag = V2b_diagonal_MO(mol, C).copy() 

    if debug:
        print("Initial Vdiag: ", Vdiag)
        verb = 5
    else:
        verb = 4


    if prescreen: # zero small diagonal matrix elements - see J. Chem. Phys. 118, 9481 (2003)
        imax = np.argmax(Vdiag); vmax = Vdiag[imax]
        toScreen = np.less(Vdiag, tol*tol/vmax)
        Vdiag[toScreen] = 0.0
    
    while True:
        imax = np.argmax(Vdiag); vmax = Vdiag[imax]
        VmaxList.append(vmax)
        print( "Inside modified Cholesky {:<9} {:26.18e}.".format(choleskyNum, vmax) )
        if(vmax<tol or choleskyNum==nbasis*nbasis):
            print( "Number of Cholesky fields is {:9}".format(choleskyNum) )
            print('\n')
            break
        else:
            if debug:
                print("\n*** getCholesky_OnTheFly: debugging info*** \n")
                print("imax = ", imax, " (i,l) ", (imax // nbasis, imax % nbasis))
                print("vmax = ", vmax)
            Vrow = V2b_row_MO(mol, C, imax, CVlist=choleskyVecAO, verb=verb)
            oneVec = Vrow/np.sqrt(vmax)
            choleskyVecAO.append(oneVec)                
            if debug:
                print("Vrow", Vrow)
            choleskyNum+=1
            Vdiag -= oneVec**2
            if prescreen:
                Vdiag = dampedPrescreenCond(Vdiag, vmax, tol)
            if debug:
                print("oneVec: ", oneVec)
                print("oneVec**2: ", oneVec**2)
                print("Vdiag: ", Vdiag)
                print("\n*** *** *** ***\n")

    return choleskyNum, choleskyVecAO

def getCholeskyAO_MOBasis_DiskIO(mol, C, tol=1e-8, prescreen=True, debug=False, erifile='temp_eri.h5', make_erifile=True):    
    def v_diagonal_file(erifile):
        # efficiently read the integrals from the hdf5 file
        f = h5.File(erifile,"a")
        #shape = f['/new'].shape
        #diag_index = np.zeros(shape[0])
        #for i in range(shape[0]):
        #    diag_index[i] = i
        #diag = f['/new'][diag_index,diag_index] # attempting to use 'fancy indexing'
        diag = f['/new'][...].diagonal().copy() # [?] why do I need copy here?
        f.close()
        return diag

    def v_row_file(erifile, ind, CVlist, M):

        def CV_row(index,CVlist,M):
            '''
            compute row at index 'index' using the current list of Cholesky Vectors 'CVlist'
            
            Note: untested!
            '''
            Sum = np.zeros((M,M))
            #i = index // M
            #l = index % M
            for gamma in range(len(CVlist)):
                L = CVlist[gamma]
                #print(f'[Debug] : _V_row_MO() -> CV_row() -> L = {L}')
                Ldag = L.reshape((M,M)).conj().T
                #print(f'[Debug] : _V_row_MO() -> CV_row() -> Ldag = {Ldag}')
                Sum += L[index]*Ldag
                #print(f'[Debug] : _V_row_MO() -> CV_row() -> Sum = {Sum}')
            return Sum.reshape((M*M))

        # efficiently read the integrals from the hdf5 file
        f = h5.File(erifile,"a")
        row = f['/new'][ind].copy()
        f.close()
        return row - CV_row(ind, CVlist, M)
   
    nbasis  = mol.nao_nr()
    nactive = C.shape[1]
    # be more careful, this is will use a very large amount of memory!
    #eri = scf._vhf.int2e_sph(mol._atm,mol._bas,mol._env)
    #V   = ao2mo.restore(1, eri, nbasis)
    #V   = V.reshape( nbasis*nbasis, nbasis*nbasis )

    # 06042020 - I think we need all of the ERIs, since even the diagonal of Vijkl involes each
    #            of the GTO basis integrals\

    if make_erifile:
        ao2mo.outcore.full(mol, C, erifile, dataname='new', compact=False)
        
    choleskyVecAO = []; choleskyNum = 0
    Vdiag = v_diagonal_file(erifile) #V.diagonal().copy()
    if debug:
        print("Initial Vdiag: ", Vdiag)

    if prescreen: # zero small diagonal matrix elements - see J. Chem. Phys. 118, 9481 (2003)
        imax = np.argmax(Vdiag); vmax = Vdiag[imax]
        toScreen = np.less(Vdiag, tol*tol/vmax)
        Vdiag[toScreen] = 0.0

    while True:
        imax = np.argmax(Vdiag); vmax = Vdiag[imax]
        print( "Inside modified Cholesky {:<9} {:26.18e}.".format(choleskyNum, vmax), flush=True ) # temporary for testing! remove flush=True (does it really make a performace difference, maybe keep?)
        if(vmax<tol or choleskyNum==nactive*nactive):
            print( "Number of Cholesky fields is {:9}".format(choleskyNum) )
            print('\n')
            break
        else:
            oneVec = v_row_file(erifile, imax, choleskyVecAO, nactive)/np.sqrt(vmax)
            if debug:
                print("\n***debugging info*** \n")
                print("imax: ", imax, " (i,l) ", (imax // nactive, imax % nactive))
                print("vmax: ", vmax)
            choleskyVecAO.append( oneVec )
            choleskyNum+=1
            #V -= np.dot(oneVec[:, None], oneVec[None,:])
            Vdiag -= oneVec**2
            if prescreen:
                Vdiag = dampedPrescreenCond(Vdiag, vmax, tol)
            if debug:
                print("oneVec: ", oneVec)
                print("oneVec**2: ", oneVec**2)
                print("Vdiag: ", Vdiag)

    return choleskyNum, choleskyVecAO

def GTO_ints(mol, index_range, verb=True):
        '''
        Inputs:
        mol - Pyscf molecule object describing the system
        index_range - an 8-ten list-like object, containing the index range of desired integrals in the following format (mu_start, mu_stop, nu_start, nu_stop, gamma_start, gamma_stop, delta_start, delta_stop) for V_{mu nu gamma delta}
        
        returns:
        result - np array containing the requested integrals
        '''

        def get_shellRange():
            index_map = map_shellIndex(mol)
            if verb:
                print("index map")
                print("length of map:", len(index_map))
                for entry in index_map:
                    print(entry)
        
            # should be a simple lookup
            shell_range = []
            for i, global_index in enumerate(index_range):
                shell_index = index_map[global_index][0] # index_map contains index pairs (I,i) where I is the shell index, and i is the index within the shell
                if i % 2 == 0: # the start index is inclusive, but the stop index is exclusive
                    shell_range.append(shell_index)
                else:
                    shell_range.append(shell_index+1)

            return shell_range

        # Can use the moleintor.getints('int2e', shls_slice=[:,nu,gamma, delta]) to
        #  get V_mu[nu,gamma,delta] for example

        #result = mol.moleintor.getints('int2e', shls_slice=index_range)
        
        #IMPORTANT, we need to convert from index to shell index before requesting ints!
        if verb:
            print(f'[DEBUG] : index_range = {index_range}' )
            print(f'[DEBUG] : mol.nbas = {mol.nbas}' )
        
        shell_range = get_shellRange()

        if verb:
            print(f'[DEBUG] : shell_range = {shell_range}' )
        
        result = gto.moleintor.getints('int2e_sph',  mol._atm, mol._bas, mol._env, aosym='s1', shls_slice=shell_range)
        if verb:
            print('[DEBUG] : result =', result)
        return result


def GTO_ints_shellInd(mol, shell_range, verb=False):
        '''
        Inputs:
        mol - Pyscf molecule object describing the system
        index_range - an 8-ten list-like object, containing the index range of desired integrals in the following format (mu_start, mu_stop, nu_start, nu_stop, gamma_start, gamma_stop, delta_start, delta_stop) for V_{mu nu gamma delta}
        
        returns:
        result - np array containing the requested integrals
        '''
        if verb:
            print(f'[DEBUG] : shell_range = {shell_range}' )
        
        result = gto.moleintor.getints('int2e_sph',  mol._atm, mol._bas, mol._env, aosym='s1', shls_slice=shell_range)

        return result


def V_diagonal_MO(mol, C, intor='int2e_sph', verb=False):
        '''
        Compute the diagonal of V in an arbitrary MO basis (given by C), while computing GTO integrals on-the-fly. The MO basis may be truncated relative to GTO basis (i.e. M_MO < M_GTO).
       
        Inputs:
        
        mol : Pyscf molecule object spcifying the system (this will be the source of ints)
        C : coefficient matrix specifying the MO orbitals in terms of the GTO basis functions
              C does not have to be a square matrix (in general, it should be:
              |psi_i> = Sum_{mu} C_{mu i} |g_mu> with C an M x M_A matrix

        Returns:
        The diagonal of V in the specified MO basis 


        KNOWN BUG - the einsum below fails if C is a numpy.matrix, but works if C is a numpy.ndarray

        '''

        MA = C.shape[1]
        M = mol.nao_nr()
        Nshell = mol.nbas
        
        Vii = np.zeros((MA,M,M))
        Vdiag = np.zeros((MA,MA))

        offset = gto.ao_loc_nr(mol) # gives the index of the first basis function in each shell (and the last index)

        # the expensive part
        for i in range(MA): # note j=i
            for gamma in range(Nshell):
                for delta in range(Nshell):
                    ints = GTO_ints_shellInd(mol, shell_range=[0,Nshell,delta,delta+1,0,Nshell,gamma,gamma+1])
                    C_dag_row = C.conj().T[i,:]
                    if verb:
                        print(f'[Debug] : ints.shape is {ints.shape} \n C_dag_row.shape is {C_dag_row.shape}')
                    Vii[i,offset[gamma]:offset[gamma+1],offset[delta]:offset[delta+1]] = \
                        np.einsum('m,mdng,n->gd', C_dag_row, ints, C_dag_row, optimize='optimal') # need to double check this one
        
        for l in range(MA):
            C_col = C[:,l].flatten()
            if verb:
                print(f'[Debug] : C_col.shape is {C_col.shape}')
            Vdiag[:,l] = np.einsum('g,igd,d->i', C_col,Vii,C_col,optimize='optimal')

        return Vdiag.flatten()

def V_row_MO(mol, C, row_index, intor='int2e_sph',verb=False):
    '''
        Compute a row of V super-matrix in an arbitrary MO basis (given by C), while computing GTO integrals on-the-fly. The MO basis may be truncated relative to GTO basis (i.e. M_MO < M_GTO).
       
        Inputs:
        
        mol : Pyscf molecule object spcifying the system (this will be the source of ints)
        C : coefficient matrix specifying the MO orbitals in terms of the GTO basis functions
              C does not have to be a square matrix (in general, it should be:
              |psi_i> = Sum_{mu} C_{mu i} |g_mu> with C an M x M_A matrix
        row_index : index of desired row

        Returns:
        The row of V, corresponding to 'row_index' in the specified MO basis 
    '''

    MA = C.shape[1]
    M = mol.nao_nr()
    Nshell = mol.nbas
        
    Vnd = np.zeros((M,M)) # an intermediate matrix V_{[i,l], (nu,gamma)} where [i,l] is the fixed row index, i,l - LMO indices, nu, gamma - GTO indicies
    Vjd = np.zeros((MA,M)) # Vnd with the nu GTO index transformed to the j LMO index
    Vrow = np.zeros((MA,MA))

    offset = gto.ao_loc_nr(mol) # gives the index of the first basis function in each shell (and the last index)

    # convert from supermatrix index to (i,l) pair - note, this relates to the active space, LMO oribtals
    i_global = row_index // MA
    l_global = row_index % MA

    if verb:
        print(f'super-matrix row index = {row_index}, LMO index pair = ({i_global,l_global})')

    # the expensive part
    for nu in range(Nshell):
        for gamma in range(Nshell):
            ints = GTO_ints_shellInd(mol, shell_range=[0,Nshell,0,Nshell,nu,nu+1,gamma,gamma+1])
            C_dag_row = C.conj().T[i_global,:]
            C_col = C[:,l_global].flatten() # is flatten() necessary?
            if verb:
                print(f'[Debug] : V_row_MO() : \n  ->  ints.shape is {ints.shape}\n  ->   C_dag_row.shape is {C_dag_row.shape}\n  ->   C_col.shape is {C_col.shape}')
            Vnd[offset[nu]:offset[nu+1],offset[gamma]:offset[gamma+1]] = \
                        np.einsum('m,mdng,d->ng', C_dag_row, ints, C_col, optimize='optimal') # need to double check this one
        
    # tranform the nu GTO index to the j LMO index
    Vjd = np.einsum('jn,nd->jd', C.conj().T, Vnd, optimize='optimal')
    # tranform the delta GTO index to the k LMO index
    Vrow = np.einsum('jd,dk->jk',Vjd,C)

    return Vrow.flatten()

def getCholeskyAO_MOBasis_NoIO(mol, C, tol=1e-8, prescreen=True, debug=False):

    def _V_diagonal_MO(mol, C, verb=False):
        '''
        Compute the diagonal of V in an arbitrary MO basis (given by C), while computing GTO integrals on-the-fly. The MO basis may be truncated relative to GTO basis (i.e. M_MO < M_GTO).
       
        Inputs:
        
        mol : Pyscf molecule object spcifying the system (this will be the source of ints)
        C : coefficient matrix specifying the MO orbitals in terms of the GTO basis functions
              C does not have to be a square matrix (in general, it should be:
              |psi_i> = Sum_{mu} C_{mu i} |g_mu> with C an M x M_A matrix

        Returns:
        The diagonal of V in the specified MO basis 
        '''
        return V_diagonal_MO(mol, C, verb)
        
    def _V_row_MO(mol, C, index, CVlist, verb=False):

        def CV_row(index,CVlist,M):
            '''
            compute row at index 'index' using the current list of Cholesky Vectors 'CVlist'
            
            Note: untested!
            '''
            Sum = np.zeros((M,M))
            #i = index // M
            #l = index % M
            for gamma in range(len(CVlist)):
                L = CVlist[gamma]
                #print(f'[Debug] : _V_row_MO() -> CV_row() -> L = {L}')
                Ldag = L.reshape((M,M)).conj().T
                #print(f'[Debug] : _V_row_MO() -> CV_row() -> Ldag = {Ldag}')
                Sum += L[index]*Ldag
                #print(f'[Debug] : _V_row_MO() -> CV_row() -> Sum = {Sum}')
            return Sum.reshape((M*M))

        M = C.shape[1]
        return V_row_MO(mol, C, index, verb) - CV_row(index, CVlist,M)

    def print_msg(tol, prescreen):
        '''
        Printing useful information
        '''
        print(f'Performing Modified Cholesky Decomposition with tolerance {tol}')
        if prescreen:
            print("Using damped pre-screening")
        print("Working in orthonormal, molecular orbital basis")
        print("Computing GTO intergals on the fly")


    ##### IMPORTANT TO_DO #####: Need to produce the embedding potential too!

    print_msg(tol, prescreen)

    nbasis  = mol.nao_nr() # [?] needed?
    nactive = C.shape[1]
    choleskyVecMO = []; choleskyNum = 0
    Vdiag = _V_diagonal_MO(mol, C, verb=debug)

    if debug:
        print("Initial Vdiag: ", Vdiag)

    if prescreen: # zero small diagonal matrix elements - see J. Chem. Phys. 118, 9481 (2003)
        imax = np.argmax(Vdiag); vmax = Vdiag[imax]
        toScreen = np.less(Vdiag, tol*tol/vmax)
        Vdiag[toScreen] = 0.0

    while True:
        imax = np.argmax(Vdiag); vmax = Vdiag[imax]
        print( "Inside modified Cholesky {:<9} {:26.18e}.".format(choleskyNum, vmax) )
        if(vmax<tol or choleskyNum==nactive*nactive):
            print( "Number of Cholesky fields is {:9}".format(choleskyNum) )
            print('\n')
            break
        else:
            oneVec = _V_row_MO(mol, C, imax, CVlist=choleskyVecMO, verb=debug)/np.sqrt(vmax) # TODO: need to sub CVs in row!
            if debug:
                print("\n***debugging info*** \n")
                print("imax: ", imax, " (i,l) ", (imax // nactive, imax % nactive))
                print("vmax: ", vmax)
            choleskyVecMO.append( oneVec )
            choleskyNum+=1
            #V -= np.dot(oneVec[:, None], oneVec[None,:]) # this won't work since we are not storing the full V tensor
            Vdiag -= oneVec**2
            if prescreen:
                Vdiag = dampedPrescreenCond(Vdiag, vmax, tol)
            if debug:
                print("oneVec: ", oneVec)
                print("oneVec**2: ", oneVec**2)
                print("Vdiag: ", Vdiag)

    return choleskyNum, choleskyVecMO


def make_green_func(Phi, Psi, S=None, debug=False):
    # Assumes that Phi and Psi are represented in an orthonormal basis!
    if S is None:
        S = np.eye(Phi.shape[0])
    O = np.matmul(Psi.conj().T,S)
    O = np.matmul(O,Phi)
    if debug:
        print(f'overlap of Phi and Psi is {np.linalg.det(O)}')
    Oinv = np.linalg.inv(O)
    Theta = np.matmul(Phi, Oinv)
    G = np.matmul(Theta, Psi.conj().T)
    return G

def get_embedding_potential(mol, C, nfc, Nel, MA, debug=False):
    '''
    computes the effective 1-body embedding potential V^{I_A}_{il}

    Inputs:
    mol - Pyscf molecule object describing the system
    C - array containing the basis orbitals - including both inactive, and active occupied orbitals!
    nfc - number of orbitals to freeze : the first nfc orbitals (that is C[:,0:nfc]) are forzen
    Nel - total number of !!doubly ocupied!! electrons 
    MA - number of active orbitals

    returns:
    VIA - embedding potential

    IDEAS:
    - use get Veff from mf object? -> doesn't use transformed Veff!
    '''

    
    if debug:
    #    logging.basicConfig(filename='debug.log',level=logging.DEBUG)
        logging.basicConfig(level=logging.DEBUG)
    
    M = mol.nao_nr()
    Nshell = mol.nbas
    offset = gto.ao_loc_nr(mol)

    C_core = C[:,:nfc]
    C_CoreDag = C_core.conj().T
    G_core = make_green_func(C_core, C_core,debus=True)
    C_active =C[:, nfc:] # assuming C has already had virtuals truncated!

    if debug:
        logging.debug(f' nfc, Nel, MA : {nfc}, {Nel}, {MA}')
        #logging.debug(f'G_core : {G_core}')

    first=True

    # get intermediate results
    VmnkL = np.zeros((M,M,MA,nfc)) # used for both Vd and Vx
    for mu in range(Nshell):
        for nu in range(Nshell):
            # get V[mu,:,nu,:] in Chemist's index convention = V(mu,nu,:,:) in Physicist's
            ints = GTO_ints_shellInd(mol, shell_range=[mu,mu+1,0,Nshell,nu,nu+1,0,Nshell])
            if debug and first:
                first=False
                logging.debug(f' shape of ints {ints.shape}')
                logging.debug(f' shape of C_core {C_core.shape}')
                logging.debug(f' shape of C_core.conj().T {C_core.conj().T.shape}')
                logging.debug(f' shape of C_active {C_active.shape}')
                logging.debug(f' shape of C_active.conj().T {C_active.conj().T.shape}')
                logging.debug(f' shape of VmnkL {VmnkL.shape}')
            VmnkL[offset[mu]:offset[mu+1],offset[nu]:offset[nu+1],:,:] = \
                    np.einsum('dg,dl,gk->kl',ints[0,:,0,:],C_core,C_active,optimize='optimal')

    # __bookmark__
    # TODO : BUG : index ordering seems to be incorrect!
    Vd = np.einsum('im,jn,mnkl,il->jk',C_core.conj().T, C_active.conj().T, VmnkL,G_core,optimize='greedy')
    Vx = np.einsum('im,jn,mnkl,jl->ik',C_active.conj().T, C_core.conj().T, VmnkL,G_core,optimize='greedy')
    
    if debug:
        logging.debug(f'Vd : {Vd}')
        logging.debug(f'Vx : {Vx}')

    return 2*Vd - Vx

def get_embedding_potential_2(mol, C, nfc, Nel, MA, debug=False):
    '''
    computes the effective 1-body embedding potential V^{I_A}_{il}

    Inputs:
    mol - Pyscf molecule object describing the system
    C - array containing the basis orbitals - including both inactive, and active occupied orbitals!
    nfc - number of orbitals to freeze : the first nfc orbitals (that is C[:,0:nfc]) are forzen
    Nel - total number of !!doubly ocupied!! electrons 
    MA - number of active orbitals

    returns:
    VIA - embedding potential

    IDEAS:
    - use get Veff from mf object? -> doesn't use transformed Veff!
    '''

    
    if debug:
    #    logging.basicConfig(filename='debug.log',level=logging.DEBUG)
        logging.basicConfig(level=logging.DEBUG)
    
    M = mol.nao_nr()
    Nshell = mol.nbas
    offset = gto.ao_loc_nr(mol)

    C_core = C[:,:nfc]
    C_CoreDag = C_core.conj().T
    # TODO: give the make_green_func function to use an overlap matrix!
    #G_core = make_green_func(C_core, C_core, debug=debug) #TODO: restirct to core only!!
    G_core = np.eye(nfc)# temporary fix!
    logging.debug(f'shape of G_core {G_core.shape}')
    logging.debug(f'G = {G_core}')
    C_active =C[:, nfc:] # assuming C has already had virtuals truncated!

    if debug:
        logging.debug(f' nfc, Nel, MA : {nfc}, {Nel}, {MA}')
        #logging.debug(f'G_core : {G_core}')

    first=True

    # TODO: delete numpy as arrays as we finish with them, and garbage collect
    # get intermediate results
    print('[+] getting GTO integrals ...')
    VmnkL = np.zeros((M,M,MA,nfc)) # used for both Vd and Vx
    for mu in range(Nshell):
        for nu in range(Nshell):
            # get V[mu,:,nu,:] in Chemist's index convention = V(mu,nu,:,:) in Physicist's
            ints = GTO_ints_shellInd(mol, shell_range=[mu,mu+1,0,Nshell,nu,nu+1,0,Nshell])
            if debug and first:
                first=False
                logging.debug(f' shape of ints {ints.shape}')
                logging.debug(f' shape of C_core {C_core.shape}')
                logging.debug(f' shape of C_core.conj().T {C_core.conj().T.shape}')
                logging.debug(f' shape of C_active {C_active.shape}')
                logging.debug(f' shape of C_active.conj().T {C_active.conj().T.shape}')
                logging.debug(f' shape of VmnkL {VmnkL.shape}')
            VmnkL[offset[mu]:offset[mu+1],offset[nu]:offset[nu+1],:,:] = \
                    np.einsum('mdng,dl,gk->mnkl',ints[:,:,:,:],C_core,C_active,optimize='optimal') 

    # __bookmark__
    #Vd = np.einsum('im,jn,mnkl,il->jk',C_core.conj().T, C_active.conj().T, VmnkL,G_core,optimize='greedy')
    #Vx = np.einsum('im,jn,mnkl,jl->ik',C_active.conj().T, C_core.conj().T, VmnkL,G_core,optimize='greedy')

    print('[+] computing Vd ...')
    VInkL = np.einsum('im,mnkl->inkl',C_core.conj().T,VmnkL)
    if debug:
        logging.debug(f' shape of VInkL {VInkL.shape}')
    VIjkL = np.einsum('jn,inkl->ijkl',C_active.conj().T,VInkL)
    if debug:
        logging.debug(f' shape of VIjkL {VIjkL.shape}') 
    Vd = np.einsum('il,ijkl->jk',G_core,VIjkL)
    if debug:
        logging.debug(f' shape of Vd {Vd.shape}')

    print('[+] computing Vx ...')
    VmJkL = np.einsum('jn,mnkl->mjkl',C_core.conj().T,VmnkL)
    if debug:
        logging.debug(f' shape of VmJkL {VmJkL.shape}')
    ViJkL = np.einsum('im,mjkl->ijkl',C_active.conj().T,VmJkL)
    if debug:
        logging.debug(f' shape of ViJkL {ViJkL.shape}')
    Vx = np.einsum('jl,ijkl->ik',G_core,ViJkL)
    if debug:
        logging.debug(f' shape of Vx {Vx.shape}')

    if debug:
        logging.debug(f'Vd : {Vd}')
        logging.debug(f'Vx : {Vx}')

    return 2*Vd - Vx


def get_embedding_potential_useh5(mol, C, nfc, Nel, MA, debug=False, make_erifile=True):
    '''
    computes the effective 1-body embedding potential V^{I_A}_{il}, using pyscf's interface to libcint, and ao2mo for 
    GTO integrals, and integral transformations, respectively

    Inputs:
    mol - Pyscf molecule object describing the system
    C - array containing the basis orbitals - including both inactive, and active occupied orbitals!
    nfc - number of orbitals to freeze : the first nfc orbitals (that is C[:,0:nfc]) are forzen
    Nel - total number of !!doubly ocupied!! electrons 
    MA - number of active orbitals

    returns:
    VIA - embedding potential
    '''

    
    if debug:
        logging.basicConfig(level=logging.DEBUG)
    
    M = mol.nao_nr()
    Nshell = mol.nbas
    offset = gto.ao_loc_nr(mol)

    C_core = C[:,:nfc]
    C_CoreDag = C_core.conj().T

    G_core = np.eye(nfc)# temporary fix!
    logging.debug(f'shape of G_core {G_core.shape}')
    logging.debug(f'G = {G_core}')
    C_active =C[:, nfc:] # assuming C has already had virtuals truncated!

    if debug:
        logging.debug(f' nfc, Nel, MA : {nfc}, {Nel}, {MA}')

    first=True

    # compute the direct term as G_{I L} * V_{I j k L} -> Pyscf (Chemists') notation, want (IL|jk) mo integrals
    print('[+] computing Vd ...')
    if make_erifile:
        ao2mo.outcore.general(mol, (C_core,C_core,C_active,C_active), erifile='eri_coreValence.h5', dataname='direct', compact=False, aosym=1)
    f = h5.File('eri_coreValence.h5', 'r')
    eri_mo = f['/direct'][...].reshape((nfc,nfc,MA,MA))
    Vd = np.einsum('il,iljk->jk',G_core,eri_mo)
    if debug:
        logging.debug(f' shape of Vd {Vd.shape}')
    f.close()

    # compute the direct term as G_{I L} * V_{i J k L} -> Pyscf (Chemists') notation, want (iL|Jk) mo integrals
    print('[+] computing Vx ...')
    if make_erifile:
        ao2mo.outcore.general(mol, (C_active,C_core,C_core,C_active), erifile='eri_coreValence.h5', dataname='exchange', compact=False, aosym=1)
    f = h5.File('eri_coreValence.h5', 'r')
    eri_mo = f['/exchange'][...].reshape((MA,nfc,nfc,MA))
    Vx = np.einsum('jl,iljk->ik',G_core,eri_mo)
    if debug:
        logging.debug(f' shape of Vx {Vx.shape}')
    f.close()

    if debug:
        logging.debug(f'Vd : {Vd}')
        logging.debug(f'Vx : {Vx}')

    return 2*Vd - Vx

def get_embedding_potential_incore(mol, C, nfc, Nel, MA, debug=False):
    '''
    computes the effective 1-body embedding potential V^{I_A}_{il}, using pyscf's interface to libcint, and ao2mo for 
    GTO integrals, and integral transformations, respectively. Uses the incore ao2mo transform to avoid disk IO for faster
    execution time assuming enough memory is available

    Inputs:
    mol - Pyscf molecule object describing the system
    C - array containing the basis orbitals - including both inactive, and active occupied orbitals!
    nfc - number of orbitals to freeze : the first nfc orbitals (that is C[:,0:nfc]) are forzen
    Nel - total number of !!doubly ocupied!! electrons 
    MA - number of active orbitals

    returns:
    VIA - embedding potential
    '''

    if debug:
        logging.basicConfig(level=logging.DEBUG)
    
    M = mol.nao_nr()
    Nshell = mol.nbas
    offset = gto.ao_loc_nr(mol)

    C_core = C[:,:nfc]
    C_CoreDag = C_core.conj().T

    G_core = np.eye(nfc)# temporary fix!
    logging.debug(f'shape of G_core {G_core.shape}')
    logging.debug(f'G = {G_core}')
    C_active =C[:, nfc:] # assuming C has already had virtuals truncated!

    if debug:
        logging.debug(f' nfc, Nel, MA : {nfc}, {Nel}, {MA}')

    # compute the direct term as G_{I L} * V_{I j k L} -> Pyscf (Chemists') notation, want (IL|jk) mo integrals
    print('[+] computing Vd ...')
    eri_mo = ao2mo.incore.general(mol, (C_core,C_core,C_active,C_active), compact=False, aosym=1)
    Vd = np.einsum('il,iljk->jk',G_core,eri_mo)
    if debug:
        logging.debug(f' shape of Vd {Vd.shape}')
    f.close()

    # compute the direct term as G_{I L} * V_{i J k L} -> Pyscf (Chemists') notation, want (iL|Jk) mo integrals
    print('[+] computing Vx ...')        
    eri_mo = ao2mo.incore.general(mol, (C_active,C_core,C_core,C_active), compact=False, aosym=1)
    Vx = np.einsum('jl,iljk->ik',G_core,eri_mo)
    if debug:
        logging.debug(f' shape of Vx {Vx.shape}')

    if debug:
        logging.debug(f'Vd : {Vd}')
        logging.debug(f'Vx : {Vx}')

    return 2*Vd - Vx


def get_embedding_constant_useh5(mol, C, nfc, debug=False, make_erifile=True):
    '''
    computes the constant two-body interactions among frozen electrons

    Inputs:
    mol - Pyscf molecule object describing the system
    C - array containing the basis orbitals - including both inactive, and active occupied orbitals!
    nfc - number of orbitals to freeze : the first nfc orbitals (that is C[:,0:nfc]) are forzen
  
    returns:
    EI - constant interaction energy of Inactive orbitals

    TODO:
    optimize a bit - using a lazy implementation
    > add option to treat Green's function as non-diagonal 
    '''

    if debug:
        logging.basicConfig(level=logging.DEBUG)
    
    M = mol.nao_nr()
    Nshell = mol.nbas
    offset = gto.ao_loc_nr(mol)

    Cfc = C[:,:nfc]
    Cfc_dag = Cfc.conj().T

    # [?] couldn't we go with an incore tranformation? the # of frozen core orbitals depends
    #   doesn't grow the GTO basis
    if make_erifile:
        ao2mo.outcore.full(mol, Cfc, erifile='eri_core.h5', dataname='new', compact=False)
    
    Vd=0.0
    Vx=0.0

    f = h5.File('eri_core.h5', 'r')
    eri_mo = f['/new'][...].reshape((nfc,nfc,nfc,nfc))
    
    Vd+=np.einsum('iijj->',eri_mo)
    Vx+=np.einsum('ijji->',eri_mo)

    if debug:
        logging.debug(f'Vd : {Vd}')
        logging.debug(f'Vx : {Vx}')

    f.close()
    return 2*Vd - Vx

def get_embedding_constant_incore(mol, C, nfc, debug=False):
    '''
    computes the constant two-body interactions among frozen electrons, using the incore (pyscf) ao2mo transformation 

    Inputs:
    mol - Pyscf molecule object describing the system
    C - array containing the basis orbitals - including both inactive, and active occupied orbitals!
    nfc - number of orbitals to freeze : the first nfc orbitals (that is C[:,0:nfc]) are forzen
  
    returns:
    EI - constant interaction energy of Inactive orbitals
    '''

    if debug:
        logging.basicConfig(level=logging.DEBUG)
    
    M = mol.nao_nr()
    Nshell = mol.nbas
    offset = gto.ao_loc_nr(mol)

    Cfc = C[:,:nfc]
    Cfc_dag = Cfc.conj().T

    eri_mo = ao2mo.incore.full(mol, Cfc, compact=False)

    Vd=0.0
    Vx=0.0

    Vd+=np.einsum('iijj->',eri_mo)
    Vx+=np.einsum('ijji->',eri_mo)

    if debug:
        logging.debug(f'Vd : {Vd}')
        logging.debug(f'Vx : {Vx}')

    return 2*Vd - Vx

def get_embedding_constant(mol, C, nfc, debug=False):
    '''
    computes the constant two-body interactions among frozen electrons

    Inputs:
    mol - Pyscf molecule object describing the system
    C - array containing the basis orbitals - including both inactive, and active occupied orbitals!
    nfc - number of orbitals to freeze : the first nfc orbitals (that is C[:,0:nfc]) are forzen
  
    returns:
    EI - constant interaction energy of Inactive orbitals

    TODO:
    optimize a bit - using a lazy implementation
    > add option to treat Green's function as non-diagonal 
    '''

    if debug:
        logging.basicConfig(level=logging.DEBUG)
    
    M = mol.nao_nr()
    Nshell = mol.nbas
    offset = gto.ao_loc_nr(mol)

    Cfc = C[:,:nfc]
    Cfc_dag = Cfc.conj().T

    Vd=0.0
    Vx=0.0

    first=True
    
    for mu in range(Nshell):
        for nu in range(Nshell):
            # get V[mu,:,nu,:] in Chemist's index convention = V(mu,nu,:,:) in Physicist's
            ints = GTO_ints_shellInd(mol, shell_range=[mu,mu+1,0,Nshell,nu,nu+1,0,Nshell])
            if debug and first: 
                first=False
                logging.debug(f' shape of ints {ints.shape}')
                logging.debug(f' shape of Cfc_dag {Cfc_dag.shape}')
                logging.debug(f' shape of Cfc {Cfc.shape}')
            # we are using the offsets below because we are accessing the ints shell-by-shell here in mu, nu indices, but full basis in gamma, delta indices
            Vd+=np.einsum('im,jn,mdng,gj,di->',Cfc_dag[:,offset[mu]:offset[mu+1]],Cfc_dag[:,offset[nu]:offset[nu+1]],ints,Cfc,Cfc,optimize='optimal') #TODO: only run optimize on first runs then save/reuse the path!
            Vx+=np.einsum('im,jn,mdng,gi,dj->',Cfc_dag[:,offset[mu]:offset[mu+1]],Cfc_dag[:,offset[nu]:offset[nu+1]],ints,Cfc,Cfc,optimize='optimal')
    
    if debug:
        logging.debug(f'Vd : {Vd}')
        logging.debug(f'Vx : {Vx}')

    return 2*Vd - Vx

def get_one_body_embedding(mol, C, nfc, debug=False):
    '''
    Computes and returns the contributions to the embedding / downfolding Hamiltonian due to the one-body terms in the full Hilbert space
    
    Inputs:
    mol - Pyscf molecule object describing the system
    C - array containing the basis orbitals - including both inactive, and active occupied orbitals!
    nfc - number of orbitals to freeze : the first nfc orbitals (that is C[:,0:nfc]) are forzen
  
    returns:

    K_active - one-body Hamiltonian terms in the active space (not including the effective one-body embedding potential from two-body interactions - use 'get_embedding_potential' instead
    EI_K - constant energy, due to frozen electrons, from K

    '''
    if debug:
        logging.basicConfig(level=logging.DEBUG)

    MA = C.shape[1] - nfc

    logging.debug(f' get_one_body_embedding : C shape : {C.shape}')
    logging.debug(f' get_one_body_embedding : MA : {MA}')

    # get full one-body Hamiltonian
    K = get_one_body_H(mol)
    S = get_ovlp(mol)
    
    # K_active is simply K within the chosen active space with no extra transforms
    K_active = np.einsum('im,mn,nl->il', C.conj().T[nfc:,:],K,C[:,nfc:]) #K[nfc:nfc+MA,nfc:nfc+MA]
    logging.debug(f' get_one_body_embedding : K_active shape : {K_active.shape}')
    
    # get G_Core_{IL}
    G_core = np.eye(nfc) # make_green_func(C[:,:nfc],C[:,:nfc])[:nfc, :nfc]
    logging.debug(f' get_one_body_embedding : G_core : {G_core.shape}')

    # EI_K is given by K_{IL} * G{IL} - wrong! should be [C^dag K C]_{II} since K is in GTO basis
    #EI_K = np.trace(np.matmul(K[:nfc,:nfc],G_core))
    EI_K = np.einsum('im,mn,ni->',C.conj().T[:nfc,:],K,C[:,:nfc])

    return K_active, EI_K
