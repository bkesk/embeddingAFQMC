import numpy as np
import h5py as h5
import sys
import logging

import pyscf
from pyscf  import ao2mo, scf, gto
from pyscf.gto import getints_by_shell

#from V2b_inspect import load_V2b_dump, save_V2b_dump, sym_2d_unpack, sym_2d_pack
#import pyqmc.matrices.gms as gms

#from .input_mod import save_cholesky, load_cholesky, save_one_body, load_one_body

from .integrals.base import IntegralGenerator

'''

THis is a simplified Cholesky module. Let's take care to follow some good style!


Format - Choleksy vectors will be called generically called A, and will be stored in a 3d array with shape (Ncv,M,M)


 TODO : use submodules to define different types of Integral generators

 TODO : need to standardize how we are working:

        What format do we want to work with the CVs in? -> numpy array with shape (Ncv,M,M)
                 - this should allow quick linear algebra operations on the last 2 axes (row-major arrays)
                 - avoids constant "reshape()" calls which needlessly harm peformance we actually need shape 
                   (Ncv,M,M) vs (Ncv, M*M) for some operations. or a clever way to transpose the CVs
        
        Structure: How can we structure the code so that we don't keep copy-pasting the same CD algorithm just using
                      different functions to get the diagonal, and to get the rows?
           
                  def cholesky(..., row=V2b_row, diag=V2b_diag) ?
                       # do work
                       return num_cholesky, cvs

                  - For this to work, all row / coloumn functions need to take the same arguments (and ignore extra args)
                     which is doable, since they work in essentially the same way
        
         naming! horribly inconsistent!!!

'''

class _IntegralGenerator:
    def __init__(self,*args,**kwargs):
        pass

    def row(self,index,*args,**kwargs):
        return self.get_row(index,*args,**kwargs)

    def diagonal(self,*args,**kwargs):
        return self.get_diagonal(*args,**kwargs)


class _GTOIntegralGenerator(IntegralGenerator):
    def __init__(self,mol,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.mol = mol
    
    def get_row(self, index,*args,**kwargs):
        return V2b_row(self.mol, index, Alist=kwargs['Alist'])

    def get_diagonal(self,*args,**kwargs):
        return V2b_diagonal(self.mol)

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
   
    Vdiag =np.zeros((nbasis,nbasis))

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
                    Vdiag[i_global, l_global] = shell_ints[i,l,j,k]
            if verb > 5:
                print("shell pair index = ", pairIndS)
            pairIndS += 1
        
    return Vdiag

def V2b_row(mol, mu, Alist=None, intor_name='int2e_sph', verb=None):
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
   
    Vrow=np.zeros((nbasis,nbasis)) # row indices are (i,l), column are (j,k)

    index_Map = map_shellIndex(mol)
    if verb > 5:
        print("index map")
        print("length of map:", len(index_Map))
        for entry in index_Map:
            print(entry)
    
    pairIndS = 0
    
    num_shells= mol.nbas


    i_global = mu // nbasis
    l_global = mu % nbasis
    
    I, i = index_Map[i_global]
    L, l = index_Map[l_global]

    Vrow = GTO_ints_shellInd(mol, shell_range=[I,I+1,L,L+1,0,num_shells,0,num_shells], verb=(verb>5))[i,l,:,:]
 
    if Alist is not None and len(Alist) > 0:
             
        def build_row(Alist, mu, debug=False):
            '''
            compute the diagonal of V from the set of vectors, A
            if a,b are pair indices:
        
            V_ab = sum_g (A^g_a * (A^g)^dagger_b)
            
            and the diagnonl is given by: 
            V_aa = sum_g (A^g_a *((A^g)^dagger_a))
            
            Alist is a 3-D numpy array with shape (# cholesky vectors, nbasis, nbasis)
            '''
            
            #TODO: implement for complex CVs
            #AdagList=np.transpose(Alist.conj(),axes=[0,2,1]) # how expensive is this?
            
            M = Alist.shape[1] #int(np.sqrt(Alist.shape[1]))
            i = mu // M
            l = mu % M
            #row = np.einsum('g,gjk',Alist[:,i,l],AdagList)
            row = np.einsum('g,gjk',Alist[:,i,l],Alist) # this works since A are real

            return row

        Ncv = len(Alist)
        if verb > 4:
            print("\n   *** V2b_row: debug info ***\n")
            print("Vrow direct from integrals", Vrow)
            Vrow_temp = Vrow.copy()

        Vrow = Vrow - build_row(Alist, mu)
    
        if verb > 4:
            print("   Vrow - A*A^dag (i.e. residual matrix row):", Vrow)
            print("   A*A^dag (direct function call to build_row()):", build_row(Alist, i_global*nbasis + l_global, nbasis))
            print("   delta Vrow: ", Vrow_temp - Vrow)
            print("\n   *** *** *** ***\n")
            
    return Vrow

def dampedPrescreenCond(diag, vmax, delta, s=None):
    '''
    NOTE: This has not been completed!!!! 
    
    here, we evaluate the damped presreening condition as per:
    MOLCAS 7: The Next Generation. J. Comput. Chem., 31: 224-247. 2010. doi:10.1002/jcc.21318

    Also see J. Chem. Phys. 118, 9481 (2003)

    This function will return a prescreened version of the diagonal:

    a diagonal element, diag_(mu), will be set to zero if:

    s* sqrt( diag_(mu) * vmax ) <= delta

    where mu = (i,l) is a pair index, s is a numerical parameter, v is a Cholesky vector, 
    delta is the cholesky threshold and vmax is the maxium value on the diagonal

    s is the damping factor and typically:
    s = max([delta*1E9, 1.0])

    TODO: test this, need a system where the possible numerical instbility could be a problem.
    '''

    if s is None:
        s = max([delta*1E9, 1.0])

    # need to clear negative values (should all be zero anyway) to feed to np.sqrt
    negative = np.less(diag, 0.0) 
    diag[negative] = 0.0

    sDeltaSqr=(delta/s)*(delta/s)
    # the actual damped prescreening
    #toScreen = np.less_equal(s*np.sqrt(diag*vmax), delta)
    toScreen = np.less(diag*vmax, sDeltaSqr)
    diag[toScreen] = 0.0
    return diag, toScreen

def cholesky(integral_generator=None,tol=1.0E-8,prescreen=True,debug=False,max_cv=None):

    if not isinstance(integral_generator, IntegralGenerator): 
        raise TypeError('Invalide integral generator, must have base class IntegralGenerator')
    
    # TDOO check inputs
    nbasis = integral_generator.nbasis

    delCol = np.zeros((nbasis,nbasis),dtype=bool)
    choleskyNum = 0
    
    if max_cv:
        choleskyNumGuess = max_cv
    else:
        choleskyNumGuess = 10*nbasis
    cholesky_vec = np.zeros((choleskyNumGuess,nbasis,nbasis))
    
    Vdiag = integral_generator.diagonal() #V2b_diagonal_Array(mol).copy()

    if debug:
        print("Initial Vdiag: ", Vdiag)
        verb = 5
    else:
        verb = 4

    # np.argmax returns the 'flattened' max index -> need to 'unflatten'
    unflatten = lambda x : (x // nbasis, x % nbasis)
    
    # Note: screening should work still as long as we keep shape of Vdiag consistent!
    if prescreen: # zero small diagonal matrix elements - see J. Chem. Phys. 118, 9481 (2003)
        imax = np.argmax(Vdiag); vmax = Vdiag[unflatten(imax)]
        toScreen = np.less(Vdiag, tol*tol/vmax)
        Vdiag[toScreen] = 0.0
    
    while True:
        imax = np.argmax(Vdiag)
        vmax = Vdiag[unflatten(imax)]
        print( "Inside modified Cholesky {:<9} {:26.18e}".format(choleskyNum, vmax), flush=True )
        if(vmax<tol or choleskyNum==nbasis*nbasis or choleskyNum >= choleskyNumGuess):
            print( "Number of Cholesky fields is {:9}".format(choleskyNum) )
            print('\n')
            if (choleskyNum >= choleskyNumGuess):
                print(f'WARNING!!! : reached maximum number of Cholesky vectors {choleskyNumGuess}. \n use \'max_cv\' option to use a larger value. ')
            break
        else:
            if debug:
                print("\n*** getCholesky_OnTheFly: debugging info*** \n")
                print("imax = ", imax, " (i,l) ", (imax // nbasis, imax % nbasis))
                print("vmax = ", vmax)
            #_bookmark
            Vrow = integral_generator.row(imax, Alist=cholesky_vec[:choleskyNum,:,:]) #V2b_row_Array(mol, imax, Alist=cholesky_vec[:choleskyNum,:,:], verb=verb)
            oneVec = Vrow/np.sqrt(vmax)
            if prescreen:
                oneVec[delCol]=0.0
            cholesky_vec[choleskyNum] = oneVec                
            if debug:
                print("Vrow", Vrow)
            choleskyNum+=1
            Vdiag -= oneVec**2
            if prescreen:
                Vdiag, removed = dampedPrescreenCond(Vdiag, vmax, tol) # should work for any shape of Vdiag (1xM*M or MxM)
                delCol[removed] = True 
            if debug:
                print("oneVec: ", oneVec)
                print("oneVec**2: ", oneVec**2)
                print("Vdiag: ", Vdiag)
                print("\n*** *** *** ***\n")

    return choleskyNum, cholesky_vec[:choleskyNum,:,:]

