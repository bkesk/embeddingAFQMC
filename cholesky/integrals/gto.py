import numpy as np

from pyscf import gto
from pyscf.gto import getints_by_shell

from .base import IntegralGenerator

class GTOIntegralGenerator(IntegralGenerator):
    def __init__(self,mol,*args,**kwargs):
        if not isinstance(mol,gto.Mole):
            raise TypeError('mol must be a Pyscf molecule object')

        super().__init__(*args,**kwargs)
        self.mol = mol
        self.nbasis = mol.nao_nr()
    
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
