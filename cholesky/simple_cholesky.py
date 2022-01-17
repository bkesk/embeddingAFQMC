import numpy as np

from .integrals.base import IntegralGenerator

def dampedPrescreenCond(diag, vmax, delta, s=None):
    '''    
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
    '''

    if s is None:
        s = max([delta*1E9, 1.0])

    # need to clear small negative values (should all be zero anyway) to feed to np.sqrt
    negative = np.less(diag, 0.0) 
    diag[negative] = 0.0

    sDeltaSqr=(delta/s)*(delta/s)
    
    # the actual damped prescreening
    toScreen = np.less(diag*vmax, sDeltaSqr)
    diag[toScreen] = 0.0
    return diag, toScreen

def cholesky(integral_generator=None,tol=1.0E-8,prescreen=True,debug=False,max_cv=None):

    if not isinstance(integral_generator, IntegralGenerator): 
        raise TypeError('Invalid integral generator, must have base class IntegralGenerator')
    
    nbasis = integral_generator.nbasis

    delCol = np.zeros((nbasis,nbasis),dtype=bool)
    choleskyNum = 0
    
    if max_cv:
        choleskyNumGuess = max_cv
    else:
        choleskyNumGuess = 10*nbasis
    cholesky_vec = np.zeros((choleskyNumGuess,nbasis,nbasis))
    
    Vdiag = integral_generator.diagonal()

    if debug:
        print("Initial Vdiag: ", Vdiag)
        verb = 5
    else:
        verb = 4

    # np.argmax returns the 'flattened' max index -> need to 'unflatten'
    unflatten = lambda x : (x // nbasis, x % nbasis)
    
    if prescreen: # zero small diagonal matrix elements - see J. Chem. Phys. 118, 9481 (2003)
        if debug:
            imax = np.argmax(Vdiag)
            print("imax , unflatten(imax) = ", imax, unflatten(imax))
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
            break
        else:
            if debug:
                print("\n*** getCholesky_OnTheFly: debugging info*** \n")
                print("imax = ", imax, " (i,l) ", (imax // nbasis, imax % nbasis))
                print("vmax = ", vmax)
            #_bookmark
            Vrow = integral_generator.row(imax, Alist=cholesky_vec[:choleskyNum,:,:])
            oneVec = Vrow/np.sqrt(vmax)
            if prescreen:
                oneVec[delCol]=0.0
            cholesky_vec[choleskyNum] = oneVec                
            if debug:
                print("Vrow", Vrow)
            choleskyNum+=1
            Vdiag -= oneVec**2
            if prescreen:
                Vdiag, removed = dampedPrescreenCond(Vdiag, vmax, tol)
                delCol[removed] = True 
            if debug:
                print("oneVec: ", oneVec)
                print("oneVec**2: ", oneVec**2)
                print("Vdiag: ", Vdiag)
                print("\n*** *** *** ***\n")

    return choleskyNum, cholesky_vec[:choleskyNum,:,:]
