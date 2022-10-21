import numpy as np
import h5py as h5

def writeModel(nElec,oneBody,twoBody,fname='model_param'):
    ''' Export Hamiltonian to the disk.'''
    
    oneBodyAdj = oneBody.copy()

    for i in range(twoBody.shape[0]):
        oneBodyAdj += (-0.5)*np.dot( twoBody[i], twoBody[i] )

    choleskyN = twoBody.shape[0]
    basisN = twoBody.shape[1]

    with h5.File(fname, "w") as f:
        f.create_dataset("L",               (1,),                    data=[basisN],             dtype='int')
        f.create_dataset("Nup",             (1,),                    data=[nElec[0]],           dtype='int')
        f.create_dataset("Ndn",             (1,),                    data=[nElec[1]],           dtype='int')
        f.create_dataset("choleskyNumber",  (1,),                    data=[choleskyN],          dtype='int')
        f.create_dataset("t",               (basisN**2,),            data=oneBody.ravel(),      dtype='float64')
        f.create_dataset("K",               (basisN**2,),            data=oneBodyAdj.ravel(),   dtype='float64')
        f.create_dataset("choleskyVecs",    (choleskyN*basisN**2,),  data=twoBody.ravel(),      dtype='float64')
        f.create_dataset("choleskyBg",      (choleskyN,),            data=np.zeros(choleskyN),  dtype='float64')
