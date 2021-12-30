import numpy as np

from pyscf import gto

def compute_sigma2(mol, C_in, start_ind=0):
    '''
    Computes the second central moment orbital spread (sigma2) for each orbital in C
    sigma2 = sqrt(mu_2) with
    mu_2 = <(r - <r>)^2>
    
    STATUS:
    - we are computing centroids that agree with ERKALE, but mu_2 / sigma_2 do not agree
       - this confirms that we have the correct convention for C as well
       - also confirms that we are working in Bohr!
    - IMPORTANT! C must be a np.matrix and not a np.array in the current implementation! 
                 This has to do with the arithmetic operator's defition of * which is different
                 for np.array, and np.matrix
      - can use einsum, or np.matul to allow arrays too
      - if that fails, add a type check
    '''
    C = np.matrix(C_in)
    M,N = C.shape
    
    C_dag = C.conj().T

    rsq = mol.intor('int1e_r2')
    r = mol.intor('int1e_r')

    sigma_sum=0.0
    mu_sum=0.0

    orbitals = []

    print(f'\n\n{"i":>5} {"mu_2":>12} {"sigma_2":>12} {"Centroid (x,y,z)":>26}')
    for i in range(N):
        # compute <r>
        centroid = np.zeros(3)
        for a in range(3):
            centroid[a] = C_dag[i,:]*r[a]*C[:,i]
        # compute <r^2> - <r>^2
        mu_2 = (C_dag[i,:]*rsq*C[:,i] - np.dot(centroid,centroid))[0,0]
        
        sigma_2 = np.sqrt(mu_2)
        print(f'{i+start_ind:>5} {mu_2:>12.8e} {sigma_2:>12.8e} ({centroid[0]:>8.6e},{centroid[1]:>8.6e},{centroid[2]:>8.6e})')
        sigma_sum+=sigma_2
        mu_sum+=mu_2
        
        orbitals.append((i+start_ind,mu_2,sigma_2,(centroid[0],centroid[1],centroid[2])))
        
    print(f'sum of sigma_2 over all orbitals is : {sigma_sum}')
    print(f' [FB cost function] sum of mu_2 over all orbitals is : {mu_sum}')

    return orbitals

def euclid3d(r1,r2):
    dist_sqr = sum([ (x1 - x2)**2 for x1,x2 in zip(r1,r2)])
    return np.sqrt(dist_sqr)

def order_orbitals(orbitals, metric=None, origin=(0.0,0.0,0.0), outname='sigma_vs_centroidDist'):
    
    if metric is None:
        metric = euclid3d

    sigma2 = []
    for orb in orbitals:
        centroid = orb[-1]
        sigma2.append([orb[0], orb[1], metric(origin, centroid)])

    # sort
    sigma2.sort(key=lambda orbital : orbital[2])

    # write
    f2 = open(outname+'.sigma2.dat','w')
    for entry in sigma2:
        f2.write(f'{entry[0]} {entry[1]} {entry[2]}\n')
    f2.close()

    # return the reordered orbital indices:
    return [ old_ind for old_ind,_,_ in sigma2 ]


def show_cumulative(fname):

    import matplotlib.pyplot as plt

    _,sigma2,r  = np.loadtext(fname,unpack=True)

    _hist, binLabels = np.histogram(metVals, bins=bins, range=(-0.001,np.amax(metVals)), density=False)
    count = np.cumsum(_hist)

    plt.plot(binLabels[1:], count)
    plt.title("Orbital count vs R")
    plt.show()




