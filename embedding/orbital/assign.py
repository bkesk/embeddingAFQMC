'''
embeddingAFQMC
author: Kyle Eskridge (bkesk)

assign.py : 

provides functionality to assist in analyzing local orbitals /
assigning them to the active space.

#TODO:
- compute fourth central moment: mu_4 = <(r - <r>)^4>
'''

from collections.abc import Generator, Iterable
from dataclasses import dataclass

import numpy as np

from pyscf import gto

@dataclass(order=True)
class OrbitalStat:
    '''
    Dataclass to hold statistics on specific orbitals.

    stats include:

    - orbital index (for reference)
    - orbital centroid position.
    - distance from origin
    - second central moment: mu_2 = <(r - <r>)^2>
    '''

    distance: float

    position: tuple[float,float,float]
    index: int
    second_moment: float


def euclid_nd(r1,r2):
    '''
    return the n-dimensional Euclidean distance between r1 and r2
    '''
    return np.sqrt(sum([ (x1 - x2)**2 for x1,x2 in zip(r1,r2,strict=True)]))


def gen_orbital_stats(mol: gto.Mole,
                      orbitals : np.array,
                      start_index : int = 0,
                      metric : callable = euclid_nd,
                      origin : tuple[float,float,float] = (0.0,0.0,0.0)) -> Generator[OrbitalStat] :
    '''
    Generator function to build orbital statistics.

    Inputs:
    - mol : PySCF Mole instance
    - orbitals : np.array -> convention columns are interpreted as orbitals
                          -> may be a subset of the full set of orbitals (ex:
                                                    occupied orbitals only)
    - start_index : starting index for *labelling* orbitals
    '''

    M,N = orbitals.shape
    orbitals_dagger = orbitals.conj().T

    rsq = mol.intor('int1e_r2')
    r = mol.intor('int1e_r')

    # < \vec{r} > : TODO: THIS IS [Cx, Cy, Cz] - want [ (x1,y1,z1), (x2,y2,z2), ... ]
    centroids = [
        np.matmul(orbitals_dagger,
        np.matmul(r[x], orbitals))
        for x in range(3)
        ]

    # < r^2 >
    expect_rsq = np.matmul(orbitals_dagger, np.matmul(rsq, orbitals))
  
    for i in range(N):
        # (see TODO above) : this is not clear!
        position = [ c[i,i] for c in centroids ]

        yield OrbitalStat(
            index=i + start_index,
            distance=metric(position,origin),
            position=position,
            second_moment=expect_rsq[i,i] - np.dot(position,position)
            )


def print_stats(orb_stats : Iterable[OrbitalStat] = None):
    '''
    write orbital statistics
    '''
    print(f'{"i":>5} {"mu_2":>12} {"sigma_2":>12} {"Centroid (x,y,z)":^43} {"Distance":<12}')
    for orb in orb_stats:
        print(f"{orb.index:>5} {orb.second_moment:>12f} {np.sqrt(orb.second_moment):>12f} ", end="")
        print(f"({orb.position[0]:>+8.6e},{orb.position[1]:>+8.6e},{orb.position[2]:>+8.6e}) {orb.distance:>12f}")


def orbital_count(name : str, orb_stats : Iterable[OrbitalStat], bins : int = 100, display_plot=False, save_data=True):
    '''
    writes a file called 'name' which contains the cumulative distribution of local orbitals
    as a function of the 'distance' attribute of the orb_stats.
    '''

    import matplotlib.pyplot as plt

    distances = [ orb_stat.distance for orb_stat in orb_stats ]

    _hist, binLabels = np.histogram(distances, bins=bins, range=(-0.001,np.amax(distances)), density=False)
    count = np.cumsum(_hist)

    fig,ax = plt.subplots(1,1)

    ax.plot(binLabels[1:], count)
    ax.set_title(f"{name} Orbital count vs R")
    
    if display_plot:
        plt.show()
    else:
        plt.savefig(name + "_cdf.png")

    if save_data:
        with open(name + '.dat', 'w') as f:
            for bl,c in zip(binLabels[1:], count, strict=True):
                f.write(f"{bl:>8.6f} {c:>6d}\n")
    
    return count