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
                      indices : list = None,
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
    - indices : list of indices to use (start_index is ignored if indices is not None!)
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
  
    if indices is not None:
        get_index = lambda x: x[i]
    else:
        get_index = lambda x: x + start_index

    for i in range(N):
        # (see TODO above) : this is not clear!
        position = [ c[i,i] for c in centroids ]

        yield OrbitalStat(
            index=get_index(i),
            distance=metric(position,origin),
            position=position,
            second_moment=expect_rsq[i,i] - np.dot(position,position)
            )


def print_stats(orb_stats : Iterable[OrbitalStat] = None):
    '''
    write orbital statistics
    '''
    print(f'{"i":>5} {"mu_2":>12} {"sigma_2":>12} {"Centroid (x,y,z)":^43} {"Distance":<12}')
    print(f'{"":_^100}')
    for orb in orb_stats:
        print(f"{orb.index:>5} {orb.second_moment:>12f} {np.sqrt(orb.second_moment):>12f} ", end="")
        print(f"({orb.position[0]:>+8.6e},{orb.position[1]:>+8.6e},{orb.position[2]:>+8.6e}) {orb.distance:>12f}")


def active_from_radii(R:float, bins:np.array, count:np.array):
    '''
    get number of active orbitals from localization radius.
    '''
    for i,b in enumerate(bins[:-1]):
        if b > R:
            return[count[i]]

def orbital_count(orb_stats:Iterable[OrbitalStat], name:str='orbitals', R:float=None, bins:int=100, display_plot=False, save_fig=False, save_data=True, fig=None, ax=None, **kwargs):
    '''
    Computes the cummulative distribtion of orbitals vs. centroid position.
    Returns the number of orbitals with centroid less than R (if given)
    or the total number of orbitals.

    Optionally, can:
    - write cummulative orbital distribution to a file
    - plot cummulative orbital distribution in a new plot / add to existing
    '''

    import matplotlib.pyplot as plt

    distances = [ orb_stat.distance for orb_stat in orb_stats ]

    _hist, binLabels = np.histogram(distances, bins=bins, range=(-0.001,np.amax(distances)), density=False)
    count = np.cumsum(_hist)

    #import pdb
    #pdb.set_trace()

    if R is not None:
        Na = active_from_radii(R,binLabels,count)
    else:
        Na = count[-1]

    if fig is None or ax is None:
        fig,ax = plt.subplots(1,1)
        ax.set_title(f"{name} orbital count vs R")
    
    ax.plot(binLabels[1:], count, **kwargs)

    if display_plot:
        plt.show()

    if save_fig:
        plt.savefig(name + "_cdf.png")

    if save_data:
        with open(name + '.dat', 'w') as f:
            for bl,c in zip(binLabels[1:], count, strict=True):
                f.write(f"{bl:>8.6f} {c:>6d}\n")
    
    return Na

def sigma2_vs_dist(orb_stats : Iterable[OrbitalStat], name : str, display_plot=False, save_fig=False, save_data=True, fig=None, ax=None, **kwargs):
    '''
    produce sigma2 vs orbital distance, and generate corresponding plot.
    '''
    
    import matplotlib.pyplot as plt

    sigma2 = [ (orb_stat.distance, np.sqrt(orb_stat.second_moment)) for orb_stat in sorted(orb_stats)]
    
    if save_data:
        with open(f"{name}_sigma2.dat","w") as f:
            f.writelines(f"{d} {s2}\n" for d,s2 in sigma2)

    def add_bar(ax, x, y, **kwargs):
        ax.bar(x,y,width=0.5,
                edgecolor='black',
                alpha=0.5,
                hatch='...',
                **kwargs)
    
    if fig is None or ax is None:
        fig, ax = plt.subplots(1,1)
    
        ax.set_xlabel(r'Distance from origin (Bohr)',fontsize=16)
        ax.set_ylabel(r'$\sigma_2 \equiv \sqrt{\langle (r - \langle r \rangle)^2 \rangle}$ (Bohr)',fontsize=16)
        ax.set_title(f'{name} orbital size ($\sigma_2$) vs. position')

        fig.tight_layout()

    add_bar(ax, [d for d,_ in sigma2],
                [s for _,s in sigma2],
                **kwargs)

    if display_plot:
        plt.show()

    if save_fig:
        plt.savefig(name + "_orbital_bars.png")

    return sigma2
