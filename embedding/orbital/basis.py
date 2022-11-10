'''
embeddingAFQMC
author: Kyle Eskridge (bkesk)

basis.py : 

implements a 'Basis' class which is used to
organize various basis sets.
'''
import logging

from collections.abc import Iterable

import numpy as np
from pyscf import lo,gto
from pyscf.scf.hf import SCF


class Basis:
    '''
    The Basis class is a convenience class for tracking / documenting
    partions that are imposed on a particular basis. A common example
    would be the partitioning of restricted open-shell Hartree-Fock (ROHF)
    orbitals into core, doubly-occupied, singly-occupied, and virtual orbitals.
    
    The Basis.parts dictionary tracks the partitioning of the basis into
    smaller sets. The "default" partition is a holding place for any orbitals
    not assigned to a specific partition.
    '''

    def __init__(self, mf:SCF=None, C:np.array=None, S:np.array=None, ncore:int=0 ) -> None:
        
        if mf is not None:
            logging.info("Initializing basis from PySCF SCF instance. (Ignores input C,S)")
            self._init_from_mf(mf, ncore=ncore)
        else:
            try:
                assert C is not None and S is not None
                self.C = C
                self.S = S
                self.parts = {'default' : set(range(C.shape[0]))}
            except AssertionError:
                logging.error("Basis must be inialized with C and S")


    def _init_from_mf(self, mf:SCF, ncore:int=0) -> None:
        '''
        initialize from pyscf 'SCF' instance. Should work for CASCI/CASSCF
        instances as well (or anything with .mol, .mo_coeff, and .mo_occ attributes)
        '''
        
        occ=mf.mo_occ
        self.C = mf.mo_coeff
        self.S = mf.mol.intor("int1e_ovlp")

        mbasis = self.C.shape[0]

        self.parts = {'default' : set(range(mbasis))}

        self.add_part("core", idx=range(ncore))
        # stricly double-occ only
        self.add_part("double_occ", idx=(a for a in range(ncore,mbasis) if occ[a] == 2.0))
        # single and fractionally occupied
        self.add_part("single_occ", idx=(a for a in range(ncore,mbasis) if (occ[a] > 0.0 and occ[a] < 2.0)))
        # stricly virtual
        self.add_part("virtual", idx=(a for a in range(ncore,mbasis) if occ[a] == 0.0))


    def add_part(self, name:str, idx:Iterable[int]) -> None:
        '''
        add a partition to orbital basis.

        inputs:
        - name : str - a name/key for the new partition
        - idk : Iterable[int] - an iterable of integers corresponding to the orbital
                                indices which should be included in the new orbital partition.
        '''

        name = name.lower()

        try:
            assert name not in self.parts.keys()
        except AssertionError:
            logging.error(f"could not partition basis. Partition name '{name}' already exists.")
            return

        proposed_set = set(idx)
        try:
            assert proposed_set.intersection(self.parts['default']) == proposed_set
            self.parts[name] = proposed_set
            self.parts['default'] = self.parts['default'] - proposed_set
            return
        except AssertionError:
            logging.error(f"could not partition basis with {idx}. Some orbitals are already assigned to a different partition.")


    def get_part(self, name:str) -> np.array:
        try:
            orbitals = self.parts[name]
            return self.C[:,list(orbitals)]
        except KeyError:
            logging.error(f"no partition with name {name}")

    def set_part(self, name:str, new_C:np.array) -> np.array:
        try:
            orbitals = list(self.parts[name])
            assert new_C.shape[1] == len(orbitals)
            self.C[:,orbitals or slice(0,0)] = new_C
        except AssertionError:
            logging.error('input orbitals have incorrect shape')
        except KeyError:
            logging.error(f"no partition with name {name}")


def localize(mol:gto.Mole, basis:Basis) -> None:
    '''
    localize basis while respecting the partitioning
    defined in 'basis'.

    #TODO: add option to choose different localization method
    #TODO: optionally accept a single partition name and localize only that one.
    '''

    for part in basis.parts.keys():
        orbitals = basis.get_part(part)
        loc_orbitals = lo.Boys(mol, orbitals).kernel()
        basis.set_part(part, loc_orbitals)


def sort_analyze(mol:gto.Mole, basis:Basis, origin:Iterable[float]=None) -> None:
    '''
    High-level convenince function to analyze and sort a (local) basis.
    Will run for a non-local basis, but results may not be meaningful.

    Basis instance is updated in-place.
    '''
    import matplotlib.pyplot as plt
    from embedding.orbital import gen_orbital_stats, print_stats, orbital_count,sigma2_vs_dist


    if origin is None:
        # if no origin given, use first atom in geometry
        origin = mol.atom_coord(0)
        print(f"Using origin: {origin}")

    # sigma_2 vs. Distance plot
    fig1,ax1 = plt.subplots(1,1)
    ax1.set_xlabel(r'Distance from origin (Bohr)',fontsize=16)
    ax1.set_ylabel(r'$\sigma_2 \equiv \sqrt{\langle (r - \langle r \rangle)^2 \rangle}$ (Bohr)',fontsize=16)
    ax1.set_title(r'orbital size ($\sigma_2$) vs. position',fontsize=16)

    # orbital cumulative distribution plot
    fig2,ax2 = plt.subplots(1,1)
    ax2.set_xlabel(r'$R$ - distance from origin (Bohr)',fontsize=12)
    ax2.set_ylabel(r'$N$ - num. orbitals with centroic < $R$',fontsize=12)
    ax2.set_title(r"orbital count vs R",fontsize=16)

    global_ind = 0
    for part in basis.parts.keys():

        match part:
            case "default":
                continue
            case "virtual" | "virt" | "v":
                reverse = False
            case _:
                reverse = True

        orbitals = basis.get_part(part)
        
        stats = list(
            sorted(
                gen_orbital_stats(mol, orbitals,
                                  origin=origin,
                                  start_index=global_ind),
                                  reverse=reverse
                )
            )
        
        # update basis with new order
        order = [ s.index - global_ind for s in stats ]
        basis.set_part(part, orbitals[:,order])

        global_ind += orbitals.shape[1]
        
        # print stats
        print(f'\n{"":#^6} {part} orbitals {"":#^6}')
        print_stats(stats)

        # generate info plots:
        sigma2_vs_dist(part, stats, fig=fig1, ax=ax1, label=f'{part}')
        orbital_count(part, stats, fig=fig2, ax=ax2, label=f'{part}')

    ax1.legend()
    fig1.tight_layout()
    fig1.savefig("orb_dist_bars.png")

    ax2.legend()
    fig2.tight_layout()
    fig2.savefig("cumulative_orb_dist.png")
