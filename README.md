# Embedding Tools for AFQMC

![Build and Test](https://github.com/bkesk/embeddingAFQMC/actions/workflows/python-app.yml/badge.svg)

A basic Python package for building orbitally-based embedding
Hamiltonians in 2nd quantized form.
These tools were developed for local embedding AFQMC [B. Eskridge, H. Krakauer, S. Zhang *J. Chem. Theory Comput.* 2019, 15, 7, 3949–3959](https://pubs.acs.org/doi/10.1021/acs.jctc.8b01244),
but can be used for general frozen orbital transformations.
The tools included are designed for AFQMC calculations, but are applicable to other methods as well.

## Install:

The embedding AFQMC package can be installed using:

```bash
git clone https://github.com/bkesk/embeddingAFQMC.git
cd embeddingAFQMC
python -m venv venv && . venv/bin/activate
pip install -e .
```

the embedding AFQMC package provides a convenience [command-line interface (CLI)](#command-line-interface), and
can be used directly [in Python](#in-python)

## Usage:

### command-line interface

The embedding package provides a cli to the high-level `make_embedding_H` function that can be run with the `embed` command.
Help is available via the cli with `$ embed --help`. Alternatively, the following command provides similar functionality: `$ python -m embedding --help`.

Using the cli requires a yaml-based input file. An example for an oxygen atom is provided below.

```yaml
geom:
  comment: an oxygen atom
  atoms: |
    O 0.0 0.0 0.0
basis:
  O:
    pyscf_lib: True
    data: ccpvdz
molecule:
  spin: 2
  charge: 0
orbital_basis: 'ROHF.chk:/scf/mo_coeff'
embedding:
  nfc: 0
  nactive:
  E0: 0.0
  tol: 1.0E-5
```

In this example, molecular orbitals from a PySCF checkpoint file are used as a basis.

### In Python

The primary interface to the embedding / frozen orbital transformation is the `make_embedding_H` function.
Here is a minimal example of using this function to freeze the 1s electrons in an Oxygen atom:

```Python
from pyscf import gto,scf

from embedding import make_embedding_H

from embedding.cholesky import cholesky
from embedding.cholesky.integrals.gto import GTOIntegralGenerator

# perform an SCF calculation for reference wavefunction
mol = gto.M(atom='''O 0.0 0.0 0.0''',
            basis='ccpvdz',
            spin=2)

mf = scf.ROHF(mol)
mf.kernel()

# use canonical orbital basis
basis = mf.mo_coeff

# get GTO-basis integrals
S = mol.intor('int1e_ovlp')
K = mol.intor('int1e_kin') + mol.intor('int1e_nuc')

# compute Cholesky vectors in GTO basis
gto_gen = GTOIntegralGenerator(mol)
numcholesky,choleskyAO = cholesky(gto_gen,tol=1.0E-6)

# create frozen orbital Hamiltonian
twoBody,numCholeskyActive,oneBody,S,E0 = make_embedding_H(ncore=1,
                                                          C=basis,
                                                          twoBody=choleskyAO,
                                                          oneBody=K,
                                                          S=S,
                                                          transform_only=True)
```

## Legacy version

Recently, the embeddingAFQMC Library was reorganized, and some redundant and/or 
deprecated code has been romved.
A legacy version of the embedding exists in the `legacy` branch for code that still uses it.
Some "helper" tools from `legacy` may be updated and pushed back into the main branch.

## TODO:

- General
  - change print to logging.info (or similar)
  - define and configure logger
  - add basic input file description
  - add a Dockerfile

- make_embedding_H:
  - no need to return number of Cholesky vectors

- cholesky:
  - add ability to use comlpex-valued orbitals

- `legacy` branch:
  - update tools for analyzing local orbitals for `main` branch
