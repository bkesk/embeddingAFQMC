# Embedding Tools for AFQMC

![Build and Test](https://github.com/bkesk/embeddingAFQMC/actions/workflows/python-app.yml/badge.svg)

A basic Python package for building orbitally-based embedding
Hamiltonians in 2nd quantized form.
These tools were developed for local embedding AFQMC [B. Eskridge, H. Krakauer, S. Zhang *J. Chem. Theory Comput.* 2019, 15, 7, 3949–3959](https://pubs.acs.org/doi/10.1021/acs.jctc.8b01244),
but can be used for frozen orbital embedding in general.
The tools included are designed for AFQMC calculations, but the
tools are applicable to other methods as well.

## Dependencies

- PySCF : used mostly for its interface to libcint.
- numpy : general matrix/tensor operations
- h5py : file i/o

## TODO:

- add basic use instructions
- add a simple tutorial
- reorganize with:
   - `/`
      - `embedding/`
      - `tests/`
      - etc.
- add setup.py : improve installability
- add test for correct ordering of orbitals
- add a Dockerfile
