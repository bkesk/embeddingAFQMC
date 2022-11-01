from embedding.cli import get_cli_args
from embedding import make_embedding_H, get_one_body, get_two_body
from embedding.io.infile import get_settings, mol_from_input, orbitals_from_input, emb_params_from_input

def _make_embedding_H(input_fname, outcode='afqmclab'):

    # TODO: check if reading from input is succesfull : end gracefully if not

    mol = mol_from_input(input_fname)
    basis_orbitals = orbitals_from_input(input_fname)
    emb_params = emb_params_from_input(input_fname)
    
    S, oneBody = get_one_body(mol)
    twoBody = get_two_body(mol)

    H2,_,H1,S,E0 = make_embedding_H(C=basis_orbitals,oneBody=oneBody,S=S,twoBody=twoBody,**emb_params)

    match outcode:
        case "afqmclab":
            print("Saving Hamiltonian in AFQMCLab format")
            _save_afqmclab(mol.nelec,H2,H1,E0)
        case "gafqmc":
            print("Saving Hamiltonian in GAFQMC format")
            _save_gafqmc(H2,H1,E0)
        case "raw_hdf5":
            print("Saving Hamiltonian in HDF5 format")
            _save_hdf5(H2,H1,S,E0)
        case _:
            print(f"No interface for {outcode} : saving raw matrix elements in hdf5")
            _save_hdf5(H2,H1,S,E0)


def _save_afqmclab(nelec,H2,H1,E0=0.0,model_name='model_param'):
    from embedding.io.afqmclab import writeModel
    writeModel(nelec,H1,H2)

def _save_hdf5(H2,H1,E0=0.0):
    raise NotImplementedError


def _save_gafqmc(H2,H1,E0=0.0):
    raise NotImplementedError


def main():
    '''
    Entry point function for embedding AFQMC.
    '''

    args = get_cli_args()

    # read inputs
    inputs = get_settings(args.input_file)
    
    # build AFQMC
    if inputs is not None:
        _make_embedding_H(args.input_file, args.outcode)


main()
