import logging

from .infile_yaml import get_settings, runtime_cast

def show_emb_params():
    pass

def emb_params_from_input(fname):
    '''
    read embedding parameters from input file

    Input file format details:
    
    ```yaml
    embedding:
        comment: Notes on paramer choices: Ex: (Ro,Rv)=3.0,5.0 Bohr
        ncore: [int (optional) : number of core orbitals]
        nactive: [int (optional) : number of active electrons]
        E0: [float: constant energy term - usually nuclear repulsion]
        transform_only: [bool: true -> only change basis of Cholesky vectors
                               (defualt) false -> perform a modified Cholesky decomp. (mCD) in
                               active space ]
        delta: [float: mCD threshold - only used if transform_only == false]
    ```
    
    '''

    emb_params = {'ncore' : 0,
                 'nactive' : None,
                 'E0' : 0.0,
                 'tol' : 0.0,
                 'transform_only' : True}

    try:
        emb = get_settings(fname, block_name='embedding',no_log=True)

        for k in emb_params.keys():
            if k in emb.keys():
                emb_params[k] = runtime_cast(emb[k], type(emb_params[k]).__name__)
        return emb_params

    except ValueError:
        logging.info(f"no embedding block defined: treating all orbitals as active")
        return emb_params
