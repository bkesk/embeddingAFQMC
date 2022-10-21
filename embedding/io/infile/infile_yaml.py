import logging
import importlib

from yaml import safe_load

from pyscf import gto

def runtime_cast(value, _type):

    if value is None:
        logging.debug(f"attempted to cast None to {_type}. This could be a mistake.")
        return value

    try:
        mod = importlib.import_module('builtins')
        return getattr(mod, _type)(value)
    except AttributeError:
        logging.error(f"No known conversion to {_type} : only builtin types are supported")
        
def get_settings(fname : str, block_name: str = None, no_log : bool = False) -> dict:
    '''
    Get settings dictionary from input file (YAML)

    Inputs:
        - fname (str) : name of input file (yaml format)
        - (optional) block_name (str) : name of a specific block with input file
    '''

    try:
        with open(fname, 'r') as f:
            input_file = safe_load(f)

        if block_name is None:
            return input_file
        
        try:
            return input_file[block_name]
        except KeyError:
            if not no_log: logging.error(f"couldn't read block \"{block_name}\" from {fname}")
            raise ValueError

    except FileNotFoundError:
        logging.error(f"Input file {fname} not found.")


def mol_from_input(fname) -> gto.Mole:
    '''
    generate PySCF Mole instance based on input file.
    '''
    
    mol_params = _read_mol_params(fname)
    ecp, basis = _read_ecp_basis(fname)
    geom = _read_geom(fname)

    if geom is None:
        logging.error("Couldn't build Mole from input file: {fname}\n"
                      "No geometry definied!")
        return None

    return gto.M(atom=geom, basis=basis, ecp=ecp, parse_arg=False, **mol_params)

def _read_ecp_basis(fname):
    '''
    read ecp and basis in NWChem format from yaml file
    Format details for a system with atomic species 'atom1', 'atom2', ... :
    
    ```yaml
    basis:
      [atom1]:
        pyscf_lib: True # basis included in pyscf's library? True/False
        data: ccpvdz #pyscf basis name!
      [atom2]:
        pyscf_lib: False
        data: |
          [atom2] S
              exp1     c1
              exp2     c2
              ... basis set data in NWChem format ...
      ... entry for each atom in system (each must have an entry here)
    ecp:  # optional section.
      [atom1]: |
        [atom1] nelec [n1]
        ... ecp data ...
        ... ecp data ...
      [atom2]: |
        [atom2] nelec [n2]
        ... ecp data ...
        ... ecp data ...
      ... entry for each atom with ecp
    ```
    '''

    def _process_basis(input_basis):
        '''
        process basis into format that PySCF can directly use.
        '''
        pyscf_basis = {}
        for key in input_basis:
            if input_basis[key]['pyscf_lib'] == False:
                logging.debug(f'parsing basis data for {key}')
                parsed_basis = gto.parse(input_basis[key]['data'])
                pyscf_basis[key] = parsed_basis
            else:
                logging.debug(f'use pyscf library basis {input_basis[key]["data"]} for {key}')
                pyscf_basis[key] = input_basis[key]['data']
        return pyscf_basis

    with open(fname,'r') as f:
        ecp_basis = safe_load(f) # limits load to simple python objects, just what we want!
    
    logging.debug(f'ecp_basis = {ecp_basis}')
    
    try:
        ecp = get_settings(fname, block_name='ecp')
    except ValueError:
        ecp = None
    
    basis = _process_basis(get_settings(fname, block_name='basis'))

    logging.debug(f'ecp = {ecp}')
    logging.debug(f'basis = {basis}')

    return ecp, basis

def _read_geom(fname):
    '''
    read geometry in xyz format from yaml file
    Format details for a system with atomic species 'atom1', 'atom2', ... :
    
    ```yaml
    geom:
      comment: a description of the geometry
      atoms: |
        [atom1] [x coord.] [y coord.] [z coord.]
        [atom2] [x coord.] [y coord.] [z coord.]
        ... entry for each atom in system
    ```
    '''

    geom = get_settings(fname, block_name="geom")
    logging.debug(f' geom = {geom} ')

    if "comment" in geom.keys():
        logging.info(f'geometry comment: {geom["comment"]}')
    
    try:
        assert "atoms" in geom.keys()
        return geom["atoms"]
    except AssertionError:
        logging.error("No 'atoms' block within 'geom' block")
        return None



def _read_mol_params(fname) -> dict:
    '''
    read molecule parameters from yaml file.
    All parameters are optional!
    Format details:
    ```yaml
    molecule:
      spin: [int Nup - Ndn]
      charge: [int total charge]
      symmetry: [str for desired symmetry]
    ```
    note: a value of None will be interpreted as 0 for spin, charge,
         bit
    '''

    try:
        mol_params = get_settings(fname, block_name='molecule')
        logging.debug(f' mol_params = {mol_params} ')
        return mol_params
    except KeyError as e:
        print(f'\n [+] no molecule parameters found in input file : attempting to proceed with PySCF defaults')
        logging.debug(f'could not read molecule parameters: {e}')
        return None
