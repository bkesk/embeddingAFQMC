import argparse

def get_cli_args() -> argparse.Namespace:
    '''
    Get high-level cli arguments.

    Design notes:

    1. The cli is meant for "forgetable" settings such as the *name* of
        input files, logging level, etc. This is distinct from settings
        that specify what the embedding tool should do like the source 
        of the orbital basis, number of core orbitals, etc.
    '''

    parser = argparse.ArgumentParser(description='Generate an embedding Hamiltonian')
    
    parser.add_argument('--input', '-i', metavar='input file', type=str,
                        action='store',
                        dest='input_file',
                        default='input.yaml',
                        help="name of input file (YAML format)")

    parser.add_argument('--outcode', '-o', metavar='output code', type=str,
                        action='store',
                        dest='outcode',
                        default='afqmclab',
                        help='output format (default: AFQMCLab format)')

    parser.add_argument('--log-level','-ll', 
                        choices=['error', 'warning','info','debug'],
                        default='warning',
                        dest='log_level',
                        help='logging level for Python. order of increasing verbosity : error -> warning -> info -> debug')

    args = parser.parse_args()
    return args
