import sys

#TODO: fix the import below - can do this more elegantly
if sys.version_info[0] >= 3:
    from .density_matrix import *
else:
    raise Exception("Not Implemented for Python 2 : Must use Python 3")
