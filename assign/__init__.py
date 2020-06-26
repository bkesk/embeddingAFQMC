import sys

#TODO: fix the import below - should only imporant the main Cholesky object
if sys.version_info[0] >= 3:
    from .orbital_assign_utils_py3 import *
elif sys.version_info[0] < 3:
    from .orbital_assign_utils import *
