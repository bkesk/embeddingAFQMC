import sys

#TODO: fix the import below - should only imporant the main Cholesky object
if sys.version_info[0] >= 3:
    from .Cholesky_utils_py3 import *
else:
    raise Exception("Not Implemented for Python 2 : Must use Python 3")
