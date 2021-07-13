import sys

if sys.version_info[0] >= 3:
    from .Cholesky_utils_py3 import *
    from .simple_cholesky import cholesky
else:
    raise Exception("Not Implemented for Python 2 : Must use Python 3")
