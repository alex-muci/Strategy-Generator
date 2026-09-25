"""
Run every test process with single-threaded numerical libraries.

`pytest -n auto` starts one worker per core, and each worker's numpy
(OpenBLAS / MKL / OpenMP) and numba start their own pool of one thread per
core. On Windows each of those threads commits its buffers up front, so
cores x cores threads exhaust the commit limit and a worker dies with
"Windows fatal exception: code 0x8007000e" (E_OUTOFMEMORY) in whichever
test it happens to be running. The tests are too small to gain from BLAS
threads anyway.

This must run before numpy is imported: pytest loads conftest.py before the
test modules, and the multiprocessing workers of the pool tests inherit the
environment. setdefault leaves an explicit setting alone.
"""

import os

for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
             "NUMEXPR_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(_var, "1")
