"""
Run every test process with single-threaded numerical libraries, and
matplotlib on the Agg backend.

`pytest -n auto` starts one worker per core, and each worker's numpy
(OpenBLAS / MKL / OpenMP) and numba start their own pool of one thread per
core. On Windows each of those threads commits its buffers up front, so
cores x cores threads exhaust the commit limit and a worker dies with
"Windows fatal exception: code 0x8007000e" (E_OUTOFMEMORY) in whichever
test it happens to be running. The tests are too small to gain from BLAS
threads anyway.

The entry points plot to files and pick the Agg backend themselves
(main.py: matplotlib.use("Agg")), but a test process must not depend on
that line staying there -- a GUI backend on a headless runner is an import
error, and on Windows a window per figure.

This must run before numpy is imported, in both runners: this is the
`tests` package's __init__, so pytest executes it before `tests/conftest`
or any test module (in place of a conftest.py), and `python -m unittest
discover -s tests -t .` imports the package before its modules (without
`-t .` unittest takes tests/ as the top level and never imports it). The
multiprocessing workers of the pool tests inherit the environment.
setdefault leaves an explicit setting alone.
"""

import os

for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
             "NUMEXPR_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(_var, "1")
os.environ.setdefault("MPLBACKEND", "Agg")
