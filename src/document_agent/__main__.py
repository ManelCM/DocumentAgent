# ── OpenMP thread guard ───────────────────────────────────────────────────────
# PaddleOCR v3 loads server-grade C++ models that crash (SIGSEGV) unless the
# OpenMP thread pool is capped to 1 before ANY DLL initialises.  Setting
# os.environ inside main() is too late — some DLLs latch the count at load time.
# Solution: re-exec this very process with the env var already present if it
# isn't set yet.  The sentinel _DA_OMP_FIXED prevents infinite recursion.
import os as _os, sys as _sys

if _os.environ.get("_DA_OMP_FIXED") != "1":
    import subprocess as _sp
    _env = {
        **_os.environ,
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "_DA_OMP_FIXED": "1",
    }
    _result = _sp.run(
        [_sys.executable, "-m", "document_agent"] + _sys.argv[1:],
        env=_env,
    )
    _sys.exit(_result.returncode)
# ─────────────────────────────────────────────────────────────────────────────

from .cli import main

main()
