# This file is intentionally left minimal.
# The native extension module (built by maturin from the Rust cdylib)
# will be injected into this package at install time.
try:
    from .nextpnr import *  # noqa: F401,F403
except ImportError:
    # Pure-Python helpers under nextpnr/ should remain importable even when the
    # native extension has not been built in the current environment.
    pass
