# This file is intentionally left minimal.
# The native extension module (built by maturin from the Rust cdylib)
# will be injected into this package at install time.
from .nextpnr import *  # noqa: F401,F403
