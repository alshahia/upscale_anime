"""Shared pytest fixtures.

Tkinter only allows one root per process. Several test files need a
live Tk root; instead of each file creating its own (which causes
"Can't find a usable init.tcl" errors after the first destroy on
Windows), we share one session-scoped root here.
"""
import tkinter as tk

import pytest


@pytest.fixture(scope="session")
def shared_tk_root():
    """One Tk root for the whole test session.

    Tests that need a parent widget should use this fixture (or chain
    off it). It's never destroyed; pytest reclaims the process.
    """
    r = tk.Tk()
    r.withdraw()
    yield r
    # Don't destroy on teardown — subsequent tests in the same process
    # can't create a new Tk root on Windows after one is destroyed.
