# SPDX-FileCopyrightText: 2024 Amgen
#
# SPDX-License-Identifier: BSD-3-Clause

from bisect import bisect
import sys
import contextlib
import pkgutil
from importlib import import_module

def stepwise_input(u_list, tx_vector, t):
    U = []
    for u in u_list:
        tx_index = max(0, min(len(u), bisect(tx_vector, t)-1))
        U.append(u[tx_index])
    return U

#Is this being used somewhere?
def interp_input(u_list, tx_vector, t):
    U = []
    for u in u_list:
        tx_index = max(0, min(len(u)-1, bisect(tx_vector, t)))
        tx_index_last = max(0, tx_index-1)
        t0 = tx_vector[tx_index_last]
        t1 = tx_vector[tx_index]
        u0 = u[tx_index_last]
        u1 = u[tx_index]
        if t1 == t0:
            U.append(u[tx_index])
        else:
            U.append(u0 + (u1-u0)*(t-t0)/(t1-t0))
    return U

class InfeasibleError(Exception):
    pass

class IntegrationError(Exception):
    pass

class DummyFile(object):
    def write(self, x): pass

@contextlib.contextmanager
def nostdout():
    save_stdout = sys.stdout
    sys.stdout = DummyFile()
    yield
    sys.stdout = save_stdout

def load_module(pkgName, pkg):
    """Walks through packages in module and loads the submodule
    with the name provided, raises an error if the module is not found.
    """
    chosen_module = None
    for loader, module_name, _ in  pkgutil.walk_packages(pkg.__path__):
        if pkgName == module_name:
            full_name = pkg.__name__ + '.' + module_name
            chosen_module = import_module(full_name)
    if chosen_module is None:
        raise AttributeError("Module Not Found")
    return chosen_module