"""Handling importing mpi4py

This package should be imported by __main__ only. This prevents
multiprocessing.Process from importing mpi4py. It provides the MPI
communication. Should the import mpi4py fail, provide a dummy COMM.
"""

import logging


class DummyComm:
    def __init__(self):
        pass
    def Get_size(self):
        return 1
    def Get_rank(self):
        return 0
    def gather(self, result, *args, **kwargs):
        return [result]


try:
    from mpi4py import MPI
    COMM = MPI.COMM_WORLD
except ImportError as err:
    logging.warning("ImportError when import mpi4py, assume rank 0 size 1")
    logging.warning(str(err))
    COMM = DummyComm()
