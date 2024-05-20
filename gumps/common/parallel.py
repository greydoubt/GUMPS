# SPDX-FileCopyrightText: 2024 Amgen
#
# SPDX-License-Identifier: BSD-3-Clause

import multiprocessing
import multiprocessing.pool
import lambda_multiprocessing
import multiprocess  #used for jupyter notebooks
import signal
from typing import Callable, Iterable, Iterator, Any
import os
from abc import ABCMeta, abstractmethod
import contextlib

import logging
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

def init_worker():
    "handle sigint"
    signal.signal(signal.SIGINT, signal.SIG_IGN)


class AbstractParallelPool(metaclass=ABCMeta):
    "Parallel pool base class"

    def __init__(self, poolsize: int | None =None):
        "initialize the pool"
        self.poolsize = poolsize
        self.pool = None

    @abstractmethod
    def start(self):
        "create the pool start"

    @abstractmethod
    def runner(self, function: Callable, sequence: Iterable) -> Iterator:
        "apply this function to this sequence"

    @abstractmethod
    def stop(self):
        "stop the pool"

    def __enter__(self):
        "handle the context manager enter event"
        self.start()
        return self.runner

    def __exit__(self, *exc):
        "handle the context manager exit event"
        self.stop()

class Parallel(AbstractParallelPool):
    "create a contextmanager for handling parallelization"

    def __init__(self, poolsize:int | None =None, ordered:bool =True, maxtasksperchild=None):
        "initialize the pool"
        super().__init__(poolsize)
        self.ordered = ordered
        self.maxtasksperchild = maxtasksperchild

    def start(self):
        "create the pool start"
        if self.poolsize is None:
            self.poolsize = os.cpu_count() or 1
        if self.poolsize > 1:
            ctx = multiprocessing.get_context('spawn')
            self.pool = ctx.Pool(self.poolsize, initializer=init_worker,
                maxtasksperchild=self.maxtasksperchild)
            logger.info("Using parallel pool with %d workers", self.poolsize)
        else:
            logger.info("Using default map and no parallelization")

    def stop(self):
        "stop the pool"
        if self.pool is not None:
            logger.debug("parallel pool closing down")
            self.pool.close()

            #wait for the pool to close gracefully (helps with coverage)
            self.pool.join()

            self.pool.terminate()

    def get_iterator(self) -> Callable:
        "return the type of iterator to use"
        if self.pool is not None:
            if self.ordered:
                logger.debug("using imap")
                return self.pool.imap
            else:
                logger.debug("using imap_unordered")
                return self.pool.imap_unordered
        else:
            logger.debug("using map function")
            return map


    def runner(self, function: Callable, sequence: Iterable) -> Iterator:
        "apply this function to this sequence"
        iter = self.get_iterator()
        return iter(function, sequence)


class ThreadParallel(AbstractParallelPool):
    "create a contextmanager for handling parallelization"

    def __init__(self, poolsize:int | None =None, ordered:bool =True):
        "initialize the pool"
        super().__init__(poolsize)
        self.ordered = ordered

    def start(self):
        "create the pool start"
        if self.poolsize is None:
            self.poolsize = os.cpu_count() or 1
        if self.poolsize > 1:
            self.pool = multiprocessing.pool.ThreadPool(self.poolsize)
            logger.info("Using parallel thread pool with %d workers", self.poolsize)
        else:
            logger.info("Using default map and no parallelization")

    def stop(self):
        "stop the pool"
        if self.pool is not None:
            logger.debug("parallel pool closing down")
            self.pool.close()

            #wait for the pool to close gracefully (helps with coverage)
            self.pool.join()

            self.pool.terminate()

    def get_iterator(self) -> Callable:
        "return the type of iterator to use"
        if self.pool is not None:
            if self.ordered:
                logger.debug("using imap")
                return self.pool.imap
            else:
                logger.debug("using imap_unordered")
                return self.pool.imap_unordered
        else:
            logger.debug("using map function")
            return map


    def runner(self, function: Callable, sequence: Iterable) -> Iterator:
        "apply this function to this sequence"
        iter = self.get_iterator()
        return iter(function, sequence)

class LambdaParallel(AbstractParallelPool):
    "create a contextmanager for handling parallelization with a lambda version of multiprocessing"

    def __init__(self, poolsize:int | None =None):
        "initialize the pool"
        super().__init__(poolsize=poolsize)
        self.stack = contextlib.ExitStack()

    def start(self):
        "create the pool start"
        if self.poolsize is None:
            self.poolsize = os.cpu_count() or 1
        if self.poolsize > 1:
            self.pool = lambda_multiprocessing.Pool(self.poolsize)
            self.stack.enter_context(self.pool)
            logger.info("Using parallel pool with %d workers", self.poolsize)
        else:
            logger.info("Using default map and no parallelization")

    def stop(self):
        "stop the pool"
        if self.pool is not None:
            logger.debug("parallel pool closing down")
            self.stack.close()

    def runner(self, function: Callable, sequence: Iterable) -> Iterator:
        "apply this function to this sequence"
        if self.pool is not None:
            logger.debug("using lambda map")
            return self.pool.map(function, sequence)
        else:
            logger.debug("using map function")
            return map(function, sequence)

class MultiprocessParallel(AbstractParallelPool):
    "create a contextmanager for handling parallelization for jupyter notebooks using multiprocess library"

    def __init__(self, poolsize:int | None =None, ordered:bool =True, maxtasksperchild=None):
        "initialize the pool"
        super().__init__(poolsize)
        self.ordered = ordered
        self.maxtasksperchild = maxtasksperchild

    def start(self):
        "create the pool start"
        if self.poolsize is None:
            self.poolsize = os.cpu_count() or 1
        if self.poolsize > 1:
            ctx = multiprocess.get_context('spawn')
            self.pool = ctx.Pool(self.poolsize, initializer=init_worker,
                maxtasksperchild=self.maxtasksperchild)
            logger.info("Using parallel pool with %d workers", self.poolsize)
        else:
            logger.info("Using default map and no parallelization")

    def stop(self):
        "stop the pool"
        if self.pool is not None:
            logger.debug("parallel pool closing down")
            self.pool.close()

            #wait for the pool to close gracefully (helps with coverage)
            self.pool.join()

            self.pool.terminate()

    def get_iterator(self) -> Callable:
        "return the type of iterator to use"
        if self.pool is not None:
            if self.ordered:
                logger.debug("using imap")
                return self.pool.imap
            else:
                logger.debug("using imap_unordered")
                return self.pool.imap_unordered
        else:
            logger.debug("using map function")
            return map


    def runner(self, function: Callable, sequence: Iterable) -> Iterator:
        "apply this function to this sequence"
        iter = self.get_iterator()
        return iter(function, sequence)
