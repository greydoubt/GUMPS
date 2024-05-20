# SPDX-FileCopyrightText: 2024 Amgen
#
# SPDX-License-Identifier: BSD-3-Clause

import time

import logging
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

class Timer(object):
    "create a timer class to make timing simpler"

    def __init__(self):
        self.start: float = 0.0
        self.stop: float = 0.0

    def __enter__(self):
        "start the context manager"
        self.start = time.time()

    def __exit__(self, type, value, traceback):
        "stop the context manager"
        self.stop = time.time()
        logger.debug("elapsed time is %.2f", self.elapsed())

    def elapsed(self):
        "return the elapsed time"
        return self.stop - self.start

class Timeout(object):
    """Timeout context manager that sets a flag when timeout has expired
    This is intended to be used for graceful shutdowns."""

    def __init__(self, seconds:float):
        "set how many seconds before the termination flag is set"
        self.seconds:float = seconds
        self.termination_time:float = time.time() + seconds

    def __enter__(self):
        "start the context manager"
        return self

    def __exit__(self, type, value, traceback):
        "stop the context manager"
        logger.debug("leaving timeout expired: %s", self.timed_out)

    @property
    def timed_out(self):
        "check if the timeout has expired"
        return time.time() > self.termination_time
