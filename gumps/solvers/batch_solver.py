# SPDX-FileCopyrightText: 2024 Amgen
#
# SPDX-License-Identifier: BSD-3-Clause

from abc import ABCMeta, abstractmethod
import numpy as np

import logging
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

class AbstractBatchSolver(metaclass=ABCMeta):
    "this is an abstract solver class"

    @abstractmethod
    def has_next(self) -> bool:
        "return True if another step is needed"
        pass

    @abstractmethod
    def ask(self) -> np.ndarray:
        "return the population to be evaluated"
        pass

    @abstractmethod
    def tell(self, loss: np.ndarray):
        "provide the loss"
        pass