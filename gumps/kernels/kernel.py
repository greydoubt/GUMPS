# SPDX-FileCopyrightText: 2024 Amgen
#
# SPDX-License-Identifier: BSD-3-Clause

from abc import ABCMeta, abstractmethod
from typing import Type, TypeVar
import attrs
import copy
import logging
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

T = TypeVar('T')

class AbstractKernel(metaclass=ABCMeta):
    "Simple abstract kernel to define the interface"
    def __init__(self, model_variables: dict|None = None) -> None:
        self.model_variables = model_variables
        self.cache: dict = {}
        self.initialize()
        logger.debug("states found %s", self.allowed_state)


    @classmethod
    def create_kernel(cls, model_variables):
        """
            Create a new kernel for the current class
            This simplifies new_kernel method for creating subclasses correctly
        """
        return cls(model_variables = model_variables)


    def f(self, variables):
        "f Function for a solver to use"
        logger.debug("creating a new state class and calling user_defined_function")
        state = self.get_state_object(variables)
        self.user_defined_function(state)
        return state


    def initialize(self):
        "Initialize objects that are expensive to calculate"
        ...


    @abstractmethod
    def get_state_class(self) -> Type:
        "return the class used to hold the state"
        ...


    @property
    def allowed_state(self):
        "return the allowed state"
        return {a.name for a in attrs.fields(self.get_state_class())}


    def get_state_object(self, variables:dict) -> T:
        "return an object create from the state class"
        return self.get_state_class()(**variables)


    @abstractmethod
    def user_defined_function(self, variables) -> None:
        "Callable that users define to add their model code"
        ...


    def new_kernel(self, data:dict):
        """
            Create a new kernel using this data
            Python 3.11: The return type for this function should be SelfType
        """
        if self.model_variables is not None:
            model_variables = copy.deepcopy(self.model_variables)
            model_variables.update(data)
        else:
            model_variables = data

        return self.create_kernel(model_variables = model_variables)

