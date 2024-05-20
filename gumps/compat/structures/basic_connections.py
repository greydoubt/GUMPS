# SPDX-FileCopyrightText: 2024 Amgen
#
# SPDX-License-Identifier: BSD-3-Clause

#TODO
#Fix type check here, discussion on interface on variable pairs
#discuss when variable pairs should be used and how they play into
#definition of problems

from gumps.compat.abstractclasses import AbstractBidirectionalTransform
import numpy as np

class UnityTransform(AbstractBidirectionalTransform):
    """
    The trivial transform - data is passed without being changed.
    """
    #TODO: this is probably pretty inefficient, and certainly reuses too much
    #code. Refine.
    def forward_transform(source, target):
        c_target, sym_target = target
        c_source, sym_source = source
        vr_target = c_target.get_variables()
        vr_source = c_source.get_variables()
        try:
            vr_target[sym_target] = vr_source[sym_source]
        except KeyError:
            raise KeyError(f'transforming source {sym_source} in ' \
                f'{vr_source._parent.id} to target {sym_target} in ' \
                f'{vr_target._parent.id}')


    def backward_transform(target, source):
        c_target, sym_target = target
        c_source, sym_source = source
        vr_target = c_target.get_variables()
        vr_source = c_source.get_variables()
        vr_source[sym_source] = vr_target[sym_target]


class VariablePair:
    """
    An object describing the transform between two variables.

    Attributes

    -----------
    _source : Container  
        Container that holds the original variable before transformation
    _target : Container
        Container that holds the variable post transformation
    _transform : Transform
        The particular transform to be used to modify the variables.

    """

    def __init__(self, source, target,
                 transform = UnityTransform):
        self._source = source
        self._target = target
        self._transform = transform

    # @property
    # def source(self): return self._source

    # @property
    # def target(self): return self._target

    # @property
    # def transform(self): return self._transform
    
    def apply_forward(self):
        """
        Apply transform from source structure to target structure.

        Returns :
        None
        """
        # if self._source[0].id == 'ODE IV solver':
        #     print(self._source[0].get_variables())
        self._transform.forward_transform(self._source, self._target)

    def apply_backward(self):
        """
        Apply backward transform from target structure to source structure.
        Returns :
        None
        """
        self._transform.backward_transform(self._target, self._source)
    


class Connection:
    """
    A connection between two models, which defines their data dependencies
    (and therefore implicitly the order in which they're solved) and
    transforms between shared data. Provides methods for setting source and
    target models and defining a data transform. Assumes a unidirectional
    connection. 

    Aggregates multiple variable pairs.

    Attributes

    -----------
    _pairs : List of VariablePairs
    _source : Container  
        Container that holds the original variable before transformation
    _target : Transform
        The particular transform to be used to modify the variables.
    """

    def __init__(self, source, target,
                 bidirectional = False,
                 pairs = None):
        self._pairs = pairs if pairs is not None else []
        self._bidirectional = bidirectional
        self._source = source
        self._target = target

    def add_variable_pair(self, source, target,
                          transform = UnityTransform):
        #TODO: overload this for addition of pair by passing a VariablePair
        #or references to source, target (and transform)
        self._pairs.append(VariablePair(source, target, transform))

    def get_all_vp_targets(self):
        vp_targets = []
        for pair in self._pairs:
            if type(pair._target) != tuple:
                vp_targets.extend(pair._target)
            else:
                vp_targets.append(pair._target)
        return vp_targets
    @property
    def variable_pairs(self): return self._pairs
    
    @property
    def source(self): return self._source
    @source.setter
    def source(self, model): self._source = model

    @property
    def target(self): return self._target
    @target.setter
    def target(self, model): self._target = model

    @property
    def transform(self): return self._transform
    @transform.setter
    def transform(self, transform): self._transform = transform

    @property
    def bidirectional(self): return self._bidirectional


class BidirectionalConnection(Connection):
    """
    A bidirectional connection
    """
    def __init__(self, source, target):
        super().__init__(self, source, target,
                         bidirectional = True)