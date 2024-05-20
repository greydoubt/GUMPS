# SPDX-FileCopyrightText: 2024 Amgen
#
# SPDX-License-Identifier: BSD-3-Clause

import hashlib
import json
from pathlib import Path
from typing import Optional

import attr
import numpy
import pandas as pd


def convert_dict(seq: Optional[dict[str, float]]) -> Optional[pd.Series]:
    """convert a dictionary type object to a pandas series"""
    if seq is not None:
        #make sure an item is a dict like object
        dict(seq)

        return pd.Series(seq, dtype=numpy.float64)
    return None

@attr.s
class Loss:
    """Loss class which stores data as a pandas series"""
    _values: Optional[pd.Series] = attr.ib(converter=convert_dict)

    @property
    def values(self):
        "return _values"
        return self._values

    @values.setter
    def values(self, value: dict[str, float]):
        "set _values"
        self._values = convert_dict(value)

    @values.deleter
    def values(self):
        "delete values"
        self._values = None

    @property
    def valid(self):
        "return if the fitness is valid"
        return self._values is not None and len(self._values)

@attr.s
class Individual:
    """Individual dataclass that represents a single individual, its transformed space, and its fitness,"""
    parameters:pd.Series = attr.ib(converter=convert_dict)
    loss:Loss = attr.ib(converter=Loss, default=attr.Factory(dict))
    transformed_parameters:pd.Series = attr.ib(converter=convert_dict, default=attr.Factory(dict))

    @property
    def valid(self):
        "property for testing if an individual is valid"
        return self.loss.valid

    @property
    def save_name_base(self):
        "property for the save name"
        return hashlib.md5(str(list(self.parameters.items())).encode("utf-8", "ignore")).hexdigest()

    def save(self, meta_dir: Path):
        """Save an individual to a json file"""
        save_path = meta_dir / (self.save_name_base + ".json")
        self.save_full_path(save_path)

    def save_full_path(self, save_path: Path):
        "save the individual to a json file"
        dv_vars = {}

        dv_vars['parameters'] = self.parameters.to_dict()
        dv_vars['transformed_parameters'] = self.transformed_parameters.to_dict()

        dv_vars['loss'] = self.loss.values.to_dict()

        with save_path.open('w', encoding='utf-8') as outfile:
            json.dump(dv_vars, outfile, indent=4, sort_keys=True)

@attr.s
class IndividualMeta(Individual):
    """Meta individual class that also adds additional error information
    fitness_raw is the raw error vector without any weighting or combinations
    fitness_full is the error vector after processing"""
    loss_full: pd.Series = pd.Series(dtype=numpy.float64)
    loss_raw: pd.Series = pd.Series(dtype=numpy.float64)
    loss_absolute: pd.Series = pd.Series(dtype=numpy.float64)

    def get_loss(self):
        "return the fitness of the meta individual"
        temp = {'loss':self.loss_full.to_dict(),
                'loss_raw':self.loss_raw.to_dict()}
        if len(self.loss_absolute) and not self.loss_absolute.isnull().values.all():
            temp['loss_absolute'] = self.loss_absolute.to_dict()
        return temp
