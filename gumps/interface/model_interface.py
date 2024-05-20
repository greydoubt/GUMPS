# SPDX-FileCopyrightText: 2024 Amgen
#
# SPDX-License-Identifier: BSD-3-Clause

"Model interface specification that groups together a value, units, and bounds."
"This interface is NOT final and is intended to be used as a starting point for discussion."

import attrs
import pint
import numpy as np
import pandas as pd
from gumps.common import units
from abc import abstractmethod


ureg = units.unit_registry


def validate_units(self, attribute, value: pint.Quantity):
    "validate that the units are compatible with the base units of the class"
    return value.check(self.base_units())


def validate_bounds(self, attribute, value: float|int|np.ndarray):
    "validate that the value is within the bounds of the class while taking into account units"
    value_converted = ureg.convert(value, self.units, self.bounds_units())
    if np.any(value_converted < self.lower_bound()):
        raise ValueError(f"Value {value_converted} {self.bounds_units()} is less than {self.lower_bound()}")
    if np.any(value_converted > self.upper_bound()):
        raise ValueError(f"Value {value_converted} {self.bounds_units()} is greater than {self.upper_bound()}")


@attrs.define
class Unit:
    "hold a single unit and value, and validate that the value is within bounds"
    value: float|int|np.ndarray = attrs.field(validator=validate_bounds)
    units: pint.Quantity = attrs.field(validator=validate_units, converter=ureg.Quantity)


    @classmethod
    @abstractmethod
    def bounds_units(cls) -> str:
        ...


    @classmethod
    @abstractmethod
    def base_units(cls) -> str:
        ...

    @classmethod
    @abstractmethod
    def lower_bound(cls) -> float|int:
        pass


    @classmethod
    @abstractmethod
    def upper_bound(cls) -> float|int:
        pass


    def convert(self, new_units: str):
        return self.__class__(ureg.convert(self.value, self.units, new_units), new_units)


def validate_units_pandas(self, attribute, value: dict[str, Unit]):
    "validate that the units are compatible with the base units of the class"
    base_units = self.base_units()
    ok = []
    for key, unit in value.items():
        ok.append(unit.check(base_units[key]))
    return all(ok)


def validate_bounds_pandas(self, attribute, value: pd.Series | pd.DataFrame):
    "validate that the value is within the bounds of the class while taking into account units"
    bounds_units = self.bounds_units()
    lower_bound = self.lower_bound()
    upper_bound = self.upper_bound()
    for name, value in value.items():
        value_converted  = ureg.convert(value, self.units[name], bounds_units[name])

        if np.any(value_converted < lower_bound[name]):
            raise ValueError(f"Value {value_converted} {bounds_units[name]} is less than {lower_bound[name]}")
        if np.any(value_converted > upper_bound[name]):
            raise ValueError(f"Value {value_converted} {bounds_units[name]} is greater than {upper_bound[name]}")


@attrs.define
class Units:
    "holds a pandas series or dataframe with one unit per column, and a dict of units"
    value: pd.DataFrame | pd.Series = attrs.field(validator=validate_bounds_pandas)
    units: dict[str, pint.Quantity] = attrs.field(validator=validate_units_pandas, converter=lambda x: {name: ureg.Quantity(unit) for name, unit in x.items()})

    @staticmethod
    @abstractmethod
    def units_used() -> dict[str, Unit]:
        pass

    @classmethod
    def bounds_units(cls) -> dict[str, str]:
        return {name: unit.bounds_units() for name, unit in cls.units_used().items()}

    @classmethod
    def base_units(cls) -> dict[str, str]:
        return {name: unit.base_units() for name, unit in cls.units_used().items()}

    @classmethod
    def lower_bound(cls) -> pd.Series:
        return pd.Series({name: unit.lower_bound() for name, unit in cls.units_used().items()})

    @classmethod
    def upper_bound(cls) -> pd.Series:
        return pd.Series({name: unit.upper_bound() for name, unit in cls.units_used().items()})

    def convert(self, new_units: dict[Unit, str]):
        converted_values = {}
        converted_units = {}
        units_used = self.units_used()
        for name, value in self.value.items():
            if units_used[name] in new_units:
                from_units = self.units[name]
                to_units = new_units[units_used[name]]
                converted_values[name] = ureg.convert(value, from_units.units, to_units)
                converted_units[name] = to_units
            else:
                converted_values[name] = value
                converted_units[name] = self.units[name]
        if isinstance(self.value, pd.Series):
            return self.__class__(pd.Series(converted_values), converted_units)
        else:
            return self.__class__(pd.DataFrame(converted_values), converted_units)


    def __getitem__(self, index):
        "select by index position if value is a dataframe"
        if isinstance(self.value, pd.DataFrame):
            return self.__class__(self.value.iloc[index].rename(None), self.units)
        else:
            raise TypeError("Can only select by index position if value is a Pandas Dataframe")


@attrs.define
class UnitCollection:
    "hold a collection of units and allow conversion of all the values at once"

    def convert(self, new_units:dict[Unit, str]):
        temp = {}
        for field in attrs.fields(self.__class__):
            if field.type in new_units:
                temp[field.name] = getattr(self, field.name).convert(new_units[field.type])
            else:
                temp[field.name] = getattr(self, field.name)

        return self.__class__(**temp)


    def get_series(self) -> pd.Series:
        temp = {}
        for field in attrs.fields(self.__class__):
            temp[field.name] = getattr(self, field.name).value
        return pd.Series(temp)


    def get_units(self) -> dict[str, str]:
        temp = {}
        for field in attrs.fields(self.__class__):
            temp[field.name] = getattr(self, field.name).units
        return temp
