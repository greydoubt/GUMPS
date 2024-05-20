# SPDX-FileCopyrightText: 2024 Amgen
#
# SPDX-License-Identifier: BSD-3-Clause

"Base units for the model"

import attrs

from gumps.interface.model_interface import Unit


@attrs.define
class Mass(Unit):
    @classmethod
    def base_units(cls):
        return "[mass]"


@attrs.define
class Density(Unit):
    @classmethod
    def base_units(cls):
        return "[mass]/[length]^3"


@attrs.define
class Length(Unit):
    @classmethod
    def base_units(cls):
        return "[length]"


@attrs.define
class Area(Unit):
    @classmethod
    def base_units(cls):
        return "[length]^2"


@attrs.define
class Volume(Unit):
    @classmethod
    def base_units(cls):
        return "[length]^3"


@attrs.define
class Time(Unit):
    @classmethod
    def base_units(cls):
        return "[time]"


@attrs.define
class Pressure(Unit):
    @classmethod
    def base_units(cls):
        return "[mass]/[length]/[time]^2"


@attrs.define
class Temperature(Unit):
    @classmethod
    def base_units(cls):
        return "[temperature]"


@attrs.define
class Viscosity(Unit):
    @classmethod
    def base_units(cls):
        return "[mass]/[length]/[time]"


@attrs.define
class SurfaceTension(Unit):
    @classmethod
    def base_units(cls):
        return  "[force]/[length]"


@attrs.define
class BulkModulus(Unit):
    @classmethod
    def base_units(cls):
        return "[force]/[length]^2"


@attrs.define
class Angle(Unit):
    @classmethod
    def base_units(cls):
        return "radians"


@attrs.define
class Speed(Unit):
    @classmethod
    def base_units(cls):
        return "[length]/[time]"
