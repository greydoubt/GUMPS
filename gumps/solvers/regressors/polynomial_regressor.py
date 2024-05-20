# SPDX-FileCopyrightText: 2024 Amgen
#
# SPDX-License-Identifier: BSD-3-Clause

"""This method implements a polynomial regressor with save and load methods
along with automatic scaling of the input and output data."""

import logging

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

import warnings
from pathlib import Path

import attrs
import joblib
import pandas as pd
import scipy.special
import sklearn.linear_model
import sklearn.preprocessing
import sklearn.exceptions
import sklearn.feature_selection
import sklearn.base
import optuna

import gumps.solvers.regressors.regression_solver as regression_solver



@attrs.define(kw_only=True)
class PolynomialRegressionParameters(regression_solver.RegressionParameters):
    "Polynomial regression parameters with validation on the polynomial order"
    order: int = attrs.field()
    terms: list[str] | None = attrs.field(default = None)

    @order.validator
    def check(self, attribute, value):
        "validate the order"
        if not (value > 0 and value == int(value)):
            raise ValueError(f"Only integers greater than 0 are allowed for Polynomial regression {attribute} = {value}")

    def get_complete_terms(self) -> list[str]:
        poly = sklearn.preprocessing.PolynomialFeatures(degree = self.order,
                                                        include_bias = True)

        #we only need the first row to get the allowed terms
        poly.fit(self.input_data[:1])
        complete_terms = poly.get_feature_names_out()
        return list(complete_terms)

    def get_terms(self) -> list[str]:
        "get the terms of the polynomial"
        if self.terms is not None:
            return self.terms
        else:
            return self.get_complete_terms()

    def validate_terms(self) -> None:
        "validate the terms against what polynomial features can handle"
        allowed_terms = set(self.get_complete_terms())

        invalid_terms = set(self.terms) - allowed_terms
        if invalid_terms:
            raise ValueError(f"Terms {invalid_terms} are not allowed for polynomial regression with order {self.order}.")

    def __attrs_post_init__(self):
        "perform post initialization"
        input_dimensions = self.input_data.shape[1]
        input_samples = self.input_data.shape[0]

        self.terms = self.get_terms()
        self.validate_terms()

        polynomial_terms = len(self.terms)

        if input_samples <= polynomial_terms:
            raise ValueError(f"Number of samples {input_samples} is less than or equal to the number of polynomial terms {polynomial_terms}.")
        else:
            logger.info("There are %s samples and %s polynomial terms.", input_samples, polynomial_terms)
        super().__attrs_post_init__()


class PolynomialRegressor(regression_solver.AbstractRegressor):
    "Polynomial regressor class with automatic scaling of the input and output data."
    #Allows proper typechecking
    def __init__(self, parameters:PolynomialRegressionParameters):
        self.parameters: PolynomialRegressionParameters
        super().__init__(parameters)
        self.poly = self._get_poly()

    def clone(self, parameters: PolynomialRegressionParameters) -> 'PolynomialRegressor':
        "Clone the regressor"
        return PolynomialRegressor(parameters)

    def _get_regressor(self) -> sklearn.linear_model.LinearRegression:
        "Return the regressor."
        return sklearn.linear_model.LinearRegression()


    def degrees_of_freedom(self) -> int:
        "Return the degrees of freedom."
        if not self.fitted:
            raise RuntimeError("The regressor must be fitted before getting the degrees of freedom.")
        return len(self.parameters.terms)


    def _get_poly(self) -> sklearn.preprocessing.PolynomialFeatures:
        "Return the kernel."
        poly = sklearn.preprocessing.PolynomialFeatures(degree = self.parameters.order,
                                                        include_bias = True)
        return poly


    def save(self, path_dir: Path) -> None:
        "save the regressor, input, and output scaler"
        super().save(path_dir)
        joblib.dump(self.poly, path_dir / "poly.joblib")

    @classmethod
    def _load_instance(cls, path_dir:Path, instance):
        "load the regressor, input, and output scaler"
        with warnings.catch_warnings():
            warnings.filterwarnings("error", category=sklearn.exceptions.InconsistentVersionWarning)
            warnings.simplefilter("error", UserWarning)
            try:
                instance.poly = joblib.load(path_dir / "poly.joblib")
            except (UserWarning, sklearn.exceptions.InconsistentVersionWarning) as exc:
                raise RuntimeError("Failed to load the poly kernel") from exc


    def _fit(self):
        "fit the regressor"
        poly_features = self.poly.fit_transform(self.data_regression.scaled_split.train_input)
        poly_data = pd.DataFrame(poly_features, columns = self.poly.get_feature_names_out())

        poly_data = poly_data[self.parameters.terms]

        self.regressor.fit(poly_data, self.data_regression.scaled_split.train_output)
        self.fitted = True
        self.update_error_metrics()


    def _predict(self, input_data:pd.DataFrame) -> pd.DataFrame:
        "predict the output data"
        input_data_scaled = self.data_regression.input_scaler.transform(input_data)

        poly_features = self.poly.transform(input_data_scaled)
        poly_data = pd.DataFrame(poly_features, columns = self.poly.get_feature_names_out())

        if self.parameters.terms is not None:
            poly_data = poly_data[self.parameters.terms]

        output_data_scaled = self.regressor.predict(poly_data)

        output_data = self.data_regression.output_scaler.inverse_transform(output_data_scaled)

        return output_data

    def auto_tune(self, max_order:int) -> None:
        "Automatically tune the regressor."
        input_data = self.data_regression.scaled_split.full_input
        output_data = self.data_regression.scaled_split.full_output

        poly = sklearn.preprocessing.PolynomialFeatures(degree = max_order, include_bias = True)
        poly_data = poly.fit_transform(input_data)
        full_data = pd.DataFrame(poly_data, columns = poly.get_feature_names_out())
        rfcev = sklearn.feature_selection.RFECV(sklearn.base.clone(self.regressor))
        rfcev.fit(full_data, output_data)
        terms = list(rfcev.get_feature_names_out())

        self.parameters.terms = terms
        self.parameters.order = max_order
        self.poly = self._get_poly()
        self.fit()

    def get_tuned_parameters(self) -> dict:
        data = {}
        data["order"] = self.parameters.order
        data["terms"] = self.parameters.terms
        return data
