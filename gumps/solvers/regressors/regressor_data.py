# SPDX-FileCopyrightText: 2024 Amgen
#
# SPDX-License-Identifier: BSD-3-Clause

#This holds all the data for regressors and handles train/split, updates, and variable transforms
import attrs
import pandas as pd
import numpy as np
from pathlib import Path
import joblib
from gumps.scalers.scaler import AbstractScaler
import sklearn.model_selection

@attrs.define(kw_only=True)
class Split:
    train_input: pd.DataFrame
    validation_input: pd.DataFrame
    train_output: pd.DataFrame
    validation_output: pd.DataFrame
    full_input: pd.DataFrame
    full_output: pd.DataFrame

def keep_unique(input_data:pd.DataFrame, output_data:pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    "keep only the unique combination of input and output data"
    input_data = input_data.reset_index(drop=True)
    output_data = output_data.reset_index(drop=True)

    duplicate_input = input_data.duplicated()
    duplicate_output = output_data.duplicated()
    duplicates = duplicate_input & duplicate_output

    input_data_new = input_data[~duplicates].reset_index(drop=True)
    output_data_new = output_data[~duplicates].reset_index(drop=True)
    return input_data_new, output_data_new

@attrs.define(kw_only=True)
class DataSettings:
    input_data: pd.DataFrame
    output_data: pd.DataFrame
    validation_fraction: float = attrs.field(default=0.0)
    @validation_fraction.validator
    def _check_validation_fraction(self, _, value):
        if not (0 <= value < 1):
            raise ValueError(f"validation_fraction must be between 0 and 1, not {value}")
    random_state: int | None = None
    train_test_split: str = attrs.field(default="random")
    @train_test_split.validator
    def _check_train_test_split(self, _, value):
        if value not in ["random", "sequential", "manual"]:
            raise ValueError(f"train_test_split must be 'random', 'sequential', or 'manual', not {value}")
    train_indices: np.ndarray|None = attrs.field(eq=attrs.cmp_using(eq=np.array_equal), default=None)
    validation_indices: np.ndarray|None =attrs.field(eq=attrs.cmp_using(eq=np.array_equal), default=None)

    def __attrs_post_init__(self):
        "perform post initialization"
        if self.train_test_split == "manual":
            if self.train_indices is None or self.validation_indices is None:
                raise ValueError("train_indices and validation_indices must be provided when train_test_split is 'manual'")
            else:
                self.validation_fraction = self.calculate_validation_fraction()
        elif self.train_test_split in {"sequential", "random"} and self.train_indices is None and self.validation_indices is None:
            self.train_indices, self.validation_indices = self.regenerate_indices()

    def regenerate_indices(self) -> tuple[np.ndarray, np.ndarray]:
        "regenerate the indices"
        if self.validation_fraction == 0:
            return np.array(self.input_data.index), np.array([])
        elif self.train_test_split == "sequential":
            return self._sequential_split()
        elif self.train_test_split == "random":
            return self._random_split()
        else:
            raise ValueError("Cannot regenerate indices when train_test_split is 'manual'")

    def calculate_validation_fraction(self)-> float:
        "calculate the validation fraction"
        return len(self.input_data.loc[self.validation_indices]) / len(self.input_data)

    def _sequential_split(self) -> tuple[np.ndarray, np.ndarray]:
        "split the data sequentially"
        train_indices, test_indices = sklearn.model_selection.train_test_split(self.input_data.index,
                                                        test_size=self.validation_fraction,
                                                        shuffle=False)
        return np.array(train_indices), np.array(test_indices)


    def _random_split(self) -> tuple[np.ndarray, np.ndarray]:
        "split the data randomly"
        train_indices, test_indices = sklearn.model_selection.train_test_split(self.input_data.index,
                                                        test_size=self.validation_fraction,
                                                        random_state=self.random_state)
        return np.array(train_indices), np.array(test_indices)


class DataRegression:
    input_scaler: AbstractScaler
    output_scaler: AbstractScaler
    split: Split
    scaled_split: Split
    settings: DataSettings

    def __init__(self, settings: DataSettings, input_scaler: AbstractScaler, output_scaler: AbstractScaler) -> None:
        self.settings = settings
        self.input_scaler = input_scaler
        self.output_scaler = output_scaler
        self.split = self.split_data()
        self.fit_scaler()
        self.scaled_split = self.scale_data()

    def split_data(self) -> Split:
        "split the data"
        train_input = self.settings.input_data.loc[self.settings.train_indices]
        validation_input = self.settings.input_data.loc[self.settings.validation_indices]
        train_output = self.settings.output_data.loc[self.settings.train_indices]
        validation_output = self.settings.output_data.loc[self.settings.validation_indices]
        return Split(train_input=train_input,
                 validation_input=validation_input,
                 train_output=train_output,
                 validation_output=validation_output,
                 full_input=self.settings.input_data,
                 full_output=self.settings.output_data)

    def fit_scaler(self):
        "fit the scalers"
        self.input_scaler.fit(self.split.train_input, lower_bound=self.split.full_input.min(), upper_bound=self.split.full_input.max())
        self.output_scaler.fit(self.split.train_output, lower_bound=self.split.full_output.min(), upper_bound=self.split.full_output.max())

    def scale_data(self) -> Split:
        "scale the data"
        train_input = self.input_scaler.transform(self.split.train_input)

        if len(self.split.validation_input) > 0:
            validation_input = self.input_scaler.transform(self.split.validation_input)
        else:
            #This keeps the column names even if it is empty
            validation_input = self.split.validation_input.copy()

        train_output = self.output_scaler.transform(self.split.train_output)

        if len(self.split.validation_output) > 0:
            validation_output = self.output_scaler.transform(self.split.validation_output)
        else:
            #This keeps the column names even if it is empty
            validation_output = self.split.validation_output.copy()

        full_input = self.input_scaler.transform(self.split.full_input)
        full_output = self.output_scaler.transform(self.split.full_output)
        return Split(train_input=train_input,
                    validation_input=validation_input,
                    train_output=train_output,
                    validation_output=validation_output,
                    full_input=full_input,
                    full_output=full_output)

    def save(self, path_dir: Path) -> None:
        "save the regressor, input scaler and output scaler"
        data_dir = path_dir / "data"
        data_dir.mkdir(parents=True, exist_ok=True)

        joblib.dump(self.input_scaler, data_dir / "input_scaler.joblib")
        joblib.dump(self.output_scaler, data_dir / "output_scaler.joblib")
        joblib.dump(self.settings, data_dir / "settings.joblib")
        joblib.dump(self.split, data_dir / "split.joblib")
        joblib.dump(self.scaled_split, data_dir / "scaled_split.joblib")

    @classmethod
    def has_data(cls, path_dir:Path) -> bool:
        "return whether or not the data exists"
        data_dir = path_dir / "data"
        return data_dir.exists()

    @classmethod
    def load(cls, path_dir: Path):
        "load the regressor, input scaler and output scaler"
        data_dir = path_dir / "data"

        input_scaler = joblib.load(data_dir / "input_scaler.joblib")
        output_scaler = joblib.load(data_dir / "output_scaler.joblib")
        settings = joblib.load(data_dir / "settings.joblib")
        split = joblib.load(data_dir / "split.joblib")
        scaled_split = joblib.load(data_dir / "scaled_split.joblib")

        instance = cls.__new__(cls)
        instance.input_scaler = input_scaler
        instance.output_scaler = output_scaler
        instance.settings = settings
        instance.split = split
        instance.scaled_split = scaled_split

        return instance

    def update_data(self, input_data:pd.DataFrame, output_data:pd.DataFrame):
        "update the data"
        if self.settings.train_test_split == "manual":
            raise ValueError("Cannot update data when train_test_split is 'manual' use update_split instead")

        #add the new data to the existing data
        new_input = pd.concat([self.settings.input_data, input_data])
        new_output = pd.concat([self.settings.output_data, output_data])
        new_input, new_output = keep_unique(new_input, new_output)
        self.settings.input_data = new_input
        self.settings.output_data = new_output
        self.settings.train_indices, self.settings.validation_indices = self.settings.regenerate_indices()

        self.split = self.split_data()
        self.fit_scaler()
        self.scaled_split = self.scale_data()

    def reset_split(self, split:Split):
        "replace the split with new split information, refit the scalers and rescale the data"
        self.settings.input_data = split.full_input
        self.settings.output_data = split.full_output
        self.settings.train_indices = np.array(split.train_input.index)
        self.settings.validation_indices = np.array(split.validation_input.index)
        self.settings.validation_fraction = self.settings.calculate_validation_fraction()
        self.split = split
        self.fit_scaler()
        self.scaled_split = self.scale_data()

