# SPDX-FileCopyrightText: 2024 Amgen
#
# SPDX-License-Identifier: BSD-3-Clause

import copy
import bisect
import itertools
from typing import Union, List, Iterable
from pathlib import Path

import numpy
import pandas as pd

import gumps.common.population as pop

def dominates(self:pop.Loss, other:pop.Loss) -> bool:
    """Return true if each objective of *self* is not strictly worse than
    the corresponding objective of *other* and at least one objective is
    strictly better.
    This has been moved into a function to simplify type checking and out of the class
    """
    return (self.values <= other.values).all()

def update_best(ind:pop.Individual, best:pd.Series, best_items:dict, atol:float, rtol:float) -> bool:
    "update the best items and return if the progress is significant"
    significant = False
    for key in ind.loss.values.keys():
        if key in best and ind.loss.values[key] < best[key]:
            significant = not numpy.allclose(ind.loss.values[key], best[key], rtol=rtol, atol=atol)
            best[key] = ind.loss.values[key]
            best_items[key] = copy.deepcopy(ind)
        elif key not in best:
            significant = True
            best[key] = ind.loss.values[key]
            best_items[key] = copy.deepcopy(ind)
    return significant

class ParetoFront:
    """Pareto front tracking. This pareto front keeps track of the best members based on
    minimizing the loss function.mro

    There is also an anti-crowding feature to prevent many similar items
    from beig kept on the front

    The absolute best items for every loss is also kept regardless of its
    position on the pareto front."""

    def __init__(self, dimensions:int, progress_rtol:float=1e-1, progress_atol:float=1e-8):
        self.keys: List[pop.Loss] = list()
        self.items: List[pop.Individual] = list()
        self.similar = similar_func
        self.progress_rtol = progress_rtol
        self.progress_atol = progress_atol

        #due to the crowding measures if solutions are very close together it is possible for the very
        #best solution for one of the goals to not be stored so they should be kept separately and
        #merged in
        self.dimensions = dimensions
        self.best = pd.Series(dtype=numpy.float64)
        self.best_items:dict = {}

    def insert(self, item):
        """Insert a new item in sorted order"""
        item = copy.deepcopy(item)
        i = bisect.bisect(self.keys, item.loss.values.to_list(), key=lambda x:x.values.to_list())
        self.items.insert(i, item)
        self.keys.insert(i, item.loss)

    def remove(self, index):
        "remove an item from the index"
        del self.keys[index]
        del self.items[index]

    def clear(self):
        "clear the pareto front"
        del self.items[:]
        del self.keys[:]

    def __len__(self) -> int:
        "return the size of the pareto front"
        return len(self.items)

    def __iter__(self) -> Iterable[pop.Individual]:
        "iterate over the pareto front"
        return iter(self.items)

    def update(self, population: Iterable[pop.Individual]) -> tuple[list[pop.Individual],bool]:
        """Update the Pareto front hall of fame with the *population* by adding
        the individuals from the population that are not dominated by the hall
        of fame. If any individual in the hall of fame is dominated it is
        removed.

        :param population: A list of individual with a fitness attribute to
                           update the hall of fame with.
        """
        new_members = []
        significant = []
        pareto_length = len(self)
        all_sig = []

        atol = self.progress_atol
        rtol = self.progress_rtol

        for ind in population:
            is_dominated = False
            dominates_one = False
            has_twin = False
            to_remove = []
            local_significant = []

            local_significant.append(update_best(ind, self.best, self.best_items, atol=atol, rtol=rtol))

            for i, hofer in enumerate(self):  # hofer = hall of famer
                if not dominates_one and dominates(hofer.loss,
                    ind.loss
                ):
                    is_dominated = True
                    break
                elif dominates(ind.loss, hofer.loss):
                    dominates_one = True
                    to_remove.append(i)
                    local_significant.append(
                        not similar_func(ind.loss.values, hofer.loss.values)
                    )
                elif similar_func(ind.loss.values, hofer.loss.values
                ) and similar_func(ind.parameters, hofer.parameters):
                    has_twin = True
                    break

            for i in reversed(to_remove):  # Remove the dominated hofer
                self.remove(i)
            if not is_dominated and not has_twin:
                # if the pareto front is empty or a new item is added to the pareto front that is progress
                # however if there is only a single objective and it does not significantly dominate then
                # don't count that as significant progress
                if pareto_length == 0:
                    significant.append(True)
                elif self.dimensions > 1:
                    significant.append(any(local_significant))

                all_sig.append(local_significant)
                self.insert(ind)
                new_members.append(ind)

        return new_members, any(significant)

    def hashes(self) -> set[str]:
        """return the hashes for individuals on the pareto front used for saving and loading individuals"""
        return {individual.save_name_base for individual in self.get_population()}

    def total_entries(self) -> int:
        "return the total number of entries including the best items"
        return sum([1 for i in self.get_population()])


    def get_population(self) -> Iterable[pop.Individual]:
        """yield all individuals in the pareto front"""
        seen = set()
        for ind in itertools.chain(self.items, self.best_items.values()):
            if ind is not None and tuple(ind.parameters) not in seen:
                seen.add(tuple(ind.parameters))
                yield ind

    def get_entries(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """get the complete population and fitness of the pareto front as two ndarray"""
        population = []
        fitnesses  = []

        for ind in self.get_population():
            population.append(ind.parameters)
            fitnesses.append(ind.loss.values)

        return pd.DataFrame(population), pd.DataFrame(fitnesses)

    def get_best_scores(self) -> pd.Series:
        "return the best scores"
        items = pd.DataFrame([i.loss.values for i in self.get_population()])
        return items.min()

class DummyFront(ParetoFront):
    """This is a Dummy Pareto Front that does nothing and is useful for API compatibility
    Pareto front designed to be compatible with DEAP"""

    def __init__(self, dimensions=None):
        "This is here for API compatibility, don't do anything"
        super().__init__(dimensions)

    def update(self, population: Iterable[pop.Individual]) -> tuple[list[pop.Individual],bool]:
        "do not put anything in this front, it is just needed to maintain compatibility"
        return [], False


def similar_func(a:pd.Series, b:pd.Series) -> bool:
    "for minimization the rtol needs to be fairly high otherwise the pareto front contains too many entries"
    return numpy.allclose(a.to_numpy(), b.to_numpy(), rtol=1e-1)

def update_pareto_front(halloffame:Union[ParetoFront, DummyFront], offspring):
    """update the pareto front with the new offspring and determine if significant progress was made"""
    new_members, significant = halloffame.update(
        [
            offspring,
        ]
    )
    return bool(new_members), significant

def clean_dir(dir: Path, hof: ParetoFront):
    "clean the store directory for the pareto front"
    # find all items in directory
    paths = dir.glob("*.json")

    # make set of items based on removing everything after _
    exists = {str(path.stem) for path in paths}

    # make set of allowed keys based on hall of hame
    allowed = hof.hashes()

    # remove everything not in hall of fame
    remove = exists - allowed

    for save_name_base in remove:
        for path in dir.glob("%s*" % save_name_base):
            path.unlink()