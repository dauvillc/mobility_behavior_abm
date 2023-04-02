"""
Clément Dauvilliers - April 2nd 2023
Implements the ABM_Results class, which keeps track of the various results of the simulation.
"""
import numpy as np
from collections import defaultdict
from copy import deepcopy


class ABM_Results:
    """
    The ABM_Results class stores the simulation results, such as the number of tests per period,
    or the IDs of the newly infected agents.
    The class ABM_Results can store two types of results: per_period, or daily.
    For example, a per-period result can be added via 'results.store_per_period("total tests", value)'
    By default, results.get_per_period(name) returns an empty list.
    """

    def __init__(self):
        """

        """
        # Per-period results dictionary
        self.per_period = defaultdict(list)
        # Daily results dictionary
        self.daily = defaultdict(list)

    def get_per_period(self, variable_name):
        """
        Returns the stored values of a given per-period variable.
        Parameters
        ----------
        variable_name: str, name of the variable (e.g. "new infections").

        Returns
        -------
        A list giving the successively stored value of the requested variable.
        The list is actually a copy of the stored result.
        """
        return deepcopy(self.per_period[variable_name])

    def get_daily(self, variable_name):
        """
        Returns the stored values of a given daily variable.
        Parameters
        ----------
        variable_name: str, name of the variable (e.g. "new infections").

        Returns
        -------
        A list giving the successively stored value of the requested variable.
        The list is actually a copy of the stored result.
        """
        return deepcopy(self.daily[variable_name])

    def store_per_period(self, variable_name, value):
        """
        Stores a new value for a given per-period variable.
        Parameters
        ----------
        variable_name: str, name of the variable (e.g. "positive tests").
        value: value to store.
        """
        self.per_period[variable_name].append(deepcopy(value))

    def store_daily(self, variable_name, value):
        """
        Stores a new value for a given daily variable.
        Parameters
        ----------
        variable_name: str, name of the variable (e.g. "positive tests").
        value: value to store.
        """
        self.daily[variable_name].append(deepcopy(value))
