"""
Clément Dauvilliers - April 2nd 2023
Implements the ABM_Results class, which keeps track of the various results of the simulation.
"""
import numpy as np
from collections import defaultdict
from copy import deepcopy

import pandas as pd


class ABM_Results:
    """
    The ABM_Results class stores the simulation results, such as the number of tests per period,
    or the IDs of the newly infected agents.
    The class ABM_Results can store two main types of results: per period, or daily.
    For example, a per-period result can be added via 'results.store_per_period("total tests", value)'
    By default, results.get_per_period(name) returns an empty list.
    The class can also store other variables that aren't per period or daily, using
    results.store(var_name, value). In this case, value can be any Python object, not necessarily a
    number.
    """

    def __init__(self, n_periods):
        """
        Parameters
        ----------
        n_periods: int, number of periods within a day.
        """
        self.n_periods = n_periods
        # Per-period results dictionary
        self.per_period = defaultdict(list)
        # Daily results dictionary
        self.daily = defaultdict(list)
        # Other stored information:
        self.other = defaultdict(list)

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

    def get_per_period_results(self):
        """
        Returns
        -------
        A pandas DataFrame with the following columns:
        - day
        - period
        - var for every per-period variable stored.
        """
        results_df = pd.DataFrame(self.per_period)
        results_df['period'] = np.arange(0, results_df.shape[0]) % self.n_periods
        results_df['day'] = results_df.index // self.n_periods

        return results_df

    def get_daily_results(self):
        """
        Returns
        -------
        A pandas DataFrame with the following columns:
        - day
        - var for every daily variable stored;
        - "daily summed var" for every per-period var, summed over each day;
        - "daily avg var" for every per-period var, averaged over each day.
        """
        # Retrieves the daily results
        results_df = pd.DataFrame(self.daily)
        # For each per-period variable, sums its values over each day
        results_period_df = self.get_per_period_results()
        daily_sums = results_period_df.groupby('day').sum()
        for var_name in list(self.per_period.keys()):
            results_df["daily summed " + var_name] = daily_sums[var_name]
        # Same, but with the average over each day
        daily_avgs = results_period_df.groupby('day').mean()
        for var_name in list(self.per_period.keys()):
            results_df["daily avg " + var_name] = daily_avgs[var_name]

        results_df['day'] = np.arange(0, results_df.shape[0])
        return results_df

    def store(self, var_name, value):
        """
        Stores a new value for a variable that is neither per-period nor daily,
        and whose values may be anything.
        Parameters
        ----------
        variable_name: str, name of the variable (e.g. "infected IDs").
        value: value to store.
        """
        self.other[var_name].append(deepcopy(value))

    def get(self, var_name):
        """
        Returns the stored values for a given variable, which is not stored
        as a per-period or daily variable.
        Parameters
        ----------
        var_name: variable name;

        Returns
        -------
        The list of stored values of the given variable.
        """
        return self.other[var_name]
