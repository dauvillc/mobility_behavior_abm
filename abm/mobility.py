"""
Clément Dauvilliers - April 1st 2023
Implements the Mobility class, which stores and manages the activities
performed by the agents.
"""
import numpy as np
import pandas as pd


class Mobility:
    """
    The Mobility class stores the activities performed by the agents during the day. This means storing
    for every period the location of every agent, as well as the type of the corresponding activity.
    The class maintains an internal count of the numbers of agents in every facility during every period,
    and avoids useless re-computations.
    The class also maintains an internal count of the numbers of infected agents in every facility during
    every period.
    """

    def __init__(self, activity_data):
        """
        activity_data: Triplet (N, LV, LF) as returned by contacts.load_period_activities().
            - N is the pair of integers (number of agents, number of facilities).
            - LF is the list of locations of all agents during each period.
            - LT is the list of activity types of all activities during each period.
        """
        (self.n_agents, self.n_facilities), self.locations, self.activity_types = activity_data
        self.n_periods = len(self.locations)
        # We'll save a copy of the locations, which will never be modified. This is in case the locations
        # are modified at some point: we can always retrieve the original one using this copy.
        self.original_locations = self.locations.copy()

        # Initializes the visitor counts
        # The following list will contain an array for each period. Each array V is of shape
        # (n_facilities,), such that V[f] is the number of visitors of f during the associated period.
        self.visitors = None
        # The following list is similar to the previous one, but it only counts infected agents.
        # This is used to be able to retrieve the fraction of infected agents in each facility, at
        # any time. This array is initialized to all zeros, i.e. no infected agents.
        self.infected_visitors = None
        # This method is the one that actually computes the initial counts.
        self.reset()

    def reset(self):
        """
        (Re)sets the Mobility object to its state at the beginning of the simulation,
        without reloading the activity data.
        Returns
        -------
        """
        # For every period, count how many agents and infected agents are in each facility,
        # and store it.
        self.visitors, self.infected_visitors = [], []
        for period in range(self.n_periods):
            visitors = np.zeros(self.n_facilities, dtype=np.int)
            facilities, counts = np.unique(self.locations[period], return_counts=True)
            visitors[facilities] += counts
            self.visitors.append(visitors)
            self.infected_visitors.append(np.zeros(self.n_facilities, dtype=np.int))

    def get_visitors(self, period):
        """
        Parameters
        ----------
        period: int, which time period to consider.
        Returns
        -------
        The number of visitors at every facility during the given period,
        as an array of shape (n_facilities,).
        """
        return self.visitors[period]

    def get_infected_visitors(self, period):
        """
        Parameters
        ----------
        period: int, which time period to consider.
        Returns
        -------
        The number of infected visitors at every facility during the given period,
        as an array of shape (n_facilities,).
        """
        return self.infected_visitors[period]

    def get_locations(self, period=None):
        """
        Parameters
        ----------
        period: int, optional. If specified, returns the locations for that period only; otherwise
            return a list containing the locations for all periods.
        Returns
        -------
        If period is given, returns an array L of shape (n_agents) such that L[i] is the location of agent i
            during the specified period.
        If period is None, a list containing such arrays for every period.
        """
        # We don't return a copy to limit computation time, although it would be safer.
        if period is None:
            return self.locations
        else:
            return self.locations[period]

    def add_infected_visitors(self, agent_ids):
        """
        Indicates to the Mobility object that a given set of formerly non-infected
        agents are now infected. This method takes care of adding
        those agents to the count of infected visitors.
        Parameters
        ----------
        agent_ids: np array of integers, IDs of the agents concerned.
        """
        if agent_ids.shape[0] == 0:
            return
        # We must take the change into account for every period
        for period in range(self.n_periods):
            # Retrieves:
            # - the locations of the concerned agents;
            # - How many concerned agents are situated in each of those locations.
            specific_locations, counts = np.unique(self.locations[period][agent_ids], return_counts=True)
            # Adds them to the count of infected visitors
            self.infected_visitors[period][specific_locations] += counts

    def remove_infected_visitors(self, agent_ids):
        """
        Opposite function of self.add_infected_visitors().
        Indicates to the Mobility object that a given set of formerly infected
        agents are now no longer infected. This method takes care of subtracting
        those agents from the count of infected visitors.
        Parameters
        ----------
        agent_ids: np array of integers, IDs of the agents concerned.
        """
        if agent_ids.shape[0] == 0:
            return
        # We must take the change into account for every period
        for period in range(self.n_periods):
            # Retrieves:
            # - the locations of the concerned agents;
            # - How many concerned agents are situated in each of those locations.
            specific_locations, counts = np.unique(self.locations[period][agent_ids], return_counts=True)
            # Removes them from the count of infected visitors
            self.infected_visitors[period][specific_locations] -= counts

    def change_agent_locations(self, agent_ids, new_facilities):
        """
        For a given set of agents, change their locations to a new one, for
        every period.
        WARNING: This function does NOT change the counts of infected visitors.
        It must be externally adjusted.
        Parameters
        ----------
        agent_ids: np array of integers, IDs of the agents whose location should be
            affected.
        new_facility_id: either an integer, or an array of same shape as agent_ids.
            If an integer, ID of the facility to which all agents will be relocated.
            If an array, ID of the facility to which each agent will be relocated.
        """
        # Pre-compute some variables that are common between all periods
        # We'll need the non-duplicated new locations and the count of agents at each new location.
        # If the new location is an integer, we can make a simplification:
        if isinstance(new_facilities, np.ndarray):
            # If new_facilities is an array, we need to count for each new facility
            # how many visitors it has gained.
            new_locations_unique, new_counts = np.unique(new_facilities, return_counts=True)
        elif isinstance(new_facilities, int) or isinstance(new_facilities, np.int64):
            # If there's only a single new facility, then its number of new visitors
            # is the number of agents that have been relocated into it.
            new_locations_unique, new_counts = new_facilities, agent_ids.shape[0]
        else:
            raise ValueError("new_facilities should be either an int or a np array")

        # We can now proceed to change the locations for every period:
        for period in range(self.n_periods):
            # First step: remove the agents from their previous facility.
            # To do so, we need to remove them from the count of visitors.
            former_locations, former_counts = np.unique(self.locations[period][agent_ids], return_counts=True)
            self.visitors[period][former_locations] -= former_counts
            # We can now change the locations
            # Reminder: using self.original_locations, we'll still be able to retrieve the former locations.
            # Remark: the following line works both if new_facilities is an array or an integer.
            self.locations[period][agent_ids] = new_facilities
            # We also need to add the agents to their new facilities' visitors counts
            self.visitors[period][new_locations_unique] += new_counts
