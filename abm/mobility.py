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
        activity_data: Pair (LV, LF) as returned by contacts.load_period_activities().
            - LV is the list of sparse visit matrices for every period;
            - LF is the list of locations of all agents during each period.
            - LT is the list of activity types of all activities during each period.
        """
        visit_matrices, locations, activity_types = activity_data
        self.locations = locations
        self.activity_types = activity_types

        self.n_periods = len(locations)
        self.n_facilities, self.n_agents = visit_matrices[0].shape

        # Initializes the visitor counts
        print("Computing the initial number of visitors per period, per facility..")
        # The following list will contain an array for each period. Each array V is of shape
        # (n_facilities,), such that V[f] is the number of visitors of f during the associated period.
        self.visitors = []
        # The following list is similar to the previous one, but it only counts infected agents.
        # This is used to be able to retrieve the fraction of infected agents in each facility, at
        # any time. This array is initialized to all zeros, i.e. no infected agents.
        self.infected_visitors = []
        for period in range(self.n_periods):
            visitors = np.zeros(self.n_facilities)
            facilities, counts = np.unique(locations[period], return_counts=True)
            visitors[facilities] += counts
            self.visitors.append(visitors)
            self.infected_visitors.append(np.zeros(self.n_facilities))
        print("Done")

    def get_infected_visitors(self):
        """
        Returns
        -------
        The number of infected visitors at every facility, as an array of shape
        (n_facilities,).
        """
        return self.infected_visitors

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