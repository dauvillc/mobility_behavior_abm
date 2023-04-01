"""
Clément Dauvilliers - April 1st 2023
Implements the Population clas, which stores all information about the agents, including their states.
"""
import numpy as np
import pandas as pd
from abm.mobility import Mobility


# Dictionary that maps the names of the states to integers
states_dict = {
    "susceptible": 0,
    "infected": 1,
    "recovered": 2
}
n_states = len(states_dict)


class Population:
    """
    The Population class stores all information concerning the agents.
    The agent's mobility information is stored within a Mobility object, contained inside the Population.
    This includes their characteristics, as well as their disease state.
    The class also takes care of maintaining an internal count of the numbers of agents in each state,
    and avoids useless re-computations.
    """
    def __init__(self,
                 n_agents,
                 population_dataset,
                 pop_inf_characteristics,
                 pop_test_characteristics,
                 activity_data):
        """
        Parameters
        ----------
        n_agents: Integer, number of agents in the simulation.
        population_dataset: DataFrame, optional. Dataset containing the agents' attributes.
            If None, it will be loaded.
        pop_inf_characteristics: optional, Float array of shape (n_agents). Values for the
            characteristics of all agents regarding the probability of infection. If None,
            it will be computed using the parameters.
        pop_test_characteristics: optional, Float array of shape (n_agents). Values for the
            characteristics of all agents regarding the probability of being tested. If None,
            it will be computed using the parameters.
        activity_data: Pair (LV, LF) as returned by contacts.load_period_activities().
            - LV is the list of sparse visit matrices for every period;
            - LF is the list of locations of all agents during each period.
            - LT is the list of activity types of all activities during each period.
        """
        self.n_agents = n_agents
        self.population_dataset = population_dataset
        self.pop_inf_characteristics = pop_inf_characteristics
        self.pop_test_characteristics = pop_test_characteristics

        # Initializes the state of all agents to "Susceptible"
        # self.states is an array of shape (n_agents,) whose elements are integers that
        # correspond to disease states, following states_dict[].
        self.states = np.full(n_agents, states_dict['susceptible'], dtype=np.int)
        # self.state_counts is an array of shape (n_states) that counts how many
        # agents are in each state, at all moments.
        self.state_counts = np.full(n_states, 0)
        self.state_counts[states_dict['susceptible']] = n_agents

        # Initializes the Mobility object
        self.mobility = Mobility(activity_data)

    def get_state_count(self, state):
        """
        Returns the number of agents in a given state.
        Parameters
        ----------
        state: str, name of the state (e.g. "infected")

        Returns
        -------
        The number of agents that are in that state, as an integer.
        """
        return self.state_counts[states_dict[state]]

    def get_subset_in_state(self, agent_ids, state):
        """
        Given a set A of agents and a specific state s, returns
        the subset of A made up of the agents that are in state s.
        Parameters
        ----------
        agent_ids: np array, IDs of the larger set of agents.
        state: str, name of the state (e.g. "susceptible").

        Returns
        -------
        subset_ids: np array, IDs of the agents in agent_ids whose state is "state".
        """
        return agent_ids[np.where(self.states[agent_ids] == states_dict[state])[0]]

    def set_agents_state(self, agent_ids, state):
        """
        Sets a specific state for a given set of agents.
        Parameters
        ----------
        agent_ids: array containing the IDs of the agents whose state should be set;
        state: str, name of the state (e.g. "infected").
        """
        # Traduct the state's name into its corresponding integer
        state = states_dict[state]
        # First, we need to look at the current states of the concerned agents,
        # and remove them from the counters associated with those states.
        former_states, former_states_counts = np.unique(self.states[agent_ids], return_counts=True)
        self.state_counts[former_states] -= former_states_counts
        # Second, for all agents that were infected, we need to remove them from the count of infected
        # visitors per facility.
        formerly_infected = self.get_subset_in_state(agent_ids, "infected")
        self.mobility.remove_infected_visitors(formerly_infected)
        # We can now set the new state
        self.states[agent_ids] = state
        # And don't forget to add the agents to the new state's counter
        self.state_counts[state] += len(agent_ids)
        # Finally, if the new state is infected, then we need to count the agents
        # in the infected visitors count
        if state == states_dict['infected']:
            self.mobility.add_infected_visitors(agent_ids)
