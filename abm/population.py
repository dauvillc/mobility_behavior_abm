"""
Clément Dauvilliers - April 1st 2023
Implements the Population clas, which stores all information about the agents, including their states.
"""
import numpy as np
from abm.mobility import Mobility
from abm.characteristics import compute_characteristics, load_population_dataset
from abm.contacts import load_period_activities

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
    Finally, the class contains the times before infected agents become recovered, as well
    as the times since the agents were last tested positive.
    """

    def __init__(self,
                 params,
                 activity_data=None,
                 population_dataset=None,
                 rng=None):
        """
        Parameters
        ----------
        population_dataset: DataFrame, optional. Dataset containing the agents' attributes (especially, social
            and economic and health characteristics).
            If None, it will be loaded from the file indicated in the config file.
        activity_data: optional, Triplet (N, LV, LF) as returned by contacts.load_period_activities().
            - N is the pair of integers (number of agents, number of facilities).
            - LF is the list of locations of all agents during each period.
            - LT is the list of activity types of all activities during each period.
            If not given, will be loaded from the default path indicated in simulation_config.yml.
        rng: optional, specific numpy random number generator to use.
        """
        # Load the activity data if not given
        if activity_data is None:
            activity_data = load_period_activities()

        # Fetch the number of agents from the activity data
        (self.n_agents, _), _, _ = activity_data
        self.params = params
        # Creates an RNG unless one was given
        if rng is None:
            self.rng = np.random.default_rng(seed=42)
        else:
            self.rng = rng

        # Builds the agents' characteristics if required ===================
        # Begins by loading the population dataset:
        if population_dataset is None:
            population_dataset = load_population_dataset()
        self.compute_agents_characteristics(population_dataset)
        # =====================================================================

        # Initializes the Mobility object
        self.mobility = Mobility(activity_data)

        # Defines some objects that are initialized in reset()
        self.states, self.state_counts = None, None
        self.infected_agents_ids = None
        # Initializes all agents to "susceptible"
        self.reset()

    def reset(self):
        """
        (Re)sets the Population object to its state at the beginning of the simulation.
        All agents are set to "susceptible". Their mobility is set to the original activity
        data.
        Returns
        -------
        """
        # Initializes the state of all agents to "Susceptible"
        # self.states is an array of shape (n_agents,) whose elements are integers that
        # correspond to disease states, following states_dict[].
        self.states = np.full(self.n_agents, states_dict['susceptible'], dtype=np.int)
        # self.state_counts is an array of shape (n_states) that counts how many
        # agents are in each state, at all moments.
        self.state_counts = np.full(n_states, 0)
        self.state_counts[states_dict['susceptible']] = self.n_agents
        # The following variable is a set that must, at any moment, contain
        # the IDs of the infected agents. The idea behind it is that the number of
        # infected agents at a time is small compared to the total number of agents. Thus,
        # we want to be able to easily retrieve the IDs of the infected agents without checking
        # every agent one by one.
        self.infected_agents_ids = set()
        # Resets the Mobility object
        self.mobility.reset()

    def compute_agents_characteristics(self, population_df):
        """
        Computes / updates the agents' characteristics, i.e. the scalar products
        <beta_inf, X_inf> and <beta_test, X_test>, where X is the agents' attributes (soc-eco, and
        health-related).
        """
        # For the probability of infection
        self.pop_inf_characteristics = compute_characteristics(population_df, self.params['inf_params'])
        # For the probability of being tested
        self.pop_test_characteristics = compute_characteristics(population_df, self.params['test_params'])

    def get_state_count(self, state: str):
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

    def get_subset_in_state(self, agent_ids: np.array, state: str):
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

    def get_infected_agents(self):
        """
        Returns
        -------
        An array containing the IDs of all currently infected agents.
        """
        return np.array(list(self.infected_agents_ids), dtype=np.int)

    def set_agents_state(self, agent_ids: np.array, state_name: str):
        """
        Sets a specific state for a given set of agents.
        Parameters
        ----------
        agent_ids: array containing the IDs of the agents whose state should be set;
        state_name: str, name of the state (e.g. "infected").
        """
        # Translate the state's name into its corresponding integer
        state = states_dict[state_name]
        # First, we need to look at the current states of the concerned agents,
        # and remove them from the counters associated with those states.
        former_states, former_states_counts = np.unique(self.states[agent_ids], return_counts=True)
        self.state_counts[former_states] -= former_states_counts
        # Second, for all agents that were infected, we need to remove them from the count of infected
        # visitors per facility.
        formerly_infected = self.get_subset_in_state(agent_ids, "infected")
        self.mobility.remove_infected_visitors(formerly_infected)
        # We also need to remove all agents that used to be infected from the infected
        # agents set:
        for id in formerly_infected:
            self.infected_agents_ids.remove(id)
        # We can now set the new state
        self.states[agent_ids] = state
        # And don't forget to add the agents to the new state's counter
        self.state_counts[state] += len(agent_ids)

        # If the new state is "infected", then the agents need to be counted
        # in the count of infected visitors per facility.
        if state_name == 'infected':
            self.mobility.add_infected_visitors(agent_ids)
            # We also need to add the agents to the set of infected agents
            for id in agent_ids:
                self.infected_agents_ids.add(id)

    def change_agent_locations(self, agent_ids, new_facilities):
        """
        For a given set of agents, change their locations to a new one, for
        every period.
        Parameters
        ----------
        agent_ids: np array of integers, IDs of the agents whose location should be
            affected.
        new_facilites: either an integer, or a list of arrays.
            If an integer, ID of the facility to which all agents will be relocated.
            If a list, should give the new locations of the agents for each period.
                In details, a list L containing n_periods elements such that for a period p,
                L[p] is an array such that L[p][i] is the new location of agent agent_ids[i]
                during period p.
        """
        # We're going to change the agents' locations using
        # Mobility.change_agent_locations(); however that method does not
        # change the count of infected visitors, because the Mobility object doesn't
        # know which agents are infected.
        # Therefore, we'll first remove the targeted infected agents from their former
        # locations' infected counts, then change their locations, and finally re-add
        # them to their new locations' infected counts.

        # Step 1: Retrieve the infected agents within agent_ids:
        infected_agents = self.get_subset_in_state(agent_ids, "infected")
        # Step 2: Pretend to the Mobility object that they aren't infected anymore.
        self.mobility.remove_infected_visitors(infected_agents)
        # Step 3: Change the locations of all agents
        self.mobility.change_agent_locations(agent_ids, new_facilities)
        # Step 4: Set the infected agents back to infected for the Mobility object.
        self.mobility.add_infected_visitors(infected_agents)

    def reset_agent_locations(self, agent_ids):
        """
        Resets the locations of a set of agents back to their original values.
        Parameters
        ----------
        agent_ids: ndarray of integers, IDs of the targeted agents.
        """
        self.mobility.reset_locations(agent_ids)
