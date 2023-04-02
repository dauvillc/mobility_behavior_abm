"""
Clément Dauvilliers - April 2nd 2023
Implements the ABM class, which is the mother class of the simulation.
"""
import numpy as np
import pandas as pd
import abm.characteristics as ch
from copy import deepcopy
from abm.population import Population
from abm.results import ABM_Results
from abm.recovery import draw_recovery_times


class ABM:
    """
    The ABM clas takes care of creating the population, and running the algorithm. This is the class
    that updates the variables over time, while the other classes only store information
    in a static manner.
    """

    def __init__(self, params, activity_data,
                 population_dataset=None,
                 pop_inf_characteristics=None,
                 pop_test_characteristics=None,
                 seed=42):
        """

        Parameters
        ----------
        params: Dictionary of parameters to give to the model,
            such as the recovery rate.
        activity_data: Pair (LV, LF) as returned by contacts.load_period_activities().
            - LV is the list of sparse visit matrices for every period;
            - LF is the list of locations of all agents during each period.
            - LT is the list of activity types of all acitvities during each period.
        population_dataset: DataFrame, optional. Dataset containing the agents' attributes.
            If None, it will be loaded.
        pop_inf_characteristics: optional, Float array of shape (n_agents). Values for the
            characteristics of all agents regarding the probability of infection. If None,
            it will be computed using the parameters.
        pop_test_characteristics: optional, Float array of shape (n_agents). Values for the
            characteristics of all agents regarding the probability of being tested. If None,
            it will be computed using the parameters.
        visitors_counts: List of Integer arrays of shape (n_facilities), optional. Number of visitors
            per period, per facility, not accounting for activity reduction. If None, it will
            be computed from the activity matrix.
        seed: Random seed to use.

        Returns
        -------

        """
        self.params = deepcopy(params)
        self.seed = seed
        self.rng = np.random.default_rng(seed=seed)

        # Deduces some constants of the simulation from the visit matrices
        # (those aren't kept in memory however)
        visit_matrices_coo, _, _ = activity_data
        self.n_facilities, self.n_agents = visit_matrices_coo[0].shape
        self.n_periods = len(visit_matrices_coo)

        # Sets the parameters' default values
        self.set_default_param("recovery_mean_time", 8.0)
        self.set_default_param("recovery_std_time", 2.0)

        # Loads the population socio-eco attributes if required
        if population_dataset is None:
            print("Loading the population dataset..")
            population_df = pd.read_csv('data/abm/vaud/extracted/vaud_population.csv.gz', index_col='agent_index')
            population_df = population_df.sort_index()
            print("Done")
        else:
            population_df = population_dataset

        # Builds the agents' characteristics if required
        # For the probability of infection
        if pop_inf_characteristics is None:
            inf_characs = ch.compute_characteristics(population_df, self.params['inf_params'])
        else:
            inf_characs = pop_inf_characteristics

        # For the probability of being tested
        if pop_test_characteristics is None:
            test_characs = ch.compute_characteristics(population_df, self.params['test_params'])
        else:
            test_characs = pop_test_characteristics

        # Builds the Population object
        self.population = Population(self.n_agents, population_df, inf_characs, test_characs, activity_data,
                                     self.params, self.rng)

        # Timers
        # they are only defined here, not initialized. It's not mandatory in Python,
        # but helps IDEs (and humans !) to understand the class structure.
        self.infection_timer = None
        self.last_tested_pos_timer = None

        # Other variables that are defined here but initialized in init_simulation():
        self.period, self.day = None, None
        self.results = ABM_Results(self.n_periods)

    def set_default_param(self, param, value):
        """
        Checks if a parameter was specified by the user; otherwise sets
        its value to a default one.
        Parameters
        ----------
        param: str, name of the parameter.
        value: Default value to set.
        """
        if param not in self.params:
            self.params[param] = value

    def init_simulation(self, seed=None):
        """
        (Re)initializes the simulation, while conserving the model's
        parameters. Can be called either in the class constructor or
        to begin a new simulation without changing the parameters.
        Parameters
        ----------
        seed: Random seed to use. If none is specified, will default to the ABM's original
            seed.

        Returns
        -------
        """
        # Initialize general vars about the simulation
        self.period, self.day, self.n_days = 0, 0, 0
        # Initializes an empty results manager
        self.results = ABM_Results(self.n_periods)
        # Resets the RNG
        if seed is None:
            seed = self.seed
        self.rng = np.random.default_rng(seed=seed)

        # Resets the Population (sets all agents to "susceptible").
        self.population.reset()

        # Initializes the timers
        # The infection timer stores, for every infected agent, how many periods they
        # still have to spend before they become recovered.
        # The timer is implemented as an array of length (n_agents,), which contains
        # for every agent either a positive integer, or a negative value to indicate
        # that the agent is not infected.
        self.infection_timer = np.full(self.n_agents, -1)

    def set_infected(self, agent_ids):
        """
        Sets a group of agents to the "infected" state. For each of those
        agents, a timer starts that counts how many periods the agent still has to
        spend as infected. If an agent is not in the "susceptible" state, this method has no effect
        onto them.
        Parameters
        ----------
        agent_ids: np array, IDs of the newly infected agents.
        """
        # First: filters the agents that are susceptible:
        agent_ids = self.population.get_subset_in_state(agent_ids, "susceptible")
        # Modifies the state of the agents in the Population object
        self.population.set_agents_state(agent_ids, "infected")
        # Draws random recovery times
        recovery_times = draw_recovery_times(agent_ids.shape[0],
                                             self.params['recovery_mean_time'],
                                             self.params['recovery_std_time'],
                                             self.n_periods,
                                             self.rng)
        # Starts the infection timer of those agents
        self.infection_timer[agent_ids] = recovery_times

    def random_infections(self, proba):
        """
        Selects a random set of agents to infect. Every agent
        has an equal chance to be infected.
        Parameters
        ----------
        proba: float between 0 and 1, probability for every agent to become infected.
        Returns
        -------
        An array giving the IDs of the selected agents.
        """
        draws = self.rng.random(self.n_agents)
        return np.where(draws < proba)[0]

    def process_infections(self, forced_infection_proba=None):
        """
        Runs the infection process:
        - Computes the level of infection of every agent;
        - Computes their probability of infection;
        - Randomly draws according to those probabilities.
        Parameters
        ----------
        forced_infection_proba: float between 0 and 1, optional.
            If specified, the method will instead randomly infect a fraction of the population. Each
            agent then has a forced_infection_proba chance of being infected.
            In this case, the levels of infection returned are zero.
        Returns
        -------
        A pair (LI, infected_agents) where:
        - LI is an array of shape (n_agents) containing the level of infection of every agent;
        - infected_agents is an array containing the IDs of the agents that have become infected.
        """
        if forced_infection_proba is not None:
            infected_agents = self.random_infections(forced_infection_proba)
            return np.zeros(self.n_agents), infected_agents
        else:
            # TODO
            pass

    def iterate_day(self, n_forced_infections=None):
        """
        Processes a single day of simulation.
        Parameters
        ----------
        n_forced_infections: int, optional. If specified, the infection process will instead
            be to randomly draw a set of n_forced_infections agents, which will be infected
            over the day. In this case, the forced infections are uniformly spread over the periods.
        """
        for period in range(self.n_periods):
            # === Recovery process ====================
            # First: retrieve the IDs of all currently infected agents
            infected_agents = self.population.get_infected_agents()
            # for all agents whose infection time has reached zero, they recover
            recovering_agents = infected_agents[self.infection_timer[infected_agents] == 0]
            self.population.set_agents_state(recovering_agents, "recovered")
            # for all still infected agents, reduce their recovery time by one period
            self.infection_timer[infected_agents] -= 1

            # === Infection process ===================
            if n_forced_infections is not None:
                # The forced infections are uniformly spread over the periods. To do so,
                # we need to compute the proba for every agent, during each period, to be
                # randomly selected:
                infection_proba = n_forced_infections / (self.n_periods * self.n_agents)
                infection_levels, infected_agents = self.process_infections(forced_infection_proba=infection_proba)
            else:
                # TODO
                pass
            # Computes the levels of infection, and selects the agents which become infecte
            # The following filters the agents that are "susceptible", then sets them as "infected"
            # and initiates their recovery process.
            self.set_infected(infected_agents)

            # === Storing per-period results ==========
            # Number of new infections
            self.results.store_per_period("new infections", infected_agents.shape[0])
            # Number of currently infected agents
            self.results.store_per_period("infected agents", self.population.get_state_count("infected"))

            # === Variables update ====================
            self.period = (1 + self.period) % self.n_periods

        # ======= Daily variables update ==============
        self.day += 1

    def force_simulation_start(self, forced_infections):
        """
        Sets the initial conditions of the simulation by forcing a given
        number of infections for a fixed number of days.
        The forced infections are uniformly spread within the population.
        Parameters
        ----------
        forced_infections: array-like I of integers, such that I[d] is the number
            of infections that need to occur on day d.
        """
        # First: initialize the simulation variables
        self.init_simulation()
        # Iterate over a few days, with the forced infections option enabled
        for n_forced_infections in forced_infections:
            self.iterate_day(n_forced_infections=n_forced_infections)
