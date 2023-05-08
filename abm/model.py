"""
Clément Dauvilliers - April 2nd 2023
Implements the ABM class, which is the mother class of the simulation.
"""
import numpy as np
from copy import deepcopy
from abm.contacts import load_period_activities
from abm.population import Population
from abm.results import ABM_Results
from abm.recovery import draw_recovery_times
from abm.utils import sigmoid
from abm.event import EventType, EventQueue


class ABM:
    """
    The ABM clas takes care of creating the population, and running the algorithm. This is the class
    that updates the variables over time, while the other classes only store information
    in a static manner.
    """

    def __init__(self, params, activity_data=None,
                 population_dataset=None,
                 seed=42):
        """

        Parameters
        ----------
        params: Dictionary of parameters to give to the model,
            such as the recovery rate.
        activity_data: optional, Triplet (N, LV, LF) as returned by contacts.load_period_activities().
            - N is the pair of integers (number of agents, number of facilities).
            - LF is the list of locations of all agents during each period.
            - LT is the list of activity types of all activities during each period.
            If not given, will be automatically loaded from the default location in simulation_config.yml.
        population_dataset: DataFrame, optional. Dataset containing the agents' attributes (especially, social
            and economic and health characteristics).
            If None, it will be loaded.
        seed: Random seed to use.

        Returns
        -------

        """
        self.params = deepcopy(params)
        self.seed = seed
        self.set_seed(seed)

        # Load the activity data from the default location, if not given:
        if activity_data is None:
            activity_data = load_period_activities()

        # Retrieve some constants of the simulation:
        (self.n_agents, self.n_facilities), agents_locations, _ = activity_data
        self.n_periods = len(agents_locations)

        # Sets the parameters' default values
        self.set_default_param("recovery_mean_time", 8.0)
        self.set_default_param("recovery_std_time", 2.0)
        self.set_default_param("inf_proba_sigmoid_slope", 1.0)
        self.set_default_param("test_proba_sigmoid_slope", 1.0)

        # Builds the Population object
        self.population = Population(self.params, activity_data,
                                     population_dataset, self.rng)

        # Timers
        # they are only defined here, not initialized. It's not mandatory in Python,
        # but helps IDEs (and humans !) to understand the class structure.
        self.infection_timer = None
        self.last_tested_pos_timer = None

        # Other variables that are defined here but initialized in init_simulation():
        self.period, self.day = None, None
        self.results = ABM_Results(self.n_periods)
        self.event_queue = EventQueue()
        self.initialized = False

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

    def set_seed(self, seed):
        """
        Sets the model's random seed.
        Parameters
        ----------
        seed: integer, seed to use.
        """
        self.seed = seed
        self.rng = np.random.default_rng(seed=seed)

    def set_param(self, param_name, value, population_dataset=None):
        """
        Sets the value of a given parameter.
        If the param is 'inf_params' or 'test_params' (i.e. the weights
        associated with the socio-eco / health-related attributes), then also
        recomputes the characteristics.
        WARNING: using this method on a model that is being run might
        result in unpredictable behavior.
        Parameters
        ----------
        param_name: str, name of the parameter to set.
        value: new value for the parameter.
        population_dataset: pandas DataFrame, optional. Dataset containing the
            agents' attributes. Must be given when param_name is either "test_params" or
            "inf_params".
        """
        # Remark: self.params and self.population.params point towards the same object.
        # Hence, we're also changing self.population.params by doing this:
        self.params[param_name] = value
        # If the param that was changed is among the weights for the level of infection
        # or for the test interest, then we must re-compute the scalar products (that are pre-computed).
        if param_name in ["test_params", "inf_params"]:
            if population_dataset is None:
                raise ValueError(
                    "population_dataset must be given if param_name=",
                    param_name)
            self.population.compute_agents_characteristics(population_dataset)

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
        self.period, self.day = 0, 0
        # Initializes an empty results manager
        self.results = ABM_Results(self.n_periods)
        # Initializes an empty events queue, which will be used for example
        # to schedule the changes of mobility.
        self.event_queue = EventQueue()
        # Resets the RNG
        if seed is None:
            seed = self.seed
        self.set_seed(seed)

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
        spend as infected.
        IMPORTANT: This method assumes the agents are susceptible, so it should be checked
        beforehand !
        Parameters
        ----------
        agent_ids: np array, IDs of the newly infected agents.
        """
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
        - Filters the "susceptible" agents so that only they can become
          infected.
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
            # If the infections are forced, a random set of agents is selected to become infected.
            infected_agents = self.random_infections(forced_infection_proba)
            infection_levels = np.zeros(self.n_agents)
        else:

            # === Infection level computation =========
            # Retrieves the facilities that the agents are currently located at.
            locations = self.population.mobility.get_locations(self.period)
            # Retrieves the number of visitors within each of those facilities
            visitors = self.population.mobility.get_visitors(self.period)[
                locations]
            # Retrieves the number of infected visitors within each facility
            infected_visitors = \
                self.population.mobility.get_infected_visitors(self.period)[
                    locations]
            # We can now compute the fraction of infected visitors within each facility
            infected_fractions = infected_visitors / visitors

            # Computation of the levels of infection:
            error_term = self.params['inf_lvl_error_term']
            # Term related to the fraction of infected visitors at the facility
            fraction_term = self.params[
                                'inf_fraction_param'] * infected_fractions
            # Scalar product between the betas and the social, economic and sanitary characteristics
            attributes_term = self.population.pop_inf_characteristics
            infection_levels = fraction_term + attributes_term + error_term

            # === Probability of infection ===========
            infection_probas = sigmoid(
                self.params['inf_proba_sigmoid_slope'] * infection_levels)

            # === Selection of the infected agents ===
            # Randomly draws which agents will be infected based on their infection probability.
            # This generates a boolean mask of shape (n_agents), with True meaning the agent becomes infected.
            infected_mask = self.rng.random(self.n_agents) < infection_probas

            # === Post-processing ====================
            # Agents that are in the facility 0 are actually considered confined
            infected_mask &= locations != 0

            # Converts the infection mask into an array containing the IDs of the newly infected
            # agents
            infected_agents = np.where(infected_mask)[0]

        # Finally: make sure that only "susceptible" agents can become infected:
        infected_agents = self.population.get_subset_in_state(infected_agents,
                                                              "susceptible")

        return infection_levels, infected_agents

    def process_testing(self, infection_levels):
        """
        Processes the testing policy:
        - Computes the Test Interest (TI) of every agent;
        - Deduces the probability of being tested;
        - From those probabilities, randomly selects which agents will be tested;
        - Based on the agents' states and the precision / recall of the tests, selects which
          agents are tested positive.
        Parameters
        ----------
        infection_levels: float np array of shape (n_agents,). Level of infection of every
            agent, computed previously during this period.
        Returns
        -------
        A pair (tested, tested_pos):
        - tested is an array containing the IDs of the agents that were tested;
        - tested_pos is a sub-array of tested, containing the IDs of the agents that were tested positive.
        """
        # Step 1: compute the Test Interest of every agent
        # Term related to the level of infection
        inf_lvl_term = self.params['test_inf_lvl_param'] * infection_levels
        # Term related to the social, economic and sanitary characteristics
        attributes_term = self.population.pop_test_characteristics
        error_term = self.params['test_error_term']
        test_interest = inf_lvl_term + attributes_term + error_term

        # Step 2: deduce the probabilities of test
        test_probas = sigmoid(
            self.params['test_proba_sigmoid_slope'] * test_interest)

        # Step 3: draw which agents are tested based on the test probabilities
        tested_boolean = self.rng.random(self.n_agents) < test_probas
        tested = np.where(tested_boolean)[0]

        # Step 4: compute the probability of being tested positive
        # TODO: include the reliability of tests in the computation
        # for now, an infected agent who is tested is always tested positive.
        tested_pos = self.population.get_subset_in_state(tested, "infected")

        return tested, tested_pos

    def process_recovery(self):
        """
        Processes the recovery policy during each period:
        - For every infected agent, reduces their infection time by 1 period;
        - For agents whose timer has reached 0, sets them to "recovered".
        """
        # First: retrieve the IDs of all currently infected agents
        infected_agents = self.population.get_infected_agents()
        # for all agents whose infection time has reached zero, they recover
        recovering_agents = infected_agents[
            self.infection_timer[infected_agents] == 0]
        self.population.set_agents_state(recovering_agents, "recovered")
        # for all still infected agents, reduce their recovery time by one period
        self.infection_timer[infected_agents] -= 1

    def reduce_mobility(self, agent_ids=None, activity_types=None, duration_days=0, duration_periods=0):
        """
        Reduces the mobility of a set of agents over given activity types, for a given duration.
        Parameters
        ----------
        agent_ids: ndarray of integers optional. IDs of the targeted agents.
            If None (default), all agents will be affected.
        activity_types: str or list of str, optional. Indicates which activity type(s) should
            be targeted.
            Defaults to None, which means activities of any type will be changed.
        duration_days: integer, optional. Duration of the reduction in days. Defaults
            to zero, but cannot be equal to zero if duration_periods is 0 or not specified.
        duration_periods: integer, optional. Duration of the reduction in periods.
            Must be strictly inferior to the number of periods within a day. Defaults
            to zero, but cannot be equal to zero if duration_days is 0 or not specified.

        Raises
        -------
        ValueError: if duration_periods == duration_days == 0, or if duration_periods is superior
            or equal to the number of periods within a day.
        """
        if duration_days == duration_periods == 0:
            raise ValueError(
                "duration_days and duration_periods cannot be unspecified or equal"
                " to zero at the same time.")
        if duration_periods >= self.n_periods:
            raise ValueError(
                f"duration_periods cannot be superior to the number of periods "
                f"within a day (got {duration_periods} but n_periods={self.n_periods}).")
        # If the agents were not specified, create an array "agent_ids" containing all IDs:
        if agent_ids is None:
            agent_ids = np.arange(self.n_agents)
        # First: reduce the mobility of the targeted agents using the methods of Population:
        # "confining" an agent is equivalent to setting its location to 0:
        self.population.change_agent_locations(agent_ids, 0)
        # Second: compute the date (day and period) on which the reduction will end:
        end_period = (self.period + duration_periods) % self.n_periods
        end_day = self.day + duration_days + (
                self.period + duration_periods) // self.n_periods
        # Third: add an event to the event queue to indicate that the reduction has to be lifted.
        # The info relative to the events is only the IDs of the targeted agents.
        self.event_queue.add_event(end_day, end_period,
                                   EventType.MOBILITY_RESET,
                                   [agent_ids])

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
            # === Events processing ===================
            # An example of event is a group of agents coming out of quarantine
            self.process_events()

            # === Recovery process ====================
            self.process_recovery()

            # === Infection process ===================
            if n_forced_infections is not None:
                # The forced infections are uniformly spread over the periods. To do so,
                # we need to compute the proba for every agent, during each period, to be
                # randomly selected:
                infection_proba = n_forced_infections / (
                        self.n_periods * self.n_agents)
                infection_levels, infected_agents = self.process_infections(
                    forced_infection_proba=infection_proba)
            else:
                # Computes the levels of infection, and selects the agents which become infected
                infection_levels, infected_agents = self.process_infections()
            # The following sets the selected agents as "infected"
            # and initiates their recovery process.
            self.set_infected(infected_agents)

            # === Testing process =====================
            tested, tested_pos = self.process_testing(infection_levels)

            # === Storing per-period results ==========
            # Number of new infections
            self.results.store_per_period("new infections",
                                          infected_agents.shape[0])
            # Number of currently infected agents
            self.results.store_per_period("infected agents",
                                          self.population.get_state_count(
                                              "infected"))
            # Number of currently recovered agents
            self.results.store_per_period("recovered agents",
                                          self.population.get_state_count(
                                              "recovered"))
            # Number of tested agents
            self.results.store_per_period("tests", tested.shape[0])
            # Number of positive tests
            self.results.store_per_period("positive tests", tested_pos.shape[0])
            # IDs of the currently infected agents
            self.results.store("infected agents IDs",
                               self.population.get_infected_agents())
            # IDs of the newly infected agents
            self.results.store("newly infected agents IDs",
                               infected_agents)

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
        self.initialized = True
        # Iterate over a few days, with the forced infections option enabled
        for n_forced_infections in forced_infections:
            self.iterate_day(n_forced_infections=n_forced_infections)

    def run_simulation(self, n_days, verbose=False):
        """
        Runs the simulation. Supposes that the model has already been initialized with
        force_simulation_start.
        Parameters
        ----------
        n_days: int, number of simulation days to perform.
        verbose: boolean, optional (Default: False). Whether to print progress information.
        Raises
        ------
        ValueError: if the model has not been initialized before calling this function.
        """
        if not self.initialized:
            raise ValueError(
                "Please initialize the model using force_simulation_start() before running the simulation. ")
        for day in range(n_days):
            if verbose:
                print("Day ", day)
            self.iterate_day()
        if verbose:
            print("Simulation ended. ")

    def process_events(self):
        """
        Retrieves the events of the current simulation day and period,
        and processes their effects.
        """
        events = self.event_queue.poll_events(self.day, self.period)
        for event_type, event_info in events:
            # Applies a different treatment depending on the event type
            if event_type == EventType.MOBILITY_RESET:
                # This event type corresponds to resetting the mobility of some agents,
                # e.g. when they come out of quarantine.
                # For this type of event, the info is the targeted agents' IDs.
                agent_ids = event_info
                self.population.reset_agent_locations(agent_ids)
