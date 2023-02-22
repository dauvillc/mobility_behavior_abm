"""
Clément Dauvilliers - EPFL TRANSP-OR Lab - 12/12/2022
Implements an object-oriented Agent-based model for the COVID-19 epidemic
in Switzerland.
"""
import numpy as np
import pandas as pd
import abm.characteristics as ch


class ABM:
    """
    Implements the Agent-Based Model.
    """

    def __init__(self, params, activities_data,
                 population_dataset=None,
                 pop_inf_characteristics=None,
                 visitors_counts=None,
                 seed=42):
        """

        Parameters
        ----------
        params: Dictionary of parameters to give to the model,
            such as the recovery rate.
        activities_data: Pair (LV, LF) as returned by contacts.load_period_activities().
            - LV is the list of sparse visit matrices for every period;
            - LF is the list of locations of all agents during each period.
        population_dataset: DataFrame, optional. Dataset containing the agents' attributes.
            If None, it will be loaded.
        pop_inf_characteristics: Float array of shape (n_agents), optional. Values for the
            characteristics of all agents regarding the probability of infection. If None,
            it will be computed using the parameters.
        visitors_counts: List of Integer arrays of shape (n_facilities), optional. Number of visitors
            per period, per facility, not accounting for activity reduction. If None, it will
            be computed from the activity matrix.
        seed: Random seed to use.

        Returns
        -------

        """
        self.params = params
        self.seed = seed
        self.rng = np.random.default_rng(seed=seed)

        # Load the activity matrices, in the COO format; And the
        # location of every agent. Note: any modification, e.g. for activity
        # reduction, must be made on a copy of those.
        self.visit_matrices_coo, self.agent_facilities_original = activities_data
        self.agent_facilities = self.agent_facilities_original

        # Deduces some size constants of the simulation
        self.n_facilities, self.n_agents = self.visit_matrices_coo[0].shape
        self.n_periods = len(self.visit_matrices_coo)

        self.include_exposed_state = 'include_exposed_state' in self.params and \
                                     self.params['include_exposed_state']

        # Computes the original number of visitors by facility, for every period
        if visitors_counts is None:
            self.visitors_counts = [v.sum(axis=1) for v in self.visit_matrices_coo]
        else:
            self.visitors_counts = visitors_counts

        # Loads the population socio-eco attributes if required
        if population_dataset is None:
            print("Loading the population dataset..")
            self.population = pd.read_csv('data/abm/vaud/extracted/vaud_population.csv.gz', index_col='agent_index')
            self.population = self.population.sort_index()
            print("Done")
        else:
            self.population = population_dataset

        # Builds the agents' characteristics if required
        if pop_inf_characteristics is None:
            # Isolates the socio-economic attributes
            self.soceco = self.population[self.params['soceco_attributes']].to_numpy()
            print("Computing population characteristics...")
            self.inf_characs = ch.compute_inf_characteristics(self.soceco, self.params['soceco_params'])
            print("Done")
        else:
            self.inf_characs = pop_inf_characteristics

    def init_simulation(self, infected_mask, seed=None):
        """
        (Re)initializes the simulation, while conserving the model's
        parameters. Can be called either in the class constructor or
        to begin a new simulation without changing the parameters.
        Parameters
        ----------
        infected_mask: Boolean array of shape (n_agents), indicating
            which agent is infected at the beginning of the simulation.
        seed: Random seed to use. If none is specified, will default to the ABM's original
            seed.

        Returns
        -------
        """
        # Initialize general vars about the simulation
        self.period, self.n_days = 0, 1

        # Initialize the agents' status
        self.exposed_mask = np.full(self.n_agents, False)
        self.infected_mask = infected_mask.copy()
        self.recovered_mask = np.full(self.n_agents, False)

        # Will contain the IDs of the agents who became infected during each period
        self.infected_ids = [np.where(self.infected_mask)[0]]
        # Will contain the infection time (in periods)
        self.infection_times = np.zeros(self.n_agents)
        # Will contain the time spent in the 'exposed' state
        self.exposed_times = np.zeros(self.n_agents)
        # Draws the exposition times if we do use an Exposed state
        self.exposition_times = np.zeros(self.n_agents)
        if self.include_exposed_state:
            self.draw_exposition_times()

        # Will contain the recovery times
        self.recovery_times = np.zeros(self.n_agents)
        # Will contain the number of days since the agent was first tested positive.
        # A value of -1 indicates the agent was never tested positive.
        self.testing_days = np.full(self.n_agents, -1, dtype=int)
        # Will contain the number of people of have been tested positive for the first time
        # each day.
        self.daily_positive_tests = [0]
        self.daily_tests = [0]

        # Initialize the lists that will store info about the simulation
        initial_infections = self.infected_ids[-1].shape[0]
        self.new_infections = [initial_infections]
        self.daily_new_infections = [initial_infections]
        self.recoveries = [0]

        # Resets the RNG
        if seed is None:
            seed = self.seed
        self.rng = np.random.default_rng(seed=seed)

    def set_param(self, param_name, value):
        """
        Sets the value of a given simulation parameter.
        Parameters
        ----------
        param_name: String, name of the parameter.
        value: new value for the parameter.

        Returns
        -------

        """
        self.params[param_name] = value
        if param_name == 'include_exposed_state':
            raise ValueError('To change the states, please instantiate a new ABM')

    def apply_infections(self, period):
        """
        Computes the new infections based on the visits and a threshold. Any agent
        whose Level of Infection overcomes the threshold will become infected.
        Parameters
        ----------
        period: Integer, index of the period to process.

        Returns
        -------
        A new boolean array of shape (n_agents) such that NI[i] is True if and only
        if agent i became infected during that period.
        """
        # The following block computes the fractions of infected visitors at each facility.
        infected_fractions = np.zeros(self.n_facilities, dtype=np.float)
        # Begins by loading the number and infected and non-infected visitors
        visitors_counts = self.visitors_counts[period]
        # Retrieves the facilities at which the infected agents are located
        infected_agents_locations = self.agent_facilities[period][self.infected_mask]
        # Counts at which facilities there are infected agents, and for those facilities
        # also counts them
        inf_facilities, inf_counts = np.unique(infected_agents_locations, return_counts=True)
        # Adds this count to the zeros, then divide by the number of visitors to obtain
        # the fractions
        infected_fractions[inf_facilities] += inf_counts
        np.divide(infected_fractions, visitors_counts,
                  where=visitors_counts > 0, out=infected_fractions)

        # Compute the level of infection for every agent
        agent_facilities = self.agent_facilities[period]

        # Computes the infection probabilities. The infection probability is defined as
        # Pinf(i) = min {Pinf,0(i) + Prisks(i), 1}
        # where Pinf(i) is a base infection probability depending on the contacts of i, and
        # Prisks(i) depends on additional risks linked to i's health conditions and behavior.

        # The base risk of infection of an agent A is the fraction of infected agents at the facility
        # where A currently is. That fraction is multiplied by a scaling parameter. For example, 0.8 for the
        # parameter means 100% of infected contacts leads to a base infection probability of 1.
        base_probas = infected_fractions[agent_facilities] * self.params['inf_fraction_param']
        risk_probas = np.tanh(self.inf_characs)
        infection_probas = base_probas * (1 + risk_probas)
        infection_probas[infection_probas > 1] = 1

        # Exposes the agents with the just calculated probabilities
        new_exp_mask = self.rng.random(self.n_agents) < infection_probas
        # Agents that are in the facility "-1" are actually at home
        new_exp_mask = new_exp_mask & (agent_facilities > -1)
        # Agents that are already infected cannot become infected
        new_exp_mask = new_exp_mask & (~self.infected_mask)
        # Agents that have recovered cannot become infected again
        new_exp_mask = new_exp_mask & (~self.recovered_mask)
        return new_exp_mask

    def apply_inf_transition(self, new_exp_ids):
        """
        Applies the disease state transitions for every agent.
        Parameters
        ----------
        new_exp_ids: array of shape (n_new_infected) giving the ids
        of agents that have just been exposed.
        Returns
        -------
        An array giving the indices of agents that have become Infectious.
        """
        # Updates the times for which agents have been exposed already
        self.exposed_times[self.exposed_mask] += 1
        # All agents that have been exposed for long enough
        # become Infectious.
        new_inf_ids = np.where(self.exposed_times >= self.exposition_times)[0]
        # All agents that become Infectious are not Exposed anymore
        self.exposed_mask[new_inf_ids] = False

        return new_inf_ids

    def draw_recovery_times(self, new_inf_ids: np.array):
        """
        For the agents that have just become infected, randomly draws
        the times they will need to recover.
        Parameters
        ----------
        new_inf_ids: 1D array containing the IDs of the agents that have just
            become infected.
        """
        mean, std = self.params['recovery_mean_time'], self.params['recovery_std_time']
        # The mean and std parameters are given in days, but we need them in periods
        mean, std = mean * self.n_periods, std * self.n_periods
        times = self.rng.lognormal(mean, std, size=new_inf_ids.shape)
        self.recovery_times[new_inf_ids] = times

    def draw_exposition_times(self):
        """
        Draws an exposition time (in periods) for every agent. That time shall
        be used if the agent ever becomes infected during the simulation.
        Returns
        -------
        An integer array of shape (n_agents) containing the exposition times. The value
        of self.exposition_times is also set by this method.
        """
        mean, std = self.params['exposition_mean_time'], self.params['exposition_std_time']
        times = self.rng.lognormal(mean, std, self.n_agents)
        # The times were drawn in days, but we need them in periods
        times *= self.n_periods

        self.exposition_times = times
        return times

    def apply_recovery(self):
        """
        Applies the recovery process by setting to "recovered" every infected
        agent that has reached their recovery time.
        Parameters
        ----------
        period: current period of the simulation.
        """
        recovered_indexes = np.where(self.infected_mask & (self.infection_times > self.recovery_times))[0]
        self.infection_times[recovered_indexes] = 0
        self.recovered_mask[recovered_indexes] = True
        self.infected_mask[recovered_indexes] = False
        # Store the info about recoveries
        self.recoveries.append(recovered_indexes.shape[0])

    def apply_testing(self):
        """
        Applies a testing model over the full population. The method
        updates the self.testing_days array, which contains the number of
        days since the agents were last tested positive (or -1 otherwise).
        Returns
        -------
        A pair of integers
        (Number of tests performed,
         Number of positive tests)
        """
        # Every agent has a base probability to get tested each day.
        # Being infectious with symptoms increases the probability of
        # being tested, of a parameterized amount.
        probas = np.full(self.n_agents, self.params['base_test_proba'])
        probas[self.infected_mask] += self.params['inf_test_proba']

        # Randomly draws which agents will be tested, according to their
        # probabilities
        draw = self.rng.random(self.n_agents)
        tested = draw < probas
        if self.include_exposed_state:
            tested_positive = tested & (self.infected_mask | self.exposed_mask)
        else:
            tested_positive = tested & self.infected_mask
        # For the returned number of positive tests, we need to take all tests into account,
        # include people who are already infectious or were infected before.
        # However, we only set to 0 the testing times of people who have not been tested
        # positive before.
        not_tested_before = self.testing_days == -1
        new_tested_positive = tested_positive & not_tested_before
        self.testing_days[new_tested_positive] = 0

        n_tests = tested.sum()
        n_positive_tests = tested_positive.sum()
        return n_tests, n_positive_tests

    def apply_activity_reduction(self):
        """
        Applies the activity reduction policy.
        """
        # Isolates agents who haven't been tested positive less than 5 days ago.
        non_confined_agents = (self.testing_days < 0) | (self.testing_days > 5)
        return

    def iterate_day(self, force_infections=None):
        """
        Performs the simulation over a full day, which itself is divided
        into periods.
        Parameters
        ----------
        force_infections: Integer, optional. Forces a given number of new infections
            to approximately occur over the day. The new infections thus forced will be
            spread uniformly over the day's periods.
        Returns
        -------
        Two lists N and T:
        - N is the amount of new infections after each period;
        - T is the total number of infected people after each period.
        """

        # Applies the activity reduction policy
        if self.params['apply_activity_reduction']:
            self.apply_activity_reduction()

        # Actual simulation of all periods within the day
        daily_infections = 0
        for period in range(self.n_periods):
            # Simulates the recoveries
            self.apply_recovery()
            # If a number of new infections is forced, we randomly draw some agents
            # to become infected.
            if force_infections is not None:
                # The proba of becoming infected is uniform over all agents, and is divided
                # by the number of periods to amount for force_infections over the full day.
                inf_proba = (force_infections / self.n_periods) / self.n_agents
                new_exp_mask = self.rng.random(self.n_agents) < inf_proba
                new_exp_mask = new_exp_mask & (~self.exposed_mask) & (~self.recovered_mask)
            else:
                # Simulates the infections
                new_exp_mask = self.apply_infections(period)

            new_exp_ids = np.where(new_exp_mask)[0]
            self.exposed_mask |= new_exp_mask

            # === STATE TRANSITIONS ==== #
            if self.include_exposed_state:
                # If we use an Exposed state, applies the transitions
                # Exposed -> Susceptible
                new_inf_ids = self.apply_inf_transition(new_exp_ids)
            else:
                # If we don't use an Exposed state, all exposed agents
                # instantly become infected.
                new_inf_ids = new_exp_ids

            # Updates the infection times
            self.infection_times[self.infected_mask] += 1
            # Updates the infection mask (Transition Exposed -> Infectious)
            self.infected_mask[new_inf_ids] = True

            # Draws the recovery times for the newly infected people
            self.draw_recovery_times(new_inf_ids)

            # === SAVING RESULTS ==== #
            # Saves the IDs of agents who've been exposed to the disease during this period
            # Note: we actually consider an infection as "an agent enters the Exposed state",
            # not the Infectious state.
            self.infected_ids.append(new_exp_ids)
            # Computes the number of newly infected people
            self.new_infections.append(new_exp_ids.shape[0])
            daily_infections += new_exp_ids.shape[0]

            # Period update
            self.period += 1

        # Applies the testing policy
        n_tests, n_positive_tests = self.apply_testing()

        # Memorizes the day's outcome
        self.daily_new_infections.append(daily_infections)
        self.daily_positive_tests.append(n_positive_tests)
        self.daily_tests.append(n_tests)

        # For agents that have been tested positive, updates the number of days
        # since they were tested
        self.testing_days[self.testing_days > -1] += 1

        return self.new_infections[-self.n_periods:]

    def simulation(self, days, verbose=False):
        """
        Performs the simulation over a given number of successive days.
        Parameters
        ----------
        days: Integer, number of days to perform the simulation on.
        verbose: Boolean, whether to print information about the process.

        Returns
        -------
        Two lists:
            - The daily number of tests performed;
            - The daily number of positive tests.
        """
        self.n_days += days
        for day in range(days):
            if verbose:
                print(f"Day {day}        ", end="\r")
            self.iterate_day()
        if verbose:
            print("Simulation ended")

        return self.daily_tests, self.daily_positive_tests

    def force_simulation_start(self, daily_infections):
        """
        Initializes the model by forcing a given number of new infections
        for the first few days. This can be used while calibrating the model,
        to force the beginning of the simulation to fit with reality.
        The forced infections are drawn randomly among the agents, who then transit
        normally between disease states.
        Parameters
        ----------
        daily_infections: list or array-like of integers. Number of daily new
            infections to force. The number of forced simulation days will be
            len(daily_infections).
        """
        # The way this function works is by first calling init_simulation() with
        # the first forced infections. Then iterate_day() is called, with the
        # force_infections argument. That method takes advantage of the pre-existing
        # methods of the model, pretending that the forced infections are real one.
        n_forced_days = len(daily_infections)
        # We draw the very first agents to be infected
        initial_proba = daily_infections[0] / self.n_agents
        init_mask = self.rng.random(self.n_agents) < initial_proba
        self.init_simulation(init_mask)

        # Run the simulation for the next forced days
        self.n_days = n_forced_days
        for daily_forced_inf in daily_infections[1:]:
            self.iterate_day(force_infections=daily_forced_inf)

        return self.daily_positive_tests

    def get_weekly_cases(self):
        """
        Returns
        -------
        An array of shape (n_days / 7) giving the total amount of confirmed cases
        for every week. The first day is considered as the first day of the first week.
        The number of simulation days must be a multiple of 7 for this method to function.
        """
        return np.array(self.daily_positive_tests).reshape(-1, 7).sum(axis=1)
