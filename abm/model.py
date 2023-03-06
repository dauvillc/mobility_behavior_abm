"""
Clément Dauvilliers - EPFL TRANSP-OR Lab - 12/12/2022
Implements an object-oriented Agent-based model for the COVID-19 epidemic
in Switzerland.
"""
import numpy as np
import pandas as pd
import abm.characteristics as ch
from copy import deepcopy
from abm.math_tools import compute_lognormal_params


class ABM:
    """
    Implements the Agent-Based Model.
    """

    def __init__(self, params, activities_data,
                 population_dataset=None,
                 pop_inf_characteristics=None,
                 pop_test_characteristics=None,
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

        # Load the activity matrices, in the COO format; And the
        # location of every agent. Note: any modification, e.g. for activity
        # reduction, must be made on a copy of those.
        self.visit_matrices_coo, self.agent_facilities_original = activities_data
        self.agent_facilities = self.agent_facilities_original

        # Deduces some size constants of the simulation
        self.n_facilities, self.n_agents = self.visit_matrices_coo[0].shape
        self.n_periods = len(self.visit_matrices_coo)

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
        # For the probability of infection
        if pop_inf_characteristics is None:
            self.inf_characs = ch.compute_characteristics(self.population, self.params['inf_params'])
        else:
            self.inf_characs = pop_inf_characteristics

        # For the probability of being tested
        if pop_test_characteristics is None:
            self.test_characs = ch.compute_characteristics(self.population, self.params['test_params'])
        else:
            self.test_characs = pop_test_characteristics

        # DEFAULTS
        self.set_default_param("recovery_mean_time", 8.0)
        self.set_default_param("recovery_std_time", 2.0)

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
        self.infected_mask = infected_mask.copy()
        self.recovered_mask = np.full(self.n_agents, False)
        # Will contain the IDs of the agents who became infected during each period
        self.infected_ids = [np.where(self.infected_mask)[0]]

        # Will contain the infection time (in periods)
        self.infection_times = np.zeros(self.n_agents)
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

    def apply_infections(self, period):
        """
        Computes the new infections based on the visits and a threshold. Any agent
        whose Level of Infection overcomes the threshold will become infected.
        Parameters
        ----------
        period: Integer, index of the period to process.

        Returns
        -------
        A pair:
        - A float array of shape (n_agents) giving the probability that had every
            had agent to become infected;
        - A new boolean array of shape (n_agents) such that NI[i] is True if and only
            if agent i became infected during this period.
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
        # If at least one agent is at an unknown facility, then we'll get inf_facilities[0] == -1
        # Which would modify the number of infected visitors at the last facility because of
        # backwards indexing. Thus, if there is a '-1' facility, we need to remove it:
        if inf_facilities.shape[0] > 0 and inf_facilities[0] == -1:
            inf_facilities, inf_counts = inf_facilities[1:], inf_counts[1:]
        # Adds this count to the zeros, then divide by the number of visitors to obtain
        # the fractions
        infected_fractions[inf_facilities] += inf_counts
        np.divide(infected_fractions, visitors_counts,
                  where=visitors_counts > 0, out=infected_fractions)

        # Compute the level of infection for every agent
        agent_facilities = self.agent_facilities[period]

        # Computes the infection probabilities. The infection probability is defined as
        # P(inf, i) = F(f(i)) * (Beta_base + characs)
        # where characs = beta1X1 + beta2X2 + ...
        # and F(f(i)) is the fraction of infected visitors at the facility that i is at.
        inf_visitors_fracs = infected_fractions[agent_facilities]
        beta_base = self.params['inf_fraction_param']
        infection_probas = inf_visitors_fracs * (beta_base + self.inf_characs)

        # The proba could theoretically go beyond 1, so we'll manually cap it.
        infection_probas[infection_probas > 1] = 1

        # Infects the agents with the just calculated probabilities
        new_inf_mask = self.rng.random(self.n_agents) < infection_probas
        # Agents that are in the facility "-1" are actually at an unknown facility
        new_inf_mask = new_inf_mask & (agent_facilities > -1)
        # Agents that are already infected cannot become infected
        new_inf_mask = new_inf_mask & (~self.infected_mask)
        # Agents that have recovered cannot become infected again
        new_inf_mask = new_inf_mask & (~self.recovered_mask)
        return infection_probas, new_inf_mask

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
        # Compute the mean and std of the underlying normal distrib
        mean_n, std_n = compute_lognormal_params(mean, std)
        times = self.rng.lognormal(mean=mean_n, sigma=std_n, size=new_inf_ids.shape)
        # The times were drawn in days, but we need to convert them to periods
        times *= self.n_periods
        self.recovery_times[new_inf_ids] = times

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

    def apply_testing(self, daily_inf_probas):
        """
        Applies a testing model over the full population. The method
        updates the self.testing_days array, which contains the number of
        days since the agents were last tested positive (or -1 otherwise).
        Parameters
        ----------
        daily_inf_probas: Float array of shape (n_agents). For an agent i,
            daily_inf_probas[i] must describe the probability that agent i
            has had over the whole day of becoming infected. For example, it can
            be the average of the probability of being infected over all periods.
        Returns
        -------
        A pair of integers
        (Number of tests performed,
         Number of positive tests)
        """
        # The probability of being tested is defined as
        # P(test) = Beta_base + Beta_proba_inf * P(inf) + beta1 * X1 + beta2 * X2 + ...
        # The second product is so that an agent that has had a high probability of being infected
        # (for example because they've had a risky behavior) is more likely to get tested.
        # X1, X2, .. are socio-economic and health characteristics.
        probas = np.full(self.n_agents, self.params['base_test_proba'])
        probas += self.params['test_inf_proba_factor'] * daily_inf_probas

        # Adds the characteristics (obtained from the socio-eco and health) to the probabilities
        if len(self.test_characs) > 0:
            probas += self.test_characs

        # Randomly draws which agents will be tested, according to their
        # probabilities
        draw = self.rng.random(self.n_agents)
        tested = draw < probas
        # For the returned number of positive tests, we need to take all tests into account,
        # include people who are already infectious or were infected before.
        # However, we only set to 0 the testing times of people who have not been tested
        # positive before.
        not_tested_before = self.testing_days == -1
        tested_positive = tested & self.infected_mask
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
        if 'apply_activity_reduction' in self.params and self.params['apply_activity_reduction']:
            self.apply_activity_reduction()

        # Number of infections over the day
        daily_infections = 0
        # This will contain the average probability of being infected over the day:
        # after every period, sum the probability of being infected during that period;
        # At the end of the day, divide by the number of periods.
        daily_inf_probas = np.zeros(self.n_agents)
        # Actual simulation of all periods within the day
        for period in range(self.n_periods):
            # Simulates the recoveries
            self.apply_recovery()
            # If a number of new infections is forced, we randomly draw some agents
            # to become infected.
            if force_infections is not None:
                # The proba of becoming infected is uniform over all agents, and is divided
                # by the number of periods to amount for force_infections over the full day.
                inf_probas = (force_infections / self.n_periods) / self.n_agents
                new_inf_mask = self.rng.random(self.n_agents) < inf_probas
                new_inf_mask = new_inf_mask & (~self.infected_mask) & (~self.recovered_mask)
            else:
                # Simulates the infections
                inf_probas, new_inf_mask = self.apply_infections(period)

            # Adds the probabilities of infection to the daily sum
            daily_inf_probas += inf_probas

            # Saves the IDs of agents who've just become infected
            new_inf_ids = np.where(new_inf_mask)[0]
            self.infected_ids.append(new_inf_ids)
            # Computes the number of newly infected people
            self.new_infections.append(new_inf_ids.shape[0])
            daily_infections += new_inf_ids.shape[0]

            # Updates the infection times
            self.infection_times[self.infected_mask] += 1
            # Updates the infection mask
            self.infected_mask |= new_inf_mask

            # Draws the recovery times for the newly infected people
            self.draw_recovery_times(new_inf_ids)

            # Period update
            self.period += 1

        # Converts the sum [P_p(inf) over all periods p] to the mean
        daily_inf_probas /= self.n_periods

        # Applies the testing policy
        n_tests, n_positive_tests = self.apply_testing(daily_inf_probas)

        # Memorizes the day's outcome: infections, positive tests
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
        Three lists:
            - The daily number of new infections;
            - The daily number of tests performed;
            - The daily number of positive tests.
        """
        self.n_days += days
        for day in range(days):
            if verbose:
                print(f"Day {day}        ", end="\r")
            self.iterate_day()
        print("Simulation ended")

        return self.daily_new_infections, self.daily_tests, self.daily_positive_tests

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
