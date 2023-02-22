"""
Clément Dauvilliers - EPFL TRANSP-OR Lab - 27/01/2023
Implements the ParallelABM class, which allows to easily run multiple ABM simulations
simultaneously.
"""
import multiprocessing as mp
import numpy as np
import pandas as pd
import abm.characteristics as ch
from itertools import repeat
from abm.model import ABM


def run_model(model, days, verbose):
    """
    Runs a given ABM.
    Parameters
    ----------
    model: ABM Object to run.
    days: Number of simulation days to process.
    verbose: Boolean, whether to print progress information.

    Returns
    -------
    The simulation results.
    """
    return model.simulation(days, verbose=verbose)


class ParallelABM:
    """
    A ParallelABM object contains several ABM objects that can run in parallel, most
    notably to estimate the uncertainty over a set of parameters.
    """

    def __init__(self, params, activity_data, n_models=1, seed=42):
        """

        Parameters
        ----------
        params: Dictionary of parameters to give to the model,
            such as the recovery rate.
        activities_data: Pair (LV, LF) as returned by contacts.load_period_activities().
            - LV is the list of sparse visit matrices for every period;
            - LF is the list of locations of all agents during each period.
        n_models: int or None, optional. Number of models to run in parallel. Defaults to 1.
            If None is given, the maximum number of models will be used (equal to the number of
            CPU cores).
        seed: Random seed.
        """
        self.activity_data = activity_data
        self.params = params
        self.master_seed = seed
        self.rng = np.random.default_rng(seed)
        self.n_models = n_models
        self.results = None

        # ===== DATA LOADING ======================== #
        # The large datasets and characteristics computation can be shared between all models
        print("Loading the population dataset in the master..")
        self.population = pd.read_csv('data/abm/vaud/extracted/vaud_population.csv.gz', index_col='agent_index')
        self.population = self.population.sort_index()
        print("Done")

        # Builds the agents' characteristics if required
        print("Computing population characteristics in the master...")
        self.inf_characs = ch.compute_characteristics(self.population, self.params['inf_params'])
        self.test_characs = ch.compute_characteristics(self.population, self.params['test_params'])
        print("Done")

        # Creates random seeds for every model, which depend on the master seed
        # for reproducibility
        print("Assembling models ")
        self.seeds = self.rng.integers(0, 100, self.n_models)
        self.models = [ABM(params,
                           activity_data,
                           population_dataset=self.population,
                           pop_inf_characteristics=self.inf_characs,
                           pop_test_characteristics=self.test_characs,
                           seed=s) for s in self.seeds]
        print("Done")
        self.n_agents = self.population.shape[0]

    def set_param(self, param_name, value):
        """
        Sets the value of a given simulation parameter.
        Parameters
        ----------
        param_name: String, name of the parameter.
        value: new value for the parameter.
        """
        self.params[param_name] = value
        for model in self.models:
            model.set_param(param_name, value)

    def set_varying_param(self, param_name, values):
        """
        Sets varying values across the models, to make
        the parallel simulations differ.
        This method can be used to study the effect of a given
        parameter, while all others remain fix.
        Note that this functions only sets the varying parameter,
        but does not run the simulations.
        Parameters
        ----------
        param_name: Name of the parameter to study.
        values: iterable, containing successive values for the
            parameter to test. Each value will be attributed to
            one model, and all values will be run in parallel.
        """
        for param_val, model in zip(values, self.models):
            model.set_param(param_name, param_val)

    def init_simulation(self, initial_mask=None, initial_inf_proportion=None):
        """
        Initializes a new simulation in every ABM object.
        Parameters
        ----------
        initial_mask: Optional, boolean array of shape (n_agents), indicating which agents are infected
            at the beginning of the simulation. If not given, then initial_inf_proportion must be.
        initial_inf_proportion: optional, float. Proportion of agents that will be randomly infected
            at the start of every simulation.
        """
        self.n_days = 1
        for abm in self.models:
            if initial_mask is not None:
                abm.init_simulation(initial_mask)
            else:
                initial_mask = self.rng.random(self.n_agents) < initial_inf_proportion
                abm.init_simulation(initial_mask)

    def run_simulations(self, days, verbose=False):
        """
        Runs the simulations in parallel.
        Parameters
        ----------
        days: Number of simulation days.
        verbose: Boolean, whether to print progress information.

        Returns
        -------
        A List L of lists such that L[i] is the results of the ith simulation.
        The results of a simulation are a pair of lists:
            - Number of daily tests;
            - Number of daily positive tests.
        """
        print(f"Starting {self.n_models} parallel simulations")
        self.n_days += days
        with mp.Pool(processes=self.n_models) as pool:
            # Launches each model by calling run_model(i) for model i
            results = pool.starmap(run_model, [(m, days, verbose) for m in self.models])
        self.results = results
        self.daily_tests = [result_sim[0] for result_sim in self.results]
        self.daily_positive_tests = [result_sim[1] for result_sim in self.results]
        print("Simulations ended")
        return results

    def force_simulation_start(self, daily_infections, verbose=False):
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
        verbose: Boolean, whether to print progress information.
        """
        if verbose:
            print("Running forced simulations..")
        self.n_days = len(daily_infections)
        for model in self.models:
            model.force_simulation_start(daily_infections)
        if verbose:
            print("Done")

    def get_results_dataframe(self, timestep="daily"):
        """
        Returns the result in the form of a DataFrame in long-format.
        Must be called after run_simulations().
        Parameters
        ----------
        timestep: Str, either "daily" (default) or "weekly".
        Returns
        -------
        A DataFrame D whose columns are (n_simulation, daily/weekly, tests, pos tests).
        """
        dataframes = []
        for n_sim, abm in enumerate(self.models):
            df = pd.DataFrame({'simulation': [n_sim] * self.n_days,
                               'day': list(range(self.n_days)),
                               'total tests': self.daily_tests[n_sim],
                               'positive tests': self.daily_positive_tests[n_sim]})
            # Add the share of positive tests
            df['pos share'] = (df['positive tests'] / df['total tests'])
            # If "total tests" is zero, the resulting value is inf, but we consider 0 percent
            df.loc[~np.isfinite(df['pos share']), 'pos share'] = 0

            if timestep == 'weekly':
                df['week'] = df['day'] // 7
                df = df.groupby(['simulation', 'week'])[['tests', 'positive tests']].sum().reset_index()
            dataframes.append(df)
        return pd.concat(dataframes, axis=0)
