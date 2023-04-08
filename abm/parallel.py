"""
Clément Dauvilliers - EPFL TRANSP-OR Lab - 27/01/2023
Implements the ParallelABM class, which allows to easily run multiple ABM simulations
simultaneously.
"""
import multiprocessing as mp
import numpy as np
import pandas as pd
import abm.characteristics as ch
from abm.model import ABM


def run_model(model, days):
    """
    Runs a given ABM.
    Parameters
    ----------
    model: ABM Object to run.
    days: Number of simulation days to process.

    Returns
    -------
    The Results object from the run model.
    """
    model.run_simulation(days)
    return model.results


class ParallelABM:
    """
    A ParallelABM class is designed to run multiple parallel simulation.
    """

    def __init__(self, params, activity_data, n_models=1, seed=42):
        """

        Parameters
        ----------
        params: Dictionary of parameters to give to the model,
            such as the recovery rate.
        activity_data: Triplet (N, LV, LF) as returned by contacts.load_period_activities().
            - N is the pair of integers (number of agents, number of facilities).
            - LF is the list of locations of all agents during each period.
            - LT is the list of activity types of all activities during each period.
        n_models: int or None, optional. Number of models to run in parallel. Defaults to 1.
            Must not exceed the number of CPU cores available.
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
        self.population = ch.load_population_dataset()
        print("Done")

        # Builds the agents' characteristics if required
        print("Computing population characteristics in the master...")
        self.inf_characs = ch.compute_characteristics(self.population, self.params['inf_params'])
        self.test_characs = ch.compute_characteristics(self.population, self.params['test_params'])
        print("Done")

        # Creates random seeds for every model, which depend on the master seed
        # for reproducibility
        print(f"Assembling {self.n_models} models ")
        self.seeds = self.rng.integers(0, 100, self.n_models)
        self.models = [ABM(params,
                           activity_data,
                           population_dataset=self.population,
                           pop_inf_characteristics=self.inf_characs,
                           pop_test_characteristics=self.test_characs,
                           seed=s) for s in self.seeds]
        print("Done")

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

    def run_simulations(self, days):
        """
        Runs the simulations in parallel.
        Parameters
        ----------
        days: Number of simulation days.
        """
        print(f"Starting {self.n_models} parallel simulations")
        self.n_days += days
        with mp.Pool(processes=self.n_models) as pool:
            # Launches each model by calling run_model(i) for model i
            self.results = pool.starmap(run_model, [(m, days) for m in self.models])
        print("Simulations ended")

    def force_simulation_start(self, daily_infections):
        """
        Initializes the models by forcing a certain number of infections for a given
        number of days.
        The forced infections are drawn randomly among the agents, who then transit
        normally between disease states.
        Parameters
        ----------
        daily_infections: list or array-like of integers. Number of daily new
            infections to force. The number of forced simulation days will be
            len(daily_infections).
        """
        self.n_days = len(daily_infections)
        print("Forcing initial infections...")
        for k, model in enumerate(self.models):
            print(f"Model {k + 1}/{self.n_models}")
            model.force_simulation_start(daily_infections)

    def get_results_dataframe(self, timestep="daily"):
        """
        Returns the result in the form of a DataFrame in long-format.
        Must be called after run_simulations().
        Parameters
        ----------
        timestep: Str, either "per-period" or "daily".
        Returns
        -------
        A DataFrame D whose columns are (simulation, vars...) where vars...
        is the results variables stored during the simulation, for the given timestep.
        """
        dataframes = []
        for n_sim, results in enumerate(self.results):
            # Retrieves the results dataframe for the Nth model
            if timestep == "per-period":
                results_df = results.get_per_period_results()
            else:
                results_df = results.get_daily_results()
            # Adds a column 'simulation' that indicates which simulation
            # these results come from
            results_df['simulation'] = np.full(results_df.shape[0], n_sim)
            dataframes.append(results_df)
        # Concatenates the results of every model into a single DF.
        # The results from the various models can still be separated thanks to
        # the "simulation" column.
        final_results = pd.concat(dataframes, axis=0)
        return final_results
