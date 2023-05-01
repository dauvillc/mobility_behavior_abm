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


def run_model(model, initial_infections, days, seed, param_variations, population_dataset):
    """
    Runs a given ABM.
    Parameters
    ----------
    model: ABM Object to run.
    days: Number of simulation days to process.
    initial_infections: list or array-like of integers. Number of new
        infections to force for the first days of the simulation.
        initial_infections[d] should be an integer giving the number of
        new infections occurring on day d.
    seed: random seed to use for the model.
    param_variations: dictionary {param_name: value} that indicates changes that should
        be applied to the model before beginning.
    population_dataset: pandas DataFrame containing the agents' attributes.

    Returns
    -------
    The Results object from the run model.
    """
    # Apply the potential parameter changes
    for param, value in param_variations.items():
        model.set_param(param, value, population_dataset)
    # Sets the model's seed
    model.set_seed(seed)
    # Sets the model's initial conditions
    model.force_simulation_start(initial_infections)
    # Runs the simulation
    model.run_simulation(days)
    return model.results


class ParallelABM:
    """
    A ParallelABM class is designed to run multiple parallel simulation.
    Internally, only a single ABM object is created. When the parallel simulations
    are run, a copy of that model is created by each process. The copies each have their
    own random seed.
    """

    def __init__(self, params, activity_data=None, n_models=1, seed=42):
        """

        Parameters
        ----------
        params: Dictionary of parameters to give to the model,
            such as the recovery rate.
        activity_data: optional, Triplet (N, LV, LF) as returned by contacts.load_period_activities().
            - N is the pair of integers (number of agents, number of facilities).
            - LF is the list of locations of all agents during each period.
            - LT is the list of activity types of all activities during each period.
            If not given, will be loaded from the default location in simulation_config.yml.
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
        # When the ABM() constructor is called outside the ParallelABM class,
        # it does not keep the population dataset in memory. In fact this dataset is large,
        # so it wouldn't be great to store it within the ABM or the Population, since those
        # get replicated by each parallel process.
        # Instead, we load the population dataset only once, and store it only here in the
        # ParallelABM object. Thus, it won't be copied by the new processes.
        print("Loading the population dataset in the master..")
        self.population_df = ch.load_population_dataset()
        print("Done")

        # Creates random seeds for every model, which depend on the master seed
        # for reproducibility
        print(f"Assembling the model...")
        self.seeds = self.rng.integers(0, 100, self.n_models)
        # Internally, only a single ABM object is created. When the parallel simulations
        # are run, a copy of that model is created by each process. The copies each have their
        # own random seed.
        self.model = ABM(params,
                         activity_data=activity_data,
                         population_dataset=self.population_df,
                         seed=seed)
        print("Done")

        # Initializes the object, including the computation of the agents characteristics
        print("Initializing the ParallelABM...")
        self.reset()
        print("Done")

    def reset(self):
        """
        Resets the ParallelABM object, to make it ready to run new simulations
        (for example, after some parameters have been modified), without reloading
        more data than needed.
        """
        # Tells the ABM to re-compute the agents' characteristics. Those depend on the parameters, so their
        # values might have changed. We also need to provide the population dataset, which is stored in this class:
        self.model.population.compute_agents_characteristics(self.population_df)

        # Resets the list of varying parameters (see set_varying_param() )
        self.param_variations = [dict() for _ in range(self.n_models)]

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
        self.params[param_name] = value
        self.model.set_param(param_name, value, population_dataset)

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
        # Internally, the ParallelABM object only has a single ABM object,
        # which is then copied by the parallel processes. As a result we cannot
        # directly set multiple values for a parameter.
        # Instead, we will need to set the varying parameter within the other
        # processes, i.e. within the run_model() function (the one is that is
        # called by the multiprocessing package).
        # To do so, we'll structure that will let us remember which param needs
        # to vary between the models. That structure is a list L of dictionaries,
        # such that if L[i] = {'x': 1.3}, then model number i will have the param 'x'
        # set to 1.3.
        for n_model in range(self.n_models):
            self.param_variations[n_model][param_name] = values[n_model]

    def run_simulations(self, initial_infections, days):
        """
        Runs the simulations in parallel.
        Parameters
        ----------
        initial_infections: list or array-like of integers. Number of new
            infections to force for the first days of the simulation.
            initial_infections[d] should be an integer giving the number of
            new infections occurring on day d.
        days: Number of simulation days.
        """
        print(f"Starting {self.n_models} parallel simulations...")
        with mp.Pool(processes=self.n_models) as pool:
            self.results = pool.starmap(run_model, [(self.model, initial_infections, days,
                                                     seed, param_variations, self.population_df)
                                                    for seed, param_variations in
                                                    zip(self.seeds, self.param_variations)
                                                    ])
        print("Simulations ended")

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
