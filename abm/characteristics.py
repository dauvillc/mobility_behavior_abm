"""
Clément Dauvilliers - EPFL Transp-or - 17/01/2023
Implements functions to compute the agents' characteristics.
The characteristics are defined as functions of the agents' attributes, such
as the age, and their associated free parameters (thetas).
The characteristics have specific purposes: the impact of a given attribute
may not be the same for the recovery rate as for the level of infection. Therefore,
separate characteristics have to be defined for each purpose;
"""
import numpy as np
import pandas as pd
import yaml


def load_population_dataset():
    """
    Loads the population dataset, which contains for every agent ID the attributes
    (e.g. geographic coordinates of home, social and economic characteristics, health...).
    The population dataset's path needs to be specified in simulation_config.yml as:
    paths:
      population_dataset: 'path'
    Returns
    -------
    A pandas DataFrame indexed by agent ID.
    """
    with open("simulation_config.yml") as cfg:
        simu_config = yaml.safe_load(cfg.read())
        population_df = pd.read_csv(simu_config['paths']['population_dataset'], index_col='agent_index')
        population_df = population_df.sort_index()
        return population_df


def compute_characteristics(attributes, params):
    """
    Given the attributes (e.g. socio-economic, health-related) of the agents
    and the associated parameters (betas), computes their scalar product.
    Parameters
    ----------
    attributes: DataFrame of shape (n_agents, n_characteristics); population characteristics. Each column
        is an attribute, such as "age".
    params: map {attribute: param_value} whose keys are the attributes to use (e.g., 'age') and
        the values are the betas (float values).

    Returns
    -------
    An array C of shape (n_agents), such that C[i] is the
    characteristic of agent i regarding their probability of infection.
    """
    names = list(params.keys())
    betas = np.array(list(params.values()))
    # Retrieves the data associated with the desired attributes
    attrib_data = attributes[names].to_numpy()
    characs = attrib_data @ betas
    # If only one param is used, then the "characs" is already 1-dimensional (beta * X),
    # otherwise we need to sum over the params (betaX1 + betaX2 + betaX3 + ..)
    if len(names) >= 2:
        characs = characs.sum(axis=1)
    return characs
