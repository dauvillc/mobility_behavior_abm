"""
Clément Dauvilliers - April 1st 2023
Tests the Population class
"""
import numpy as np
import pandas as pd
import abm.characteristics as ch
from abm.contacts import load_period_activities
from abm.population import Population

if __name__ == "__main__":
    # Loads the population socio-eco attributes
    print("Loading the population dataset..")
    population_df = pd.read_csv('data/abm/vaud/extracted/vaud_population.csv.gz', index_col='agent_index')
    population_df = population_df.sort_index()
    print("Done")
    activity_data = load_period_activities()

    # Builds the agents' characteristics
    params = {
        'inf_params': {'age': 0.000},
        'test_params': {'age': 0.000},
    }
    inf_characs = ch.compute_characteristics(population_df, params['inf_params'])
    test_characs = ch.compute_characteristics(population_df, params['test_params'])

    # ACTUAL TEST
    n_agents = population_df.shape[0]
    population = Population(n_agents,
                            population_df,
                            inf_characs,
                            test_characs,
                            activity_data)

    assert population.get_state_count("susceptible") == n_agents

    # Tries to set the state of a group of agents to infected
    rng = np.random.default_rng(seed=42)
    some_agents = rng.integers(0, n_agents, 100)
    population.set_agents_state(some_agents, "infected")
    assert population.get_state_count("recovered") == 0
    assert population.get_state_count("infected") == 100
    assert population.get_state_count("susceptible") == n_agents - 100
    assert population.mobility.infected_visitors[0].sum() == 100

    # Tries to set the state of a part of that group to "recovered"
    population.set_agents_state(some_agents[:10], "recovered")
    assert population.get_state_count("recovered") == 10
    assert population.get_state_count("infected") == 90
    assert population.get_state_count("susceptible") == n_agents - 100
    assert population.mobility.infected_visitors[-1].sum() == 90

    # Tests that the counters of infected visitors function
    x = population.mobility.infected_visitors.copy()
    # supposedly retrieves the IDs of the currently infected agents
    infected_agents = population.get_subset_in_state(some_agents, "infected")
    # the following two lines should cancel each other
    population.mobility.remove_infected_visitors(infected_agents)
    population.mobility.add_infected_visitors(infected_agents)
    assert population.mobility.infected_visitors == x

    print("Test successful !")