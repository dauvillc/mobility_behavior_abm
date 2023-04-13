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
        'recovery_mean_time': 8.0,
        'recovery_std_time': 2.0
    }
    inf_characs = ch.compute_characteristics(population_df, params['inf_params'])
    test_characs = ch.compute_characteristics(population_df, params['test_params'])

    # ACTUAL TEST
    n_agents = population_df.shape[0]
    population = Population(activity_data,
                            params)
    n_periods = population.mobility.n_periods

    assert population.get_state_count("susceptible") == n_agents

    # Tries to set the state of a group of agents to infected
    rng = np.random.default_rng(seed=42)
    some_agents = rng.integers(0, n_agents, 100)
    population.set_agents_state(some_agents, "infected")
    assert population.get_state_count("recovered") == 0
    assert population.get_state_count("infected") == 100
    assert population.get_state_count("susceptible") == n_agents - 100
    assert len(population.infected_agents_ids) == 100
    assert population.mobility.infected_visitors[0].sum() == 100

    # Tries to set the state of a part of that group to "recovered"
    population.set_agents_state(some_agents[:10], "recovered")
    assert population.get_state_count("recovered") == 10
    assert population.get_state_count("infected") == 90
    assert population.get_state_count("susceptible") == n_agents - 100
    assert len(population.infected_agents_ids) == 90
    assert population.mobility.infected_visitors[-1].sum() == 90
    print("Part 1 (State counters) successful !")

    # Tests that the counters of infected visitors function
    x = population.mobility.infected_visitors.copy()
    # supposedly retrieves the IDs of the currently infected agents
    infected_agents = population.get_subset_in_state(some_agents, "infected")
    # the following two lines should cancel each other
    population.mobility.remove_infected_visitors(infected_agents)
    population.mobility.add_infected_visitors(infected_agents)
    assert population.mobility.infected_visitors == x
    print("Part 2 (Visitors / Infected visitors counts) successful !")

    # Tests the reset methods
    population.reset()
    assert population.get_state_count("infected") == 0
    assert population.get_state_count("recovered") == 0
    assert len(population.infected_agents_ids) == 0
    # verifies that the mobility is identical to the original one.
    assert (population.mobility.get_locations(0) == activity_data[1][0]).all()
    print("Part 3 (Reset) successful !")

    # Checks the mobility modification method when no one is infected ===========================
    new_location = population.mobility.locations[0][some_agents].max() + 1
    # changes the locations of some agents to a common facility. We use the max of their former
    # location + 1, to make sure their locations actually change.
    former_count = population.mobility.visitors[0][new_location]
    population.change_agent_locations(some_agents, new_facilities=new_location)
    new_count = population.mobility.visitors[0][new_location]
    # verifies that the visitors count of the new location has increased by the right amount
    assert new_count == former_count + some_agents.shape
    # Changes their locations to some random ones
    new_facilities = [rng.integers(new_location, population.mobility.n_facilities, some_agents.shape)
                      for _ in range(n_periods)]
    new_facilities_unique = np.unique(new_facilities[0])
    former_counts = population.mobility.visitors[0][new_facilities_unique]
    population.change_agent_locations(some_agents, new_facilities=new_facilities)
    new_counts = population.mobility.visitors[0][new_facilities_unique]
    # Verifies that the visitors in the new facilities has increased by the right amount
    assert former_counts.sum() + some_agents.shape[0] == new_counts.sum()
    print("Part 4 (Mobility change without infections) successful !")

    # Now, reset the population, and do the same tests but this time with infected agents.
    population.reset()
    # Saves the state of the counters before we change the mobility, for later
    visitors_before_change = population.mobility.visitors[0].copy()
    inf_visitors_before_change = population.mobility.infected_visitors[0].copy()
    locations_before_change = population.mobility.locations[0].copy()
    # ==== With a single new location (every agent to the same new location) ======================
    # Sets some agents as infected
    infected_agents = some_agents[:15]
    population.set_agents_state(infected_agents, "infected")
    # Saves how many infected agents are at the future locations, before the mobility change
    former_inf_count = population.mobility.infected_visitors[0][new_location]
    # Changes the agents' locations the new ones
    population.change_agent_locations(some_agents, new_facilities=new_location)
    # Retrieves how many infected agents are at the new locations, after the change
    new_inf_count = population.mobility.infected_visitors[0][new_location]
    # verifies that the infected visitors count of the new location has increased by the right amount
    assert new_inf_count == former_inf_count + infected_agents.shape[0]

    # === With a different new location for every agent ==========================
    # Generates some random new facilities
    new_facilities = [rng.integers(new_location, population.mobility.n_facilities, some_agents.shape)
                      for _ in range(n_periods)]
    # Changes the mobility, with the generated facilities
    new_facilities_unique = np.unique(new_facilities[0])
    former_inf_counts = population.mobility.infected_visitors[0][new_facilities_unique]
    population.change_agent_locations(some_agents, new_facilities=new_facilities)
    new_inf_counts = population.mobility.infected_visitors[0][new_facilities_unique]
    # Verifies that the infected visitors in the new facilities has increased by the right amount
    assert former_inf_counts.sum() + infected_agents.shape[0] == new_inf_counts.sum()
    # Reset the mobility changes to verify that the counters are back to normal
    population.reset_agent_locations(some_agents)
    assert (locations_before_change == population.mobility.locations[0]).all()
    assert (visitors_before_change == population.mobility.visitors[0]).all()
    print("Part 5 (Mobility change with infections) successful !")

    print("Test successful !")
