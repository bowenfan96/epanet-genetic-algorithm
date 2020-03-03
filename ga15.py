import epamodule as ep
import numpy as np
import PySimpleGUI as sg
import matplotlib.pyplot as plt
import csv
import random

# global variables
commercial_diameters = []
commercial_costs = []
ridiculous_value = float('inf')


def pipe_cost(diameter):
    cost_index = commercial_diameters.index(diameter)
    pipe_cost = commercial_costs[cost_index]
    return pipe_cost


def network_cost(length_array, diameter_array):
    assert len(length_array) == len(diameter_array)
    network_cost = 0
    for i in range(len(length_array)):
        network_cost += length_array[i] * (pipe_cost(diameter_array[i]))
    return network_cost


def constraint_checker(diameter_array, min_pre, min_vel, max_vel,
                       file, num_nodes, num_links, num_pipes):
    # reconstruct .inp file from diameter array and store it as a temp file
    for pipe in range(len(diameter_array)):
        if diameter_array[pipe] > 0:
            ep.ENsetlinkvalue(pipe + 1, 0, diameter_array[pipe])

    ep.ENsaveinpfile(file)

    ep.ENopenH()
    ep.ENinitH()
    ep.ENrunH()

    # initialize an array of nodal heads
    head_array = np.zeros(num_nodes)
    # initialize an array of pipe velocities
    velocity_array = np.zeros(num_links)

    fail_pressure_list = []
    fail_min_vel_list = []
    fail_max_vel_list = []
    negative_diameter_list = []

    fail_pressure = False
    fail_min_vel = False
    fail_max_vel = False
    negative_diameter = False

    for node in range(num_nodes):

        if ep.ENgetnodevalue(node + 1, 11) < min_pre and ep.ENgetnodetype(node + 1) == 0:
            fail_pressure_list.append(node)
            fail_pressure = True

    for pipe in range(num_pipes):

        if ep.ENgetlinkvalue(pipe + 1, 9) < min_vel:
            fail_min_vel_list.append(pipe)
            fail_min_vel = True
        elif ep.ENgetlinkvalue(pipe + 1, 9) > max_vel:
            fail_max_vel_list.append(pipe)
            fail_max_vel = True

    i = 0
    for diameter in diameter_array:
        i += 1
        if diameter <= 0:
            negative_diameter_list.append(i)
            negative_diameter = True

    print('Fail pressure list (nodes):', fail_pressure_list)
    print('Lower than min velocity list (links):', fail_min_vel_list)
    print('Higher than max velocity list (links):', fail_max_vel_list)

    if fail_pressure or fail_min_vel or fail_max_vel or negative_diameter:
        return False
    else:
        return True


def random_diameter_generator(num_pipes):
    random_diameter_array = []
    for pipe in range(num_pipes):
        random_diameter_array.append(random.choice(commercial_diameters))
    assert len(random_diameter_array) == num_pipes
    return random_diameter_array


def fitness_function(cost, constraint_pass):
    if constraint_pass:
        return cost
    else:
        return ridiculous_value


def population_cost(length_array, population_array, population_size, min_pre, min_vel, max_vel,
                    file, num_nodes, num_links, num_pipes):
    # this function calculates the cost of each diameter configuration in the population
    # and append it to an array listing the fitness values
    # if the network configuration fails the pressure or velocity constraints, then it receives an infinite penalty
    # i.e. the cost of infeasible networks is set at infinity, so they will not be chosen for the next generation

    config_costs = np.zeros(population_size)

    for diameter_config in range(population_size):
        current_config = population_array[diameter_config, :]
        config_costs[diameter_config] = fitness_function(network_cost(length_array, current_config),
                                                        constraint_checker(current_config, min_pre, min_vel,
                                                                           max_vel, file, num_nodes,
                                                                           num_links, num_pipes))
    return config_costs


def get_best_configs(population, config_fitness, num_parents):
    best_configs = np.zeros((num_parents, population.shape[1]))

    for config in range(num_parents):
        best_config_index = np.where(config_fitness == np.min(config_fitness))[0][0]
        best_configs[config, :] = population[best_config_index, :]
        config_fitness[best_config_index] = ridiculous_value

    return best_configs


def merge_best_configs(best_configs, num_children, num_pipes):
    evolved_generation = np.zeros((num_children, num_pipes))

    merge_index = np.random.randint(0, num_pipes)

    # merge the best configurations into evolved configurations in a rotary relay
    # i.e. 1 merge with 2, 2 with 3, 3 with 4, 4 with 1

    for i in range(num_children):
        if i < best_configs.shape[0]:

            config_1 = i % best_configs.shape[0]
            config_2 = (i + 1) % best_configs.shape[0]

            evolved_generation[i, 0:merge_index] = best_configs[config_1, 0:merge_index]
            evolved_generation[i, merge_index:] = best_configs[config_2, merge_index:]

        else:
            evolved_generation[i, :] = random_diameter_generator(num_pipes)

    return evolved_generation


def mutate_config(evolved_generation, mutations=1):
    for pipe_config in range(evolved_generation.shape[0]):
        for mutation in range(mutations):
            evolved_generation[pipe_config, np.random.randint(0, evolved_generation.shape[1])] \
                = random.choice(commercial_diameters)

    return evolved_generation


def main():
    inp_file = values['inp_filepath']
    csv_file = values['csv_filepath']
    min_pre = float(values['min_pressure'])
    min_vel = float(values['min_velocity'])
    max_vel = float(values['max_velocity'])
    population_size = int(values['pop_size'])
    num_generations = int(values['num_generations'])

    with open(csv_file, 'r') as cost_file:
        read_csv = csv.reader(cost_file, delimiter=',')
        next(read_csv)
        for rows in read_csv:
            commercial_diameters.append(float(rows[0]))
            commercial_costs.append(float(rows[1]))

    assert len(commercial_diameters) == len(commercial_costs)

    # call the EPANET engine to open the supplied network input file
    ep.ENopen(inp_file)

    # get from EPANET the number of nodes in the network (nodes: junctions, reservoirs, tanks)
    num_nodes = ep.ENgetcount(0)

    # get from EPANET the number of tanks and reservoirs in the network
    num_tanks = ep.ENgetcount(1)

    # get from EPANET the number of links in the network (links: pipes, valves, pumps)
    num_links = ep.ENgetcount(2)

    # get from EPANET the number of pipes in the network
    num_pipes = 0
    for link in range(num_links):
        if ep.ENgetlinktype(link + 1) == 0 or 1:
            num_pipes += 1

    # initialize an array of pipe diameters
    diameter_array = np.zeros(num_pipes)
    # initialize an array of pipe lengths
    length_array = np.zeros(num_pipes)

    # this FOR loop makes 2 tables, listing the lengths and diameters of the pipes
    for pipe in range(num_pipes):
        diameter_array[pipe] = ep.ENgetlinkvalue(pipe + 1, 0)
        length_array[pipe] = ep.ENgetlinkvalue(pipe + 1, 1)

    # check that the diameter and length arrays contain the same number of pipes
    assert len(diameter_array) == len(length_array)

    # initialize a randomly generated population array
    # the population size is the number of pipe configurations
    population_array = []
    for network in range(population_size):
        population_array.append(random_diameter_generator(num_pipes))
    population_array = np.array(population_array)

    least_cost_configs = []
    num_configs_to_merge = 4

    # genetic algorithm iteration loop
    for ga_generation in range(num_generations):
        # call the EPANET engine to open the file in each loop
        ep.ENopen(inp_file)

        print('Current Generation: ', ga_generation)

        # get the cost of each pipe network configuration in the generation using the cost functions above
        # the cost is listed in an array corresponding to the cost of each network configuration
        config_costs = population_cost(length_array, population_array, population_size, min_pre, min_vel, max_vel,
                                       inp_file, num_nodes, num_links, num_pipes)
        print('Network Costs: ')
        print(config_costs)

        # record the cost of the lowest cost feasible configuration found so far
        least_cost_configs.append(np.min(config_costs))
        print('Lowest Cost: ', np.min(config_costs))

        # get the lowest cost configurations to merge (mating function)
        # these are the least cost parents that we will merge to get evolved children
        # (which are hopefully better than both the parents)
        best_configs = get_best_configs(population_array, config_costs, num_configs_to_merge)
        print('Parent Configurations: ')
        print(best_configs)

        # evolve by merging the best configurations (crossover function)
        num_children = population_size - num_configs_to_merge
        evolved_children = merge_best_configs(best_configs, num_children, num_pipes)
        print('Evolved Children Configurations: ')
        print(evolved_children)

        # mutate some pipes in the children for genetic diversity (mutation function)
        mutated_evolved_children = mutate_config(evolved_children, mutations=2)
        print('Evolved and Mutated Configurations: ')
        print(mutated_evolved_children)

        # append the evolved and mutated children to the parents
        # e.g. if the population size is 10, and the top 4 least cost configs are chosen as parents, then:
        # the 4 least cost parents will merge to output 4 evolved children configs:
        # 1 will merge with 2, 2 with 3, 3 with 4, 4 with 1

        # another 2 network configs will be randomly generated from the commercially available diameters
        # to add genetic diversity

        # thus, we now have 4 least cost parents, 4 evolved children (which are hopefully
        # even lower cost than their parents), and 2 randomly generated configs
        # these are first mutated, then merged to get back the population size of 10 (4 + 4 + 2)

        population_array[0: num_configs_to_merge, :] = best_configs
        population_array[num_configs_to_merge:, :] = mutated_evolved_children

        print('New Population: ')
        print(population_array)

        # close the EPANET engine to free up memory for the next generation
        ep.ENclose()

        # end of the generation algorithm iteration loop

    # this segment writes the least cost config into the EPANET input file and prints the config and its cost
    ep.ENopen(inp_file)

    # get the lowest cost feasible network config found by the genetic algorithm
    final_costs = population_cost(length_array, population_array, population_size, min_pre, min_vel, max_vel,
                                  inp_file, num_nodes, num_links, num_pipes)

    least_cost_index = np.where(final_costs == np.min(final_costs))[0][0]
    least_cost_network = population_array[least_cost_index, :].flatten()

    # write the least cost feasible network config back into the supplied EPANET input file
    for pipe in range(len(least_cost_network)):
        ep.ENsetlinkvalue(pipe + 1, 0, least_cost_network[pipe])

    ep.ENsaveinpfile(inp_file)
    ep.ENclose()

    # print the least cost network and its cost to the console
    print('Least Cost Network Found by Genetic Algorithm: ')
    print(least_cost_network)
    print('Cost of Least Cost Network: ')
    print(final_costs[least_cost_index])

    if final_costs[least_cost_index] == float('inf'):
        print('No solutions found. Problem may be overconstrained or insufficient number of generations.')
        sg.Popup('No solutions found. Problem may be overconstrained or insufficient number of generations.')

    # use matplotlib to plot the best cost against number of generations
    plt.plot(least_cost_configs)
    plt.xlabel("Generation")
    plt.ylabel("Cost of the Least Cost Network")
    plt.show()


# GUI window using PySimpleGUI

layout = [[sg.Text('EPANET network .inp file:', size=(21, 1)),
           sg.Input(key='inp_filepath',
                    default_text='C:\\Users\\FanBo\\PycharmProjects\\fyp_ga\\TLN.inp'),
           sg.FileBrowse(file_types=(("INP", ".inp"),))],

          [sg.Text('Pipe diameter-cost table:', size=(21, 1)),
           sg.Input(key='csv_filepath',
                    default_text='C:\\Users\\FanBo\\PycharmProjects\\fyp_ga\\csv.csv'),
           sg.FileBrowse(file_types=(("CSV", ".csv"),))],

          [sg.Text('Minimum nodal pressure (m):', size=(21, 1)),
           sg.Input(key='min_pressure', size=(10, 1),
                    default_text=0)],

          [sg.Text('Minimum pipe velocity (m/s):', size=(21, 1)),
           sg.Input(key='min_velocity', size=(10, 1),
                    default_text=0)],

          [sg.Text('Maximum pipe velocity (m/s):', size=(21, 1)),
           sg.Input(key='max_velocity', size=(10, 1),
                    default_text='inf'),

           sg.Text('', size=(7, 1)),
           sg.Button('Run Genetic Algorithm', size=(20, 1))],

          [sg.Text('Population size:', size=(21, 1)),
           sg.Input(key='pop_size', size=(10, 1),
                    default_text=10)],

          [sg.Text('Number of generations:', size=(21, 1)),
           sg.Input(key='num_generations', size=(10, 1),
                    default_text=100)],
          ]

window = sg.Window('Pipeline Cost Optimisation - UCL Third Year Project - Bowen Fan', layout)

while True:
    action, values = window.Read()

    if action == 'Run Genetic Algorithm':
        main()

    elif action is None:
        break

    break

window.Close()
