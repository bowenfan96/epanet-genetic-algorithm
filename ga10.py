import epamodule as ep
import numpy as np

# set min and max commercially available pipe diameters in mm
min_commercial_diameter = 20
max_commercial_diameter = 710

# set network requirements
min_head = -1e30
min_vel = 0
max_vel = 1e30

# set genetic algorithm population size
population_size = 10

ridiculous_value = 1e30

file = "C:\\Users\\FanBo\\PycharmProjects\\fyp_ga\\BAK.inp"
ep.ENopen(file, nomerpt='EPANET_report.rpt')

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

# equation that gives pipe cost as a function of diameter
# fitting with Excel using data from http://www.gpsuk.com/uploads/docs/1635.pdf
# cost in gbp per m, diameter in mm
def pipe_cost(diameter):
    pipe_cost_gbp = 0.0031*(diameter**2) - 0.0284*diameter + 1.7442
    return pipe_cost_gbp

def network_cost(length_array, diameter_array):
    assert len(length_array) == len(diameter_array)
    cost = 0
    for i in range(len(length_array)):
        #print(i)
        cost += length_array[i]*(pipe_cost(diameter_array[i]))
    return cost

def constraint_checker(diameter_array, min_head, min_vel, max_vel):
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
        #print('node: ', node)

        if ep.ENgetnodevalue(node + 1, 10) < min_head:
            fail_pressure_list.append(node)
            fail_pressure = True

    for pipe in range(num_pipes):
        #print("pipe: ", pipe)

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
    print('Negative diameter list (links):', negative_diameter_list)

    if fail_pressure or fail_min_vel or fail_max_vel or negative_diameter:
        return False
    else:
        return True

def random_diameter_generator(num_pipes):
    random_diameter_array = np.random.randint(min_commercial_diameter, max_commercial_diameter, num_pipes)
    return random_diameter_array

def fitness_function(cost, constraint_pass):
    if constraint_pass:
        return cost
    else:
        return ridiculous_value

def cal_pop_fitness(length_array, population_array):
    # Calculating the fitness value of each solution in the current population.
    # The fitness function calulates the sum of products between each input and its corresponding weight.
    fitness = np.zeros(population_size)
    for population in range(population_size):
        ga_diameter_array = population_array[population, :]
        print("Random array", ga_diameter_array)
        fitness[population] = fitness_function(network_cost(length_array, ga_diameter_array),
                                               constraint_checker(ga_diameter_array, min_head, min_vel, max_vel))
    return fitness

def select_mating_pool(pop, fitness, num_parents):
    # Selecting the best individuals in the current generation as parents for producing the offspring of the next generation.
    parents = np.empty((num_parents, pop.shape[1]))
    for parent_num in range(num_parents):
        max_fitness_idx = np.where(fitness == np.min(fitness))
        max_fitness_idx = max_fitness_idx[0][0]
        parents[parent_num, :] = pop[max_fitness_idx, :]
        fitness[max_fitness_idx] = ridiculous_value
    return parents

def crossover(parents, offspring_size):
    offspring = np.empty(offspring_size)
    # The point at which crossover takes place between two parents. Usually, it is at the center.
    crossover_point = np.random.randint(1, offspring_size[1])

    for k in range(offspring_size[0]):
        if k < parents.shape[0]:
            # Index of the first parent to mate.
            parent1_idx = k%parents.shape[0]
            # Index of the second parent to mate.
            parent2_idx = (k+1)%parents.shape[0]
            # The new offspring will have its first half of its genes taken from the first parent.
            offspring[k, 0:crossover_point] = parents[parent1_idx, 0:crossover_point]
            # The new offspring will have its second half of its genes taken from the second parent.
            offspring[k, crossover_point:] = parents[parent2_idx, crossover_point:]
        else:
            offspring[k, :] = (random_diameter_generator(num_pipes)/np.random.randint(1, 100)) + 10
    return offspring

def mutation(offspring_crossover, num_mutations=1):
    mutations_counter = np.uint8(offspring_crossover.shape[1] / num_mutations)
    # Mutation changes a number of genes as defined by the num_mutations argument. The changes are random.
    for idx in range(offspring_crossover.shape[0]):
        gene_idx = mutations_counter - 1
        for mutation_num in range(num_mutations):
            # The random value to be added to the gene.
            random_value = np.random.randint(min_commercial_diameter, max_commercial_diameter)
#            offspring_crossover[idx, gene_idx] = random_value
            offspring_crossover[idx, np.random.randint(0, offspring_crossover.shape[1])] = random_value
            gene_idx = gene_idx + mutations_counter
    return offspring_crossover

pop_size = (population_size, num_pipes)
population_array = np.random.randint(min_commercial_diameter, max_commercial_diameter, size=pop_size)

best_outputs = []
num_generations = 100
num_parents_mating = 4

for generation in range(num_generations):
    ep.ENopen(file)

    print("Generation : ", generation)
    # Measuring the fitness of each chromosome in the population.
    fitness = cal_pop_fitness(length_array, population_array)
    print("Fitness")
    print(fitness)

    best_outputs.append(np.min(fitness))
    # The best result in the current iteration.
    print("Best result : ", np.min(fitness))

    # Selecting the best parents in the population for mating.
    parents = select_mating_pool(population_array, fitness, num_parents_mating)
    print("Parents")
    print(parents)

    # Generating next generation using crossover.
    offspring_crossover = crossover(parents,
                                    offspring_size=(pop_size[0] - parents.shape[0], num_pipes))
    print("Crossover")
    print(offspring_crossover)

    # Adding some variations to the offspring using mutation.
    offspring_mutation = mutation(offspring_crossover, num_mutations=2)
    print("Mutation")
    print(offspring_mutation)

    # Creating the new population based on the parents and offspring.
    population_array[0:parents.shape[0], :] = parents
    population_array[parents.shape[0]:, :] = offspring_mutation

    print("POPULATION ARRAY")
    print(population_array)

    ep.ENclose()

ep.ENopen(file)

# Getting the best solution after iterating finishing all generations.
# At first, the fitness is calculated for each solution in the final generation.
fitness = cal_pop_fitness(length_array, population_array)
# Then return the index of that solution corresponding to the best fitness.
best_match_idx = np.where(fitness == np.min(fitness))

print("Best solution : ", population_array[best_match_idx, :])
print("Best solution fitness : ", fitness[best_match_idx])

ep.ENclose()

import matplotlib.pyplot

matplotlib.pyplot.plot(best_outputs)
matplotlib.pyplot.xlabel("Iteration")
matplotlib.pyplot.ylabel("Cost")
matplotlib.pyplot.show()