import music21
import music21.converter
import pygad

import src.Configuration
from src.Configuration import Composer
from src.FitnessFunction import setChordsFitnessFunction

def createProgression(configuration: src.Configuration.Composer):
    """Crea una progressione di accordi utilizzando un algoritmo genetico interattivo -> valore di fitness stabilito
    dall'utente a run-time"""

    #inizializzazione
    fitness_function = setChordsFitnessFunction(configuration)

    num_generations = 3 #non si possono creare un numero eccessivo di generazioni per via del collo di bottiglia dato dall'interazione con l'utente
    num_parents_mating = 2

    sol_per_pop = 2
    num_genes = 4 #il numero di accordi in una successione

    init_range_low = 1
    init_range_high = 7 + 1
    random_mutation_min_val = 1
    random_mutation_max_val = 7 + 1

    parent_selection_type = "sss"
    keep_parents = 1

    mutation_type = "random"
    mutation_by_replacement = True
    mutation_percent_genes = 10

    crossover_type = "single_point"

    #configurazione algoritmo genetico
    ga_instance = pygad.GA(num_generations=num_generations,
                       num_parents_mating=num_parents_mating,
                       fitness_func=fitness_function,
                       sol_per_pop=sol_per_pop,
                       num_genes=num_genes,
                       init_range_low=init_range_low,
                       init_range_high=init_range_high,
                       parent_selection_type=parent_selection_type,
                       keep_parents=keep_parents,
                       mutation_type=mutation_type,
                       mutation_by_replacement=mutation_by_replacement,
                       crossover_type=crossover_type,
                       mutation_num_genes=1,
                       random_mutation_min_val=random_mutation_min_val,
                       random_mutation_max_val=random_mutation_max_val,
                       gene_type=int)

    ga_instance.run()

    solution, solution_fitness, solution_idx = ga_instance.best_solution()
    print("Parameters of the best solution : {solution}".format(solution=solution))
    print("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=solution_fitness))

    return solution




