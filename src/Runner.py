import music21
import pygad
import numpy

from src.Chords import createProgression
from src.Configuration import ComposerConfig, EMinKey
from src.FitnessFunction import  setNotesFitnessFunction
from src.GeneticOperators import setMutation

configuration = ComposerConfig(min_duration=8, bars=8, key=EMinKey,
                               weight_notes_inchord=0.7,
                               weight_notes_inscale=0,
                               weight_intervals=0.20,
                               weight_numNotes=0.1,
                               chromatic=False,
                               succession=[2, 5, 1, 1])

#configuration.succession = createProgression(configuration)
"""
parsed1 = music21.converter.parse("../xmls/Eiffel_65_-_Blue_Da_Ba_Dee.xml")
parsed2 = music21.converter.parse("../xmls/Pinguini-Tattici-Nucleari-Ringo-Starr-Sanremo-2020.xml")

genetic1 = configuration.toGenetic(parsed1)
genetic2 = configuration.toGenetic(parsed2)
initial_population = [genetic1, genetic2]
"""



fitness_function = setNotesFitnessFunction(configuration)
mutation_function = setMutation(configuration)


num_generations = 200
num_parents_mating = 4

sol_per_pop = 8
num_genes = configuration.num_notes

init_range_low = configuration.break_value
init_range_high = configuration.repeat_value + 1
random_mutation_min_val = configuration.break_value
random_mutation_max_val = configuration.repeat_value + 1

parent_selection_type = "rws"
keep_parents = 1

crossover_type = "two_points"

mutation_type = mutation_function
mutation_percent_genes = 10

ga_instance = pygad.GA(num_generations=num_generations,
                       num_parents_mating=num_parents_mating,
                       fitness_func=fitness_function,
                       sol_per_pop=sol_per_pop,
                       num_genes=num_genes,
                       init_range_low=init_range_low,
                       init_range_high=init_range_high,
                       parent_selection_type=parent_selection_type,
                       keep_parents=keep_parents,
                       crossover_type=crossover_type,
                       mutation_type=mutation_type,
                       mutation_percent_genes=mutation_percent_genes,
                       random_mutation_min_val=random_mutation_min_val,
                       random_mutation_max_val=random_mutation_max_val,
                       gene_type=int)

ga_instance.run()

solution, solution_fitness, solution_idx = ga_instance.best_solution()
print("Parameters of the best solution : {solution}".format(solution=solution))
print("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=solution_fitness))
#ga_instance.plot_fitness()

stream = configuration.toMusic21(solution)
stream = configuration.addChordsToMusic21(stream)


stream.show()






"""
import pygad


function_inputs = [4,-2,3.5,5,-11,-4.7]
desired_output = 44

def fitness_func(*args):
    output = numpy.sum(args[0]*function_inputs)
    fitness = 1.0 / numpy.abs(output - desired_output)
    return fitness
    

fitness_function = fitness_func

num_generations = 50
num_parents_mating = 4

sol_per_pop = 8
num_genes = len(function_inputs)

init_range_low = -2
init_range_high = 5

parent_selection_type = "sss"
keep_parents = 1

crossover_type = "single_point"

mutation_type = "random"
mutation_percent_genes = 10


ga_instance = pygad.GA(num_generations=num_generations,
                       num_parents_mating=num_parents_mating,
                       fitness_func=fitness_function,
                       sol_per_pop=sol_per_pop,
                       num_genes=num_genes,
                       init_range_low=init_range_low,
                       init_range_high=init_range_high,
                       parent_selection_type=parent_selection_type,
                       keep_parents=keep_parents,
                       crossover_type=crossover_type,
                       mutation_type=mutation_type,
                       mutation_percent_genes=mutation_percent_genes)

ga_instance.run()

solution, solution_fitness, solution_idx = ga_instance.best_solution()
print("Parameters of the best solution : {solution}".format(solution=solution))
print("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=solution_fitness))

prediction = numpy.sum(numpy.array(function_inputs)*solution)
print("Predicted output based on the best solution : {prediction}".format(prediction=prediction))
"""
