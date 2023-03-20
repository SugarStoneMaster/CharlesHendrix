import music21
import pygad
import numpy

from src.Chords import createProgression
from src.Configuration import Composer, Key
from src.FitnessFunction import  setNotesFitnessFunction
from src.GeneticOperators import setMutation

#inizializzazione
key = Key(note="E", mode="Maj")
configuration = Composer(min_duration=8, bars=8, key=key,
                         weight_notes_inscale=0,
                         weight_notes_inchord=0.7,
                         weight_intervals=0.20,
                         weight_numNotes=0.10,
                         chromatic=False,
                         succession=[1, 5, 6, 4],
                         weight_sustain=0.3,
                         weight_rest=0.1,
                         weight_pitchUp=0.3,
                         weight_pitchDown=0.3)

#configuration.succession = createProgression(configuration) #crea tramite algoritmo genetico interattivo una progressione di accordi

fitness_function = setNotesFitnessFunction(configuration)
mutation_function = setMutation(configuration)

num_generations = 200
num_parents_mating = 4

sol_per_pop = 8
num_genes = configuration.num_music_elements

init_range_low = configuration.rest_value
init_range_high = configuration.sustain_value + 1
random_mutation_min_val = configuration.rest_value
random_mutation_max_val = configuration.sustain_value + 1

parent_selection_type = "rws"
keep_parents = 1

crossover_type = "two_points"

mutation_type = mutation_function


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
                       crossover_type=crossover_type,
                       mutation_type=mutation_type,
                       random_mutation_min_val=random_mutation_min_val,
                       random_mutation_max_val=random_mutation_max_val,
                       gene_type=int)

ga_instance.run()

solution, solution_fitness, solution_idx = ga_instance.best_solution()
print("{solution}".format(solution=solution))
print("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=solution_fitness))
#ga_instance.plot_fitness()

#passaggio dalla rappresentazione genetica a quella musicale con Music21
stream = configuration.toMusic21(solution)
stream = configuration.addChordsToMusic21(stream)



stream.show() #riproduzione dello spartito generato con programma esterno






