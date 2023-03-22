
import music21
import pygad
import numpy

from src.Chords import createProgression
from src.Configuration import Composer, Key
from src.FitnessFunction import  setMelodyFitnessFunction
from src.GeneticOperators import setMutation

def main(selection, crossover):

    #inizializzazione
    key = Key(note="D", mode="Min")
    composer = Composer(chromatic=False, succession=[1, 4, 5, 4], min_duration=8, bars=2, key=key,
                        weight_notes_inscale=0,
                        weight_notes_inchord=0.65,
                        weight_intervals=0.20,
                        weight_numNotes=0.15,
                        weight_sustain=0.30,
                        weight_rest=0.20,
                        weight_pitchUp=0.25,
                        weight_pitchDown=0.25)

    #composer.succession = createProgression(composer) #crea tramite algoritmo genetico interattivo una progressione di accordi

    fitness_function = setMelodyFitnessFunction(composer)
    mutation_function = setMutation(composer)

    num_generations = 50
    num_parents_mating = 4

    sol_per_pop = 8
    num_genes = composer.num_music_elements

    init_range_low = composer.rest_value
    init_range_high = composer.sustain_value + 1
    random_mutation_min_val = composer.rest_value
    random_mutation_max_val = composer.sustain_value + 1

    parent_selection_type = selection
    keep_parents = 1

    crossover_type = crossover

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
    #print("{solution}".format(solution=solution))
    #print("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=solution_fitness))
    #ga_instance.plot_fitness()

    #passaggio dalla rappresentazione genetica a quella musicale con Music21
    #stream = composer.toMusic21(solution)
    #stream = composer.addChordsToMusic21(stream)

    #stream.show() #riproduzione dello spartito generato con programma esterno

    return solution_fitness


def testCombination(selection, crossover, n):
    total_fitness = 0
    for i in range(n):
        fitness = main(selection, crossover)
        total_fitness += fitness

    return total_fitness/n


n = 1000

avg_fitness_sss_single = testCombination("sss", "single_point", n)
avg_fitness_sss_double = testCombination("sss", "two_points", n)
avg_fitness_rws_single = testCombination("rws", "single_point", n)
avg_fitness_rws_double = testCombination("rws", "two_points", n)
avg_fitness_sus_single = testCombination("sus", "single_point", n)
avg_fitness_sus_double = testCombination("sus", "two_points", n)
avg_fitness_rank_single = testCombination("rank", "single_point", n)
avg_fitness_rank_double = testCombination("rank", "two_points", n)
avg_fitness_random_single = testCombination("random", "single_point", n)
avg_fitness_random_double = testCombination("random", "two_points", n)
avg_fitness_tournament_single = testCombination("tournament", "single_point", n)
avg_fitness_tournament_double = testCombination("tournament", "two_points", n)

print("Average fitness for steady state with single point "+str(avg_fitness_sss_single))
print("Average fitness for steady state with two points "+str(avg_fitness_sss_double))
print("Average fitness for roulette wheel with single point "+str(avg_fitness_rws_single))
print("Average fitness for roulette wheel  with two points "+str(avg_fitness_rws_double))
print("Average fitness for stocastic with single point "+str(avg_fitness_sus_single))
print("Average fitness for stocastic with two points "+str(avg_fitness_sus_double))
print("Average fitness for rank with single point "+str(avg_fitness_rank_single))
print("Average fitness for rank with two points "+str(avg_fitness_rank_double))
print("Average fitness for random with single point "+str(avg_fitness_random_single))
print("Average fitness for random with two points "+str(avg_fitness_random_double))
print("Average fitness for tournament with single point "+str(avg_fitness_tournament_single))
print("Average fitness for tournament with two points "+str(avg_fitness_tournament_double))

dict = {avg_fitness_sss_single: "avg_fitness_sss_single",
        avg_fitness_sss_double: "avg_fitness_sss_double",
        avg_fitness_rws_single: "avg_fitness_rws_single",
        avg_fitness_rws_double: "avg_fitness_rws_double",
        avg_fitness_sus_single: "avg_fitness_sus_single",
        avg_fitness_sus_double: "avg_fitness_sus_double",
        avg_fitness_rank_single: "avg_fitness_rank_single",
        avg_fitness_rank_double: "avg_fitness_rank_double",
        avg_fitness_random_single: "avg_fitness_random_single",
        avg_fitness_random_double: "avg_fitness_random_double",
        avg_fitness_tournament_single: "avg_fitness_tournament_single",
        avg_fitness_tournament_double: "avg_fitness_tournament_double"}


print("The max is {} with avg {}".format(dict.get(max(dict)), max(dict)))


