import music21
from music21 import roman
import pygad

from src.fitness_function import fitness_function
from src.gene_space import total_genes, gene_space
from src.genome_to_music import genome_to_music, print_composition_details


ga_instance = pygad.GA(
    num_generations=10,
    num_parents_mating=2,
    fitness_func=fitness_function,
    sol_per_pop=2,
    num_genes=total_genes,
    gene_space=gene_space,
    gene_type=int,
    parent_selection_type="sss",
    keep_parents=2,
    crossover_type="single_point",
    mutation_type="random",
    mutation_percent_genes=5
)

ga_instance.run()

# After running, retrieve the best solution (optional)
solution, solution_fitness, solution_idx = ga_instance.best_solution()
print(f"Best Solution Fitness: {solution_fitness}")

# Convert Genome to Music and Play (optional)
composition = genome_to_music(solution)
#composition.show()  # Play the music
