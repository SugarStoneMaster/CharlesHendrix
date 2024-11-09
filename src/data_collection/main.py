import pygad

from src.data_collection.fitness_function import fitness_function
from src.data_collection.gene_space import total_genes, gene_space
from src.data_collection.genome_to_music import genome_to_music

ga_instance = pygad.GA(
    num_generations=2,
    num_parents_mating=5,
    fitness_func=fitness_function,
    sol_per_pop=20,
    num_genes=total_genes,
    gene_space=gene_space,
    gene_type=int,
    parent_selection_type="tournament",
    K_tournament=3,
    keep_parents=2,
    crossover_type="uniform",
    mutation_type="random",
    mutation_percent_genes=25
)


ga_instance.run()

# After running, retrieve the best solution (optional)
solution, solution_fitness, solution_idx = ga_instance.best_solution()
print(f"Best Solution Fitness: {solution_fitness}")

# Convert Genome to Music and Play (optional)
composition = genome_to_music(solution)
#composition.show()  # Play the music
