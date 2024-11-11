import pygad

from src.genetic_algorithm.fitness_function import fitness_function_data_collection, set_fitness_function_inference

from src.genetic_algorithm.gene_space import total_genes, gene_space
from src.genetic_algorithm.genome_to_music import genome_to_music
from joblib import dump, load



def genetic_algorithm(inference=False):

    if not inference:
        ga_instance = pygad.GA(
            num_generations=2,
            num_parents_mating=5,
            fitness_func=fitness_function_data_collection,
            sol_per_pop=20,
            num_genes=total_genes,
            gene_space=gene_space,
            gene_type=int,
            parent_selection_type="tournament",
            K_tournament=3,
            keep_parents=2,
            crossover_type="uniform",
            mutation_type="random",
            mutation_percent_genes=25,
            keep_elitism=2
        )
    elif inference:
        fitness_function_inference = set_fitness_function_inference(load("../../model/model.joblib"),
                                                                    load("../../model/scaler.joblib"),
                                                                    load("../../model/columns.joblib"))
        ga_instance = pygad.GA(
            num_generations=50,
            num_parents_mating=10,
            fitness_func=fitness_function_inference,
            sol_per_pop=30,
            num_genes=total_genes,
            gene_space=gene_space,
            gene_type=int,
            parent_selection_type="tournament",
            K_tournament=3,
            keep_parents=2,
            crossover_type="uniform",
            mutation_type="random",
            mutation_percent_genes=25,
            keep_elitism=5
        )

    ga_instance.run()

    solution, solution_fitness, solution_idx = ga_instance.best_solution()
    print(f"Best Solution Fitness: {solution_fitness}")

    if inference:
        composition, _ = genome_to_music(solution)
        composition.show('midi')
        #TODO give feedback to the model


if __name__ == "__main__":
    genetic_algorithm(inference=True)
