import os
import subprocess
import pygad
from src.genetic_algorithm.fitness_function import fitness_function_data_collection, set_fitness_function_inference, \
    save_composition_and_feedback
from src.genetic_algorithm.gene_space import total_genes, gene_space
from src.genetic_algorithm.genome_to_music import genome_to_music
from joblib import dump, load

genome_to_data_path = "/Users/carmine/PycharmProjects/CharlesHendrix/src/data_processing/data_extraction.py"
test_models_path = "/Users/carmine/PycharmProjects/CharlesHendrix/src/test_models/main.py"

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
        fitness_function_inference = set_fitness_function_inference(load("../model/model.joblib"),
                                                                    load("../model/scaler.joblib"),
                                                                    load("../model/columns.joblib"))
        ga_instance = pygad.GA(
            num_generations=10,
            num_parents_mating=20,
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
            mutation_percent_genes=40,
            keep_elitism=5,
            parallel_processing=20
        )
        ga_instance.on_generation = on_generation

    ga_instance.run()

    solution, solution_fitness, solution_idx = ga_instance.best_solution()
    print(f"Best Solution Fitness: {solution_fitness}")

    if inference:
        composition, last_used_idx = genome_to_music(solution)
        composition.show('midi')

        user_input = input("Did you like the composition? (from 1 to 5): ")
        try:
            user_input = int(user_input)
            if user_input not in [1, 2, 3, 4, 5]:
                raise ValueError
        except ValueError:
            print("Invalid input. Assuming Dislike (1).")
            user_input = 1

        save_composition_and_feedback(solution, user_input, last_used_idx)

        counter = read_and_increment_counter()

    # dataset has enough new entries, retrain
    if (not inference) or counter >= 50:
        # transform genome to data
        command = ['python3', genome_to_data_path]
        result = subprocess.run(command, capture_output=True, text=True)
        print(result.stdout)

        # pre-process data and train and save the best model
        command = ['python3', test_models_path]
        result = subprocess.run(command, capture_output=True, text=True)
        print(result.stdout)

        # reset the counter
        read_and_increment_counter(reset_counter=True)


def on_generation(ga_instance: pygad.GA):
    # Get the best solution’s fitness (tuple of MAE, QWK)
    best_solution, best_fitness, _ = ga_instance.best_solution()
    print(f"Generation {ga_instance.generations_completed}: best fitness =  {best_fitness:.4f}")


def read_and_increment_counter(filepath="../data/new_compositions_counter.txt", reset_counter=False):
    if os.path.exists(filepath):
        # Read the counter from the file
        with open(filepath, 'r') as file:
            content = file.read().strip()
            try:
                counter = int(content)
            except ValueError:
                # If the file is empty or contains invalid data, reset counter to 0
                counter = 0
    else:
        # If the file doesn't exist, start counter at 0
        counter = 0

    if reset_counter:
        counter = 0
    else:
        counter += 1

    # Write the updated counter back to the file
    with open(filepath, 'w') as file:
        file.write(f"{counter}")

    return counter


if __name__ == "__main__":
    genetic_algorithm(inference=True)
