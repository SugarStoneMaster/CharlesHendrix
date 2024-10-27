import music21
from music21 import roman
import pygad

from src.gene_space import total_genes, gene_space
from src.genome_to_music import genome_to_music, print_composition_details


def save_composition_and_feedback(solution, user_input):
    # Save the solution (genome) and user input to a file or database
    # For simplicity, we'll append to a CSV file
    import csv
    with open('user_feedback.csv', 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(solution.tolist() + [user_input])

def fitness_function(ga_instance, solution, solution_idx):
    # Convert Genome to Music
    composition = genome_to_music(solution)

    print_composition_details(composition, solution)

    composition.show('midi')  # Plays the music

    # Ask for user input
    user_input = input("Did you like the composition? (1 for Like, 0 for Dislike): ")
    try:
        user_input = int(user_input)
        if user_input not in [0, 1]:
            raise ValueError
    except ValueError:
        print("Invalid input. Assuming Dislike (0).")
        user_input = 0

    # Use user input as fitness score
    fitness = user_input

    # Save the composition and user feedback
    save_composition_and_feedback(solution, user_input)

    return fitness





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
