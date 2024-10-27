from src.genome_to_music import genome_to_music, print_composition_details


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


def save_composition_and_feedback(solution, user_input):
    # Save the solution (genome) and user input to a file or database
    # For simplicity, we'll append to a CSV file
    import csv
    with open('user_feedback.csv', 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(solution.tolist() + [user_input])