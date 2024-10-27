from src.genome_to_music import genome_to_music, print_composition_details
import csv
import os

def fitness_function(ga_instance, solution, solution_idx):
    # Convert Genome to Music
    composition, last_used_idx = genome_to_music(solution)

    print_composition_details(composition, solution)

    composition.show('midi')  # Plays the music

    # Ask for user input
    user_input = input("Did you like the composition? (from 1 to 5): ")
    try:
        user_input = int(user_input)
        if user_input not in [1, 2, 3, 4, 5]:
            raise ValueError
    except ValueError:
        print("Invalid input. Assuming Dislike (1).")
        user_input = 1

    # Use user input as fitness score
    fitness = user_input

    # Save the composition and user feedback
    save_composition_and_feedback(solution, user_input, last_used_idx)

    return fitness






def save_composition_and_feedback(solution, user_input, last_used_idx):
    max_melody_notes = 32
    # Ensure the data directory exists
    os.makedirs('data', exist_ok=True)

    # Generate column names
    column_names = ["Scale", "BPM", "Melody Instrument", "Chord Instrument", "Progression"]

    # Melody note columns
    for i in range(1, max_melody_notes + 1):
        column_names.extend([f"Note {i}", f"Octave {i}", f"Duration {i}"])


    # Add the UserInput column
    column_names.append("UserInput")

    file_path = 'data/user_feedback.csv'

    # Check if the file already exists
    file_exists = os.path.isfile(file_path)

    # Create a copy of the solution to modify
    solution_copy = solution.copy()

    # Mark unused melody genes as -1
    # Melody genes start from index 5 and each note uses 3 genes
    unused_start_idx = last_used_idx
    for idx in range(unused_start_idx, 5 + max_melody_notes * 3):
        solution_copy[idx] = -1

    # Check for total expected solution length
    expected_solution_length = len(column_names) - 1  # Exclude UserInput

    # Ensure the solution length matches expected length
    if len(solution_copy) != expected_solution_length:
        raise ValueError(f"Length of solution ({len(solution_copy)}) does not match expected length ({expected_solution_length}).")

    with open(file_path, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)

        # If the file doesn't exist or is empty, write the header
        if not file_exists or os.stat(file_path).st_size == 0:
            writer.writerow(column_names)

        # Write the data row
        writer.writerow(solution_copy.tolist() + [user_input])