import csv
import os

from src.genetic_algorithm.gene_space import scale_mapping, melody_instrument_mapping, chord_instrument_mapping, minor_chord_progressions, \
    minor_chord_type_mapping, major_chord_progressions, major_chord_type_mapping


def genome_to_data():
    input_file = '../../data/genome_user_feedback.csv'
    output_file = '../../data/user_feedback.csv'

    # Ensure the data directory exists
    os.makedirs('../../data', exist_ok=True)

    # Read the input CSV file
    with open(input_file, 'r', newline='') as csvfile_in:
        reader = csv.reader(csvfile_in)
        headers = next(reader)  # Read the header row

        # Prepare to write the output CSV file
        with open(output_file, 'w', newline='') as csvfile_out:
            writer = csv.writer(csvfile_out)
            writer.writerow(headers)  # Write the same headers

            # Process each row
            for row in reader:
                mapped_row = row.copy()  # Make a copy of the row to modify

                # Map the Scale
                scale_idx = int(mapped_row[0])
                scale_name = scale_mapping.get(scale_idx, f"Unknown ({scale_idx})")
                mapped_row[0] = scale_name

                # Map the BPM (leave as is)
                # mapped_row[1] = mapped_row[1]

                # Map the Melody Instrument
                melody_instr_idx = int(mapped_row[2])
                mapped_row[2] = melody_instrument_mapping.get(melody_instr_idx, f"Unknown ({melody_instr_idx})")().instrumentName

                # Map the Chord Instrument
                chord_instr_idx = int(mapped_row[3])
                mapped_row[3] = chord_instrument_mapping.get(chord_instr_idx, f"Unknown ({chord_instr_idx})")().instrumentName

                # Map the Progression
                progression_idx = int(mapped_row[4])
                # Determine if the scale is major or minor
                if scale_name.islower():  # Minor scales are denoted by lowercase letters
                    progression = minor_chord_progressions.get(progression_idx, [])
                    chord_type_mapping = minor_chord_type_mapping
                else:
                    progression = major_chord_progressions.get(progression_idx, [])
                    chord_type_mapping = major_chord_type_mapping

                # Map the progression to chord names
                progression_chords = [chord_type_mapping.get(chord_degree, f"Unknown ({chord_degree})") for chord_degree in progression]
                mapped_row[4] = "-".join(progression_chords)

                # Note, Octave, and Duration columns remain unchanged
                # Optionally, you can process them if needed

                # Write the mapped row to the output CSV
                writer.writerow(mapped_row)


if __name__ == '__main__':
    genome_to_data()