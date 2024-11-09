import music21.scale
from music21 import roman

from src.data_collection.gene_space import *

def genome_to_music(solution):
    # Extract global parameters
    scale_gene = solution[0]
    bpm_gene = solution[1]
    melody_instr_gene = solution[2]
    chord_instr_gene = solution[3]
    progression_gene = solution[4]

    scale_name = scale_mapping[scale_gene]
    bpm = bpm_gene
    melody_instrument = melody_instrument_mapping[melody_instr_gene]()
    chord_instrument = chord_instrument_mapping[chord_instr_gene]()

    # Initialize score
    score = music21.stream.Score()
    score.insert(0, music21.tempo.MetronomeMark(number=bpm))
    key_signature = music21.key.Key(scale_name)
    score.insert(0, key_signature)
    if key_signature.mode == 'major':
        chord_progression = major_chord_progressions[progression_gene]
    else:
        chord_progression = minor_chord_progressions[progression_gene]


    # Melody Part
    melody_part = music21.stream.Part()
    melody_part.partName = "Melody"
    melody_part.insert(0, melody_instrument)

    # Chord Part
    chord_part = music21.stream.Part()
    chord_part.partName = "Chords"
    chord_part.insert(0, chord_instrument)

    # Determine if the key is major or minor
    if key_signature.mode == 'major':
        chord_type_mapping = major_chord_type_mapping
        scale_obj = music21.scale.MajorScale(key_signature.tonic)
    else:
        chord_type_mapping = minor_chord_type_mapping
        scale_obj = music21.scale.MinorScale(key_signature.tonic)

    # Prepare scale pitches
    scale_pitches = [p.name for p in scale_obj.getPitches()]

    # Process melody notes
    idx = 5
    total_duration = 0.0
    last_used_idx = idx  # Keep track of the last used gene index
    while total_duration < 16.0 and idx + 2 < len(solution):
        scale_degree_gene = solution[idx]
        octave_gene = solution[idx + 1]
        duration_gene = solution[idx + 2]
        idx += 3

        duration_value = duration_values[duration_gene]

        if total_duration + duration_value > 16.0:
            duration_value = 16.0 - total_duration
            total_duration = 16.0
        else:
            total_duration += duration_value

        if scale_degree_gene == 0:
            n = music21.note.Rest()
        else:
            note_index = scale_degree_gene - 1
            note_name = scale_pitches[note_index] + str(octave_gene)
            n = music21.note.Note(note_name)

        n.duration = music21.duration.Duration(duration_value)
        melody_part.append(n)

        last_used_idx = idx  # Update the last used index

        if total_duration >= 16.0:
            break

    # Process chords
    segment_length = 16.0 / max_chords
    idx_chords = 5 + (max_melody_notes * 3)
    chord_count = 0
    while chord_count < max_chords:
        chord_scale_degree = chord_progression[chord_count % len(chord_progression)]
        chord_type = chord_type_mapping[chord_scale_degree]

        start_time = chord_count * segment_length

        if chord_type == 'Rest':
            c = music21.note.Rest()
            c.duration = music21.duration.Duration(segment_length)
        else:
            rn = roman.RomanNumeral(chord_type, key_signature)
            c = music21.chord.Chord(rn.pitches)
            c.duration = music21.duration.Duration(segment_length)
            if scale_gene <= 7:
                c = c.transpose(-12)
            else:
                c = c.transpose(-24)

            # Simplify enharmonics manually
            for n in c.notes:
                p = n.pitch
                if p.accidental is not None and abs(p.accidental.alter) > 1:
                    enharmonic_pitch = p.simplifyEnharmonic()
                    n.pitch = enharmonic_pitch

        chord_part.insert(start_time, c)
        chord_count += 1

    # Add parts to score
    score.insert(0, melody_part)
    score.insert(0, chord_part)

    return score, last_used_idx

def print_composition_details(composition: music21.stream.Score, solution):
    # Print the scale/key signature
    key_signature = composition.keySignature
    print(f"Composition Key: {key_signature}")
    print('=' * 40)

    # Iterate over all parts in the composition
    for part in composition.parts:
        # Get the part name or assign a default name
        part_name = part.partName if part.partName else "Unnamed Part"

        # Get the instrument used in this part
        instrument_obj = part.getInstrument()
        instrument_name = instrument_obj.instrumentName if instrument_obj else "Unknown Instrument"

        print(f"Part: {part_name}")
        print(f"Instrument: {instrument_name}")
        print('-' * 40)

        # Iterate over all elements (notes, chords, rests) in the part
        for element in part.recurse().notesAndRests:
            # Get the offset (time position) of the element in the part
            offset = element.getOffsetBySite(part)
            if isinstance(element, music21.note.Note):
                print(f"Offset: {offset:.2f} beats - Note: {element.nameWithOctave}, "
                      f"Duration: {element.duration.quarterLength} beats")
            elif isinstance(element, music21.chord.Chord):
                pitches = ' '.join(n.nameWithOctave for n in element.pitches)
                # Attempt to find the Roman numeral chord type
                rn = music21.roman.romanNumeralFromChord(element, key_signature)
                chord_type = rn.primaryFigure if rn else "Unknown"
                print(f"Offset: {offset:.2f} beats - Chord: {pitches}, "
                      f"Type: {chord_type}, "
                      f"Duration: {element.duration.quarterLength} beats")
            elif isinstance(element, music21.note.Rest):
                print(f"Offset: {offset:.2f} beats - Rest, "
                      f"Duration: {element.duration.quarterLength} beats")
        print('=' * 40)

    print(f"Composition Key: {key_signature}")
