import numpy as np
from music21 import key, scale

duration_values = {
    0: 0.25,  # Sixteenth note
    1: 0.25,  # Sixteenth note
    2: 0.5,  # Eighth note
    3: 0.5,  # Eighth note
    4: 0.5,  # Eighth note
    5: 1.0,  # Quarter note
    6: 1.0,  # Quarter note
    7: 1.5,  # Dotted quarter note
    8: 2.0,  # Half note
    9: 3.0,  # Dotted half note
}

chord_functions = ['I', 'ii', 'iii', 'IV', 'V', 'vi', 'viio', 'i', 'iio', 'III', 'iv', 'v', 'VI', 'VII']


def map_duration_code(duration_code):
    return duration_values.get(int(duration_code), np.nan)


def note_to_midi(scale_name, scale_degree, octave):
    if scale_degree == 0:
        return None  # Rest
    # Determine if the scale is major or minor
    if scale_name.islower():  # Minor scale
        key_obj = key.Key(scale_name)
        sc = scale.MinorScale(key_obj.tonic)
    else:
        key_obj = key.Key(scale_name)
        sc = scale.MajorScale(key_obj.tonic)
    # Get the pitch for the scale degree
    degree = scale_degree
    # Ensure degree is within 1-7
    if degree < 1 or degree > 7:
        return None
    pitch_obj = sc.pitchFromDegree(degree)
    pitch_obj.octave = octave
    return pitch_obj.midi


def compute_melodic_intervals(row):
    intervals = []
    previous_pitch = None
    scale_name = row['Scale']
    for i in range(1, 33):
        note_col = f'Note {i}'
        octave_col = f'Octave {i}'
        note_value = row[note_col]
        octave_value = row[octave_col]
        if note_value == -1 or note_value == 0:
            continue  # Skip unused notes and rests
        current_pitch = note_to_midi(scale_name, int(note_value), int(octave_value))
        if previous_pitch is not None and current_pitch is not None:
            interval = current_pitch - previous_pitch
            intervals.append(interval)
        previous_pitch = current_pitch
    if intervals:
        return np.mean(intervals), np.std(intervals)
    else:
        return 0.0, 0.0


def encode_chord_functions(progression_str):
    chords = progression_str.split('-')
    chord_presence = {func: 0 for func in chord_functions}
    for chord in chords:
        if chord in chord_presence:
            chord_presence[chord] += 1
    return list(chord_presence.values())


def compute_progression_complexity(progression_str):
    chords = progression_str.split('-')
    unique_chords = set(chords)
    return len(unique_chords)