import numpy as np
import pandas as pd
from music21 import scale, key
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def feature_engineering(df):
    # Initialize new feature columns
    df['num_notes_played'] = 0
    df['avg_note_duration'] = 0.0
    df['std_note_duration'] = 0.0
    df['avg_pitch'] = 0.0
    df['pitch_range'] = 0.0
    df['proportion_rests'] = 0.0

    # Process each row to compute features
    for index, row in df.iterrows():
        scale_name = row['Scale']
        notes = []
        durations = []
        rests = 0
        total_notes = 0
        for i in range(1, 33):  # Note 1 to Note 32
            note_col = f'Note {i}'
            octave_col = f'Octave {i}'
            duration_col = f'Duration {i}'
            note_value = row[note_col]
            octave_value = row[octave_col]
            duration_code = row[duration_col]
            if note_value == -1:
                continue  # Unused note
            total_notes += 1
            if note_value == 0:
                rests += 1
                continue
            midi_number = note_to_midi(scale_name, int(note_value), int(octave_value))
            if midi_number is not None:
                notes.append(midi_number)
                duration_in_beats = map_duration_code(duration_code)
                if duration_in_beats is not None:
                    durations.append(duration_in_beats)
        if len(notes) > 0:
            df.at[index, 'num_notes_played'] = len(notes)
            df.at[index, 'avg_note_duration'] = np.mean(durations)
            df.at[index, 'std_note_duration'] = np.std(durations)
            df.at[index, 'avg_pitch'] = np.mean(notes)
            df.at[index, 'pitch_range'] = max(notes) - min(notes)
        else:
            df.at[index, 'num_notes_played'] = 0
        if total_notes > 0:
            df.at[index, 'proportion_rests'] = rests / total_notes
        else:
            df.at[index, 'proportion_rests'] = 0.0

    # One-Hot Encode 'Scale'
    scale_ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    scale_encoded = scale_ohe.fit_transform(df[['Scale']])
    scale_encoded_df = pd.DataFrame(scale_encoded, columns=scale_ohe.get_feature_names_out(['Scale']))

    # One-Hot Encode 'Melody Instrument'
    melody_instr_ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    melody_instr_encoded = melody_instr_ohe.fit_transform(df[['Melody Instrument']])
    melody_instr_encoded_df = pd.DataFrame(melody_instr_encoded,
                                           columns=melody_instr_ohe.get_feature_names_out(['Melody Instrument']))

    # One-Hot Encode 'Chord Instrument'
    chord_instr_ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    chord_instr_encoded = chord_instr_ohe.fit_transform(df[['Chord Instrument']])
    chord_instr_encoded_df = pd.DataFrame(chord_instr_encoded,
                                          columns=chord_instr_ohe.get_feature_names_out(['Chord Instrument']))

    # Concatenate the new columns to the dataframe
    df = pd.concat([df, scale_encoded_df, melody_instr_encoded_df, chord_instr_encoded_df], axis=1)

    # Drop the original categorical columns

    # Define BPM bins and labels
    bpm_bins = [60, 90, 120, 150, 200, 220]
    bpm_labels = ['Slow', 'Medium', 'Fast', 'Very Fast', 'Extremely Fast']

    # Binning BPM
    df['BPM_Binned'] = pd.cut(df['BPM'], bins=bpm_bins, labels=bpm_labels, right=False)

    # One-Hot Encode the binned BPM
    bpm_ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    bpm_encoded = bpm_ohe.fit_transform(df[['BPM_Binned']])
    bpm_encoded_df = pd.DataFrame(bpm_encoded, columns=bpm_ohe.get_feature_names_out(['BPM_Binned']))

    # Concatenate and drop the 'BPM' and 'BPM_Binned' columns
    df = pd.concat([df, bpm_encoded_df], axis=1)
    df.drop(['BPM', 'BPM_Binned'], axis=1, inplace=True)



    df['avg_melodic_interval'], df['std_melodic_interval'] = zip(*df.apply(compute_melodic_intervals, axis=1))



    df['progression_complexity'] = df['Progression'].apply(compute_progression_complexity)



    # Apply the function and create a dataframe
    chord_func_values = df['Progression'].apply(encode_chord_functions)
    chord_func_df = pd.DataFrame(chord_func_values.tolist(), columns=[f'Chord_{func}' for func in chord_functions])

    # Concatenate the chord function features
    df = pd.concat([df, chord_func_df], axis=1)

    # Drop the original 'Progression' column
    df.drop(['Progression'], axis=1, inplace=True)

    df.drop(['Scale', 'Melody Instrument', 'Chord Instrument'], axis=1, inplace=True)

    # Identify numerical features for imputation
    numerical_features = ['avg_note_duration', 'std_note_duration', 'avg_pitch', 'pitch_range', 'avg_melodic_interval',
                          'std_melodic_interval']

    # Impute missing values with mean
    imputer = SimpleImputer(strategy='mean')
    df[numerical_features] = imputer.fit_transform(df[numerical_features])

    # Scale numerical features
    scaler = StandardScaler()
    df[numerical_features] = scaler.fit_transform(df[numerical_features])

    # Optionally, drop raw note features if they are not needed
    note_columns = []
    for i in range(1, 33):
        note_columns.extend([f'Note {i}', f'Octave {i}', f'Duration {i}'])
    df.drop(columns=note_columns, inplace=True)

    return df



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