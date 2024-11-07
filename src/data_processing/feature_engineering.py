import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from src.data_processing.utility import note_to_midi, map_duration_code, compute_melodic_intervals, \
    compute_progression_complexity, encode_chord_functions, chord_functions

"""
   Data cleaning: rimuovere le colonne con i -1 a partire dalla riga con più note. NO IMPUTATION
   Feature construction: Dimensionality reduction PCA e/oppure costruzione di feature derivate ad esempio numero di note suonate, durata media, etc.
   Feature scaling: scaling di features numeriche come numero di note suonate, durata media, etc.
   Feature selection: una volta costruite le features vengono droppate le note. DA FARE low-variance e univariate
   Data balancing: SMOTE
"""


def data_feature_engineering(df):
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


    df.to_csv(path_or_buf='../data/processed/feature_engineering_notes.csv', index=False)

    return df


def feature_construction(df):
    return


def split(df, smote=False):
    # Convert 'UserInput' to categorical data
    y = df['UserInput']
    df.drop(['UserInput'], axis=1, inplace=True)
    X = df

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    if smote:
        smote = SMOTE(random_state=42)
        X_train, y_train = smote.fit_resample(X_train, y_train)

    return X_train, y_train, X_test, y_test
