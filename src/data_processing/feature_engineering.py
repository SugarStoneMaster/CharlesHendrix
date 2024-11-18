import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from src.data_processing.utility import note_to_midi, map_duration_code, compute_melodic_intervals, \
    compute_progression_complexity, encode_chord_functions, chord_functions
from src.genetic_algorithm.gene_space import scale_mapping, melody_instrument_mapping, chord_instrument_mapping, \
    minor_chord_progressions, minor_chord_type_mapping, major_chord_progressions, major_chord_type_mapping
from joblib import dump, load
from sklearn.preprocessing import StandardScaler


def data_feature_engineering(df, mode='fit', scaler=None, column_structure=None):
    if 'UserInput' in df.columns or mode == 'fit':
        y = df['UserInput']  # Target column
        X = df.drop(columns=['UserInput'])  # Feature columns
    else:
        y = None
        X = df

    X = data_cleaning(X)
    X = feature_construction(X)

    if mode == 'fit':
        # For the entire dataset
        X = feature_scaling(X)  # Scaler is fitted here
    elif mode == 'transform':
        # For individual rows
        if scaler is None or column_structure is None:
            raise ValueError("Scaler and columns must be provided in transform mode.")
        X = feature_scaling(X, scaler=scaler, column_structure=column_structure)
    else:
        raise ValueError("Mode must be 'fit' or 'transform'.")

    X = feature_selection(X)

    if y is not None:
        df_processed = pd.concat([X, y.reset_index(drop=True)], axis=1)
        #df_processed.to_csv("./completo.csv", index=False)
    else:
        df_processed = X

    return df_processed

def data_cleaning(df):
    """# Identify numerical features for imputation


        # Impute missing values with mean
        imputer = SimpleImputer(strategy='mean')
        df[numerical_features] = imputer.fit_transform(df[numerical_features])"""


    return df

def feature_construction(df):
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


    df['avg_melodic_interval'], df['std_melodic_interval'] = zip(*df.apply(compute_melodic_intervals, axis=1))

    df['progression_complexity'] = df['Progression'].apply(compute_progression_complexity)

    bpm_bins = [60, 90, 120, 150, 200, 220]
    bpm_labels = ['Slow', 'Medium', 'Fast', 'Very Fast', 'Extremely Fast']
    df['BPM_Binned'] = pd.cut(df['BPM'], bins=bpm_bins, labels=bpm_labels, right=False)

    return df


def feature_scaling(df, scaler=None, column_structure=None):
    # First, encode the 'Progression' column into chord function features
    chord_func_values = df['Progression'].apply(encode_chord_functions)
    chord_func_df = pd.DataFrame(chord_func_values.tolist(), columns=[f'Chord_{func}' for func in chord_functions])
    df = pd.concat([df, chord_func_df], axis=1)

    # One-hot encode categorical features
    categorical_features = ['Scale', 'Melody Instrument', 'Chord Instrument', 'BPM_Binned']
    df = pd.get_dummies(df, columns=categorical_features)

    numerical_features = [
        'avg_note_duration',
        'std_note_duration',
        'avg_pitch',
        'pitch_range',
        'avg_melodic_interval',
        'std_melodic_interval',
    ]

    if scaler is None or column_structure is None:
        # This is the 'fit' case: we are fitting the model and need to fit the scaler and define the column structure

        # Fit the scaler on numerical features and transform them
        scaler = StandardScaler()
        df[numerical_features] = scaler.fit_transform(df[numerical_features])

        # Save the scaler to a file
        dump(scaler, "/Users/carmine/PycharmProjects/CharlesHendrix/model/scaler.joblib")

        # Save the column structure
        column_structure = df.columns
        dump(column_structure, "/Users/carmine/PycharmProjects/CharlesHendrix/model/columns.joblib")

    else:
        # This is the 'transform' case: we need to apply the existing scaler and ensure the columns match

        # Transform the numerical features using the provided scaler
        # Check if all numerical features are present
        missing_numerical = [col for col in numerical_features if col not in df.columns]
        if missing_numerical:
            raise ValueError(f"The following numerical features are missing in the input data: {missing_numerical}")
        df[numerical_features] = scaler.transform(df[numerical_features])

        # Align the dataframe columns with the original column structure
        # Add any missing columns with appropriate default values
        missing_cols = set(column_structure) - set(df.columns)
        for col in missing_cols:
            if col.startswith('Chord_') and not col.startswith('Chord Instrument_'):
                # For chord function columns (e.g., 'Chord_I', 'Chord_iv'), set default value to 0
                df[col] = 0
            elif (
                    col.startswith('Scale_')
                    or col.startswith('Melody Instrument_')
                    or col.startswith('Chord Instrument_')
                    or col.startswith('BPM_Binned_')
            ):
                # For one-hot encoded categorical columns, set default value to False
                df[col] = False
            else:
                # For other columns, set default value to 0 or appropriate value
                df[col] = 0  # Adjust as needed

        # Remove any extra columns not in the original column structure
        extra_cols = set(df.columns) - set(column_structure)
        if extra_cols:
            df.drop(columns=extra_cols, inplace=True)

        # Reorder columns to match the original column structure
        df = df[column_structure]

    return df

def feature_selection(df):
    # Drop the original 'Progression' column
    df.drop(['Progression'], axis=1, inplace=True)

    df.drop(['BPM'], axis=1, inplace=True)

    note_columns = []
    for i in range(1, 33):
        note_columns.extend([f'Note {i}', f'Octave {i}', f'Duration {i}'])
    df.drop(columns=note_columns, inplace=True)

    return df


def split(df, smote=False):
    # Convert 'UserInput' to categorical data
    y = df['UserInput']
    X = df.drop(['UserInput'], axis=1)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    if smote:
        smote = SMOTE(random_state=42)
        X_train, y_train = smote.fit_resample(X_train, y_train)

    return X_train, X_test, y_train, y_test


def solution_to_features(solution, scaler, column_structure):

    df_solution = genome_to_data(solution)

    df_solution_processed = data_feature_engineering(df_solution, mode='transform', scaler=scaler, column_structure=column_structure)

    return df_solution_processed


def genome_to_data(solution):
    # Map genome to feature names
    scale_idx = int(solution[0])
    scale_name = scale_mapping.get(scale_idx, f"Unknown ({scale_idx})")
    bpm = solution[1]
    melody_instr_idx = int(solution[2])
    melody_instrument = melody_instrument_mapping.get(melody_instr_idx,
                                                      f"Unknown ({melody_instr_idx})")().instrumentName
    chord_instr_idx = int(solution[3])
    chord_instrument = chord_instrument_mapping.get(chord_instr_idx, f"Unknown ({chord_instr_idx})")().instrumentName

    progression_idx = int(solution[4])
    if scale_name.islower():
        progression = minor_chord_progressions.get(progression_idx, [])
        chord_type_mapping = minor_chord_type_mapping
    else:
        progression = major_chord_progressions.get(progression_idx, [])
        chord_type_mapping = major_chord_type_mapping
    progression_chords = [chord_type_mapping.get(chord_degree, f"Unknown ({chord_degree})") for chord_degree in
                          progression]
    progression_feature = "-".join(progression_chords)

    # Note, Octave, Duration conversion
    notes = solution[5:]
    note_features = []
    for i in range(0, len(notes), 3):
        note, octave, duration = notes[i:i + 3]
        note_features.extend([note, octave, duration])

    # Create a DataFrame from mapped values
    data = {
        "Scale": [scale_name],
        "BPM": [bpm],
        "Melody Instrument": [melody_instrument],
        "Chord Instrument": [chord_instrument],
        "Progression": [progression_feature],
    }
    for idx in range(1, len(note_features) // 3 + 1):
        data[f"Note {idx}"], data[f"Octave {idx}"], data[f"Duration {idx}"] = note_features[
                                                                              (idx - 1) * 3:(idx - 1) * 3 + 3]

    df_solution = pd.DataFrame(data)
    return df_solution
