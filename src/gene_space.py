import music21

scale_mapping = {
    0: 'C',   # C major
    1: 'c',   # C minor
    2: 'C#',  # C# major
    3: 'c#',  # C# minor
    4: 'D',   # D major
    5: 'd',   # D minor
    6: 'D#',  # D# major
    7: 'd#',  # D# minor
    8: 'E',   # E major
    9: 'e',   # E minor
    10: 'F',  # F major
    11: 'f',  # F minor
    12: 'F#', # F# major
    13: 'f#', # F# minor
    14: 'G',  # G major
    15: 'g',  # G minor
    16: 'G#', # G# major
    17: 'g#', # G# minor
    18: 'A',  # A major
    19: 'a',  # A minor
    20: 'A#', # A# major
    21: 'a#', # A# minor
    22: 'B',  # B major
    23: 'b',  # B minor
}
num_scales = len(scale_mapping)


# Melody Instrument Mapping (All Instruments)
melody_instrument_mapping = {
    0: music21.instrument.Piano,
    1: music21.instrument.Violin,
    2: music21.instrument.Flute,
    3: music21.instrument.MezzoSoprano,
    4: music21.instrument.Saxophone,
    5: music21.instrument.Clarinet,
    6: music21.instrument.ElectricGuitar,
    7: music21.instrument.Xylophone,
}
num_melody_instruments = len(melody_instrument_mapping)

# Chord Instrument Mapping (Instruments Suitable for Chords)
chord_instrument_mapping = {
    0: music21.instrument.Piano,
    1: music21.instrument.AcousticGuitar,
    2: music21.instrument.ElectricGuitar,
    3: music21.instrument.Organ,
    4: music21.instrument.Sitar
}
num_chord_instruments = len(chord_instrument_mapping)


# Durations in quarter lengths (1 quarter length = 1 beat)
duration_values = {
    0: 0.25,  # Sixteenth note
    1: 0.5,   # Eighth note
    2: 1.0,   # Quarter note
    3: 1.5,   # Dotted quarter note
    4: 2.0,   # Half note
    5: 3.0,   # Dotted half note
    6: 4.0    # Whole note
}
num_durations = len(duration_values)

# Chord mappings for major keys
major_chord_type_mapping = {
    1: 'I',
    2: 'ii',
    3: 'iii',
    4: 'IV',
    5: 'V',
    6: 'vi',
    7: 'viio',
    8: 'Rest'
}
# Chord mappings for minor keys
minor_chord_type_mapping = {
    1: 'i',
    2: 'iio',
    3: 'III',
    4: 'iv',
    5: 'v',
    6: 'VI',
    7: 'VII',
    8: 'Rest'
}

# Major Key Progressions
major_chord_progressions = {
    0: [1, 5, 6, 4],      # I - V - vi - IV
    1: [1, 4, 5, 1],      # I - IV - V - I
    2: [1, 6, 4, 5],      # I - vi - IV - V
    3: [1, 4, 1, 5],      # I - IV - I - V
    4: [1, 2, 5, 1],      # I - V/V - V - I
    5: [1, 6, 2, 5],      # I - vi - ii - V
    6: [1, 4, 6, 5],      # I - IV - vi - V
    7: [1, 5, 4, 5],      # I - V - IV - V
}

# Minor Key Progressions
minor_chord_progressions = {
    0: [1, 6, 3, 7],      # i - VI - III - VII
    1: [1, 4, 5, 1],      # i - iv - v - i
    2: [1, 7, 6, 7],      # i - VII - VI - VII
    3: [1, 4, 7, 3],      # i - iv - VII - III
    4: [1, 5, 6, 4],      # i - v - VI - iv
    5: [1, 6, 4, 5],      # i - VI - iv - v
    6: [1, 5, 1, 4],      # i - v - i - iv
    7: [1, 2, 5, 1],      # i - ii° - v - i
}
num_progressions = len(major_chord_progressions)

max_melody_notes = 64  # Maximum number of melody notes
max_chords = 4        # Maximum number of chords

total_genes = 5 + (max_melody_notes * 3) + max_chords

scale_degree_values = list(range(0, 8))  # 0 (Rest) to 7 (7th scale degree)
octave_values = [4, 5]

# Global parameters
gene_space = []

gene_space.append({'low': 0, 'high': num_scales})                # Scale
gene_space.append({'low': 65, 'high': 215})                          # BPM (Tempo)
gene_space.append({'low': 0, 'high': num_melody_instruments})    # Melody Instrument
gene_space.append({'low': 0, 'high': num_chord_instruments})     # Chord Instrument
gene_space.append({'low': 0, 'high': num_progressions})          # Chord Progression

# Melody notes: Scale Degree, Octave, Duration
for _ in range(max_melody_notes):
    gene_space.append({'low': 0, 'high': 8})                        # Scale Degree (0-7)
    gene_space.append({'low': 4, 'high': 6})                        # Octave (4 or 5)
    gene_space.append({'low': 0, 'high': num_durations})        # Note Duration


# Chord durations (no longer need chord types)
for _ in range(max_chords):
    gene_space.append({'low': 0, 'high': num_durations})        # Chord Duration