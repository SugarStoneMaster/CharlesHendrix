import music21.stream

import src.Configuration


def setNotesFitnessFunction(configuration: src.Configuration.ComposerConfig):
    def fitness_function(solution, solution_idx):
        bad_intervals = 0
        total_intervals = 0
        inscale_notes = 0
        total_notes = 0
        note1_to_save = None
        if solution[0] == configuration.repeat_value: #repeat value at index 0, it makes no sense, there is no note before
            solution[0] = configuration.break_value
        for i in range(configuration.num_notes):
            if note1_to_save is None:
                note1 = solution[i]
                if note1 == configuration.break_value or note1 == configuration.repeat_value:
                    continue
                if configuration.in_scale(note1):
                    inscale_notes += 1
                total_notes += 1
            else:
                note1 = note1_to_save
                note1_to_save = None

            if i != configuration.num_notes - 1:
                note2 = solution[i+1]

            if note2 == configuration.break_value or note2 == configuration.repeat_value:
                note1_to_save = note1
                continue

            interval = abs(note1 - note2)
            if interval in configuration.bad_intervals:
                bad_intervals += 1
            total_intervals += 1

        total_intervals -= 1 #da fixare conteggio intervalli
        try:
            ratio_notes_inscale = inscale_notes/total_notes
        except ZeroDivisionError:
            ratio_notes_inscale = 0
        try:
            ratio_intervals = bad_intervals/total_intervals
        except ZeroDivisionError:
            ratio_intervals = 0
        try:
            ratio_numNotes = total_notes/configuration.num_notes
        except ZeroDivisionError:
            ratio_numNotes = 0
        fitness = configuration.weight_notes_inscale*ratio_notes_inscale + configuration.weight_intervals*ratio_intervals + configuration.weight_numNotes*ratio_numNotes

        return fitness

    def NEWfitness_function(solution, solution_idx):
        bad_intervals = 0
        total_intervals = 0
        total_notes = 0
        total_notes_inChord = 0
        note1_to_save = None
        if solution[0] == configuration.repeat_value: #repeat value at index 0, it makes no sense, there is no note before
            solution[0] = configuration.break_value
        for i in range(configuration.num_notes):
            if note1_to_save is None:
                note1 = solution[i]
                if note1 == configuration.break_value or note1 == configuration.repeat_value:
                    continue

                total_notes += 1
            else:
                note1 = note1_to_save
                note1_to_save = None

            if i != configuration.num_notes - 1:
                note2 = solution[i+1]

            if note2 == configuration.break_value or note2 == configuration.repeat_value:
                note1_to_save = note1
                continue


            bar_idx = i // configuration.min_duration
            if configuration.in_chord(note1, bar_idx):
                total_notes_inChord += 1

            interval = abs(configuration.key.intervals[note1+1] - configuration.key.intervals[note2+1])
            if interval in configuration.bad_intervals:
                bad_intervals += 1
            total_intervals += 1

        total_intervals -= 1 #da fixare conteggio intervalli
        try:
            ratio_inChord = total_notes_inChord/total_notes
        except ZeroDivisionError:
            ratio_inChord = 0
        try:
            ratio_intervals = bad_intervals/total_intervals
        except ZeroDivisionError:
            ratio_intervals = 0
        try:
            ratio_numNotes = total_notes/configuration.num_notes
        except ZeroDivisionError:
            ratio_numNotes = 0

        fitness = configuration.weight_notes_inchord*ratio_inChord + \
                  configuration.weight_intervals*ratio_intervals + \
                  configuration.weight_numNotes*ratio_numNotes

        return fitness



    if configuration.chromatic:
        return fitness_function
    else:
        return NEWfitness_function


def setChordsFitnessFunction(configuration: src.Configuration.ComposerConfig):
    def fitness_function(solution, solution_idx):
        stream = music21.stream.Stream()
        print(solution)
        for chord_idx in solution:
            chord = configuration.key.chords[chord_idx]
            chordMusic21 = music21.chord.Chord(chord)
            stream.append(chordMusic21)

        stream.show()
        fitness = int(input("Fitness (1 a 10): "))

        return fitness

    return fitness_function