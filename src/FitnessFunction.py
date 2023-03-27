import music21.stream

import src.Configuration


def setMelodyFitnessFunction(configuration: src.Configuration.Composer):
    def fitness_function(solution, solution_idx):
        bad_intervals = 0
        total_intervals = 0
        total_notes_inChord = 0
        total_notes_inScale = 0
        total_notes = 0

        note1_to_save = None
        if solution[0] == configuration.sustain_value: #valore di sustain all'indice 0, non c'è nessuna nota precedente da allungare
            solution[0] = configuration.rest_value #viene dunque messa una pausa
        for i in range(configuration.num_music_elements):
            if note1_to_save is None:
                note1 = solution[i]
                if note1 == configuration.rest_value or note1 == configuration.sustain_value:
                    continue
                if configuration.in_scale(note1):
                    total_notes_inScale += 1

                total_notes += 1
            else:
                note1 = note1_to_save
                note1_to_save = None



            if i != configuration.num_music_elements - 1:
                note2 = solution[i+1]
            else:
                note2 = "break"

            if note2 == configuration.rest_value or note2 == configuration.sustain_value:
                note1_to_save = note1
                continue
            elif note2 == "break":
                break

            bar_idx = i // configuration.min_duration
            if configuration.in_chord(note1, bar_idx):
                total_notes_inChord += 1

            if configuration.chromatic:
                interval = abs(note1 - note2)
            else:
                interval = abs(configuration.key.mode_intervals[note1] - configuration.key.mode_intervals[note2])
            if interval in configuration.bad_intervals:
                bad_intervals += 1
            total_intervals += 1

        if configuration.chromatic:
            try:
                ratio_inScale = total_notes_inScale/total_notes
            except ZeroDivisionError:
                ratio_inScale = 0
        else:
            ratio_inScale = 0
        try:
            ratio_inChord = total_notes_inChord/total_notes
        except ZeroDivisionError:
            ratio_inChord = 0
        try:
            good_intervals = total_intervals - bad_intervals
            ratio_intervals = good_intervals/total_intervals
        except ZeroDivisionError:
            ratio_intervals = 0
        try:
            ratio_numNotes = total_notes/configuration.num_music_elements
        except ZeroDivisionError:
            ratio_numNotes = 0

        fitness = configuration.weight_notes_inchord*ratio_inChord + \
                  configuration.weight_intervals*ratio_intervals + \
                  configuration.weight_numNotes*ratio_numNotes + \
                  configuration.weight_notes_inscale*ratio_inScale #avrà peso solo in modalità cromatica

        return fitness


    return fitness_function


def setChordsFitnessFunction(configuration: src.Configuration.Composer):
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