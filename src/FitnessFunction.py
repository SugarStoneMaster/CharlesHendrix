import src.Configuration


def config(configuration: src.Configuration.ComposerConfig):
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
        ratio_notes = inscale_notes/total_notes
        ratio_intervals = bad_intervals/total_intervals
        ratio_numNotes = total_notes/configuration.num_notes
        #fitness = ratio_notes  #tutte note in scala
        fitness = 0.8*ratio_notes + 0.15*ratio_intervals + 0.05*ratio_numNotes

        return fitness

    return fitness_function
