import music21

class Key:
    pass


class CMajKey(Key):
    name = 'CMaj'
    name_relative = 'AMin'
    allowed_notes = ['C', 'D', 'E', 'F', 'G', 'A', 'B']


class EMinKey(Key):
    name = 'EMin'
    name_relative = 'GMaj'
    allowed_notes = ['E', 'F#', 'G', 'A', 'B', 'C', 'D']


class ComposerConfig:

    def __init__(self, break_value=-1, notes={0: 'B3', 1: 'C4', 2: 'C#4', 3: 'D4', 4: 'D#4', 5: 'E4', 6: 'F4',
                                                 7: 'F#4', 8: 'G4', 9: 'G#4', 10: 'A4', 11: 'A#4', 12: 'B4', 13: 'C5',
                                             14: 'C#5', 15: 'D5', 16: 'D#5', 17: 'E5', 18: 'F5'},
                 repeat_value=19, bars=4, min_duration=16, key=CMajKey, bad_intervals=[2, 11, 10, 1, 6],
                 weight_notes_inscale = 0.65, weight_intervals = 0.2, weight_numNotes = 0.15):
        self.break_value = break_value
        self.notes = notes
        self.repeat_value = repeat_value
        self.bars = bars
        self.min_duration = min_duration
        self.min_duration_music21 = 1/(min_duration/4)
        self.num_notes = bars * min_duration
        self.key = key
        self.bad_intervals = bad_intervals
        self.weight_notes_inscale = weight_notes_inscale
        self.weight_intervals = weight_intervals
        self.weight_numNotes = weight_numNotes

    def in_scale(self, note: int):
        noteLetter = self.notes[note]
        noteLetter = noteLetter[:-1]
        if noteLetter in self.key.allowed_notes:
            return True
        else:
            return False

    def toMusic21(self, solution) -> music21.stream.Stream:
        stream = music21.stream.Stream()
        for i in range(len(solution)):
            gene = solution[i]
            if gene != self.repeat_value and gene != self.break_value:
                note = music21.note.Note(self.notes[gene])
            if gene == self.break_value:
                note = music21.note.Rest()
            if gene == self.repeat_value:
                continue

            note.duration.quarterLength = self.min_duration_music21
            if i+1 != len(solution):
                j = i+1
                next_gene = solution[j]
                while(next_gene == self.repeat_value):
                    note.duration.quarterLength += self.min_duration_music21
                    if j+1 != len(solution):
                        j += 1
                        next_gene = solution[j]
                    else:
                        break

            stream.append(note)

        return stream

    def toGenetic(self, stream: music21.stream.Stream) -> list:
        list = []
        stream = stream[1]
        for i in range(1, self.bars+1):
            bar = stream[i]
            print(bar)
            for note in bar.getElementsByClass(['Note', 'Rest']):
                duration = note.duration.quarterLength - self.min_duration_music21
                print("{} {}".format(note, note.duration.quarterLength))
                if isinstance(note, music21.note.Note):
                    name = note.name + str(note.octave)
                    gene = [k for k, v in self.notes.items() if v == name][0]
                else: #rest
                    gene = self.break_value
                list.append(gene)
                while(duration != 0):
                    list.append(self.repeat_value)
                    duration -= self.min_duration_music21

        return list






