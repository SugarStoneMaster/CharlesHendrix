import music21


class Key:
    notes = {0: 'B3', 1: 'C4', 2: 'C#4', 3: 'D4', 4: 'D#4', 5: 'E4', 6: 'F4',
             7: 'F#4', 8: 'G4', 9: 'G#4', 10: 'A4', 11: 'A#4', 12: 'B4', 13: 'C5',
             14: 'C#5', 15: 'D5', 16: 'D#5', 17: 'E5', 18: 'F5'}


class MajKey(Key):
    intervals = {1: 0, 2: 2, 3: 4, 4: 5, 5: 7, 6: 9, 7: 11}


class MinKey(Key):
    intervals = {1: 0, 2: 2, 3: 3, 4: 5, 5: 7, 6: 8, 7: 10}


class CMajKey(MajKey):
    name = 'CMaj'
    name_relative = 'AMin'
    allowed_notes = ['C', 'D', 'E', 'F', 'G', 'A', 'B']


class EMinKey(MinKey):
    name = 'EMin'
    name_relative = 'GMaj'
    allowed_notes = ['E', 'F#', 'G', 'A', 'B', 'C', 'D'] #for chromatic = True
    notes = ['E4', 'F#4', 'G4', 'A4', 'B4', 'C5', 'D5']
    chords = {1: "E2 G2 B2", 2: "F#2 A2 C3", 3: "G2 B2 D3", 4: "A1 C2 E2", 5: "B1 D2 F#2", 6: "C2 E2 G2", 7: "D2 F#2 A2"}



class ComposerConfig:

    def __init__(self, bars=4, min_duration=16, key=CMajKey, bad_intervals=[2, 11, 10, 1, 6], weight_notes_inchord = 0.3,
                 weight_notes_inscale = 0.65, weight_intervals = 0.2, weight_numNotes = 0.15, chromatic = True,
                 succession = [1, 1, 6, 7],
                 weight_pitchUp = 0.25,
                 weight_pitchDown = 0.25,
                 weight_sustain = 0.4,
                 weight_break = 0.1):
        if chromatic:
            self.break_value = min(Key.notes.keys()) - 1
            self.repeat_value = max(Key.notes.keys()) + 1
            self.notes = Key.notes
        else:
            self.break_value = -1
            self.repeat_value = len(key.notes)
            self.notes = key.notes

        self.key = key
        self.bars = bars
        self.min_duration = min_duration
        self.min_duration_music21 = 1/(min_duration/4)
        self.num_notes = bars * min_duration #number of genes in individual
        self.bad_intervals = bad_intervals
        self.weight_notes_inchord = weight_notes_inchord
        self.weight_notes_inscale = weight_notes_inscale
        self.weight_intervals = weight_intervals
        self.weight_numNotes = weight_numNotes
        self.chromatic = chromatic
        self.succession = succession
        self.weight_pitchDown = weight_pitchDown
        self.weight_pitchUp = weight_pitchUp
        self.weight_sustain = weight_sustain
        self.weight_break = weight_break

    def in_scale(self, note: int):
        noteLetter = self.notes[note]
        noteLetter = noteLetter[:-1]
        if noteLetter in self.key.allowed_notes:
            return True
        else:
            return False


    def in_chord(self, note: int, bar_idx: int):
        grade = self.succession[bar_idx%4]
        chord = self.key.chords[grade]
        chord_noPitch = ''.join([i for i in chord if not i.isdigit()])
        listChord = chord_noPitch.split(' ')

        noteLetter = self.notes[note]
        noteLetter = noteLetter[:-1]

        if noteLetter in listChord:
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

    def addChordsToMusic21(self, stream: music21.stream.Stream) -> music21.stream.Stream:
        for bar_idx in range(self.bars):
            chord = music21.chord.Chord(self.key.chords[self.succession[bar_idx%4]])
            chord.quarterLength = 4
            stream.insert(float(4*bar_idx), chord)

        stream.clef = music21.clef.TrebleClef() #chiave di violino
        return stream


    def toGeneticFromXML(self, filename: str) -> list:
        stream = music21.converter.parse(filename)
        list = []
        stream = stream[1]
        for i in range(1, self.bars+1):
            bar = stream[i]
            for note in bar.getElementsByClass(['Note', 'Rest']):
                duration = note.duration.quarterLength - self.min_duration_music21
                if isinstance(note, music21.note.Note):
                    name = note.name + str(note.octave)
                    if self.chromatic:
                        gene = [k for k, v in self.notes.items() if v == name][0]
                    else:
                        gene = self.notes.index(name)
                else: #rest
                    gene = self.break_value
                list.append(gene)
                while(duration != 0):
                    list.append(self.repeat_value)
                    duration -= self.min_duration_music21

        return list


    def toGeneticFromStream(self, stream: music21.stream.Stream) -> list:
        list = []
        for note in stream.getElementsByClass(['Note', 'Rest']):
            duration = note.duration.quarterLength - self.min_duration_music21
            if isinstance(note, music21.note.Note):
                name = note.name + str(note.octave)
                if self.chromatic:
                    gene = [k for k, v in self.notes.items() if v == name][0]
                else:
                    gene = self.notes.index(name)
            else: #rest
                gene = self.break_value
            list.append(gene)
            while(duration != 0):
                list.append(self.repeat_value)
                duration -= self.min_duration_music21

        return list





