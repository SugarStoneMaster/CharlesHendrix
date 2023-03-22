from collections import deque

import music21


class Key:
    all_notes = ['C4',  'C#4',  'D4',  'D#4',  'E4',  'F4',
              'F#4',  'G4',  'G#4',  'A4',  'A#4',  'B4',  'C5',
              'C#5',  'D5',  'D#5',  'E5',  'F5', 'F#5',  'G5',  'G#5',  'A5',  'A#5', 'B5']
    intervals_default = [2, 2, 1, 2, 2, 2, 1] # T T s T T T s di default per la scala maggiore
    def __init__(self, note = "C", mode = "Maj"):
        self.note = note
        self.mode = mode
        self.name = note+mode
        self.scale_notes = []
        self.distinct_notes = [] #note distinte senza ottava specificata
        self.mode_intervals = [0] #utilizzato per il calcolo degli intervalli tra due note nella fitness function
        self.chords = {} #gli accordi presenti nella scala
        intervals = deque(Key.intervals_default)
        if mode == "Maj" or "Ionian": #scala maggiore
            intervals.rotate(0)
        if mode == "Dorian": #scala dorica
            intervals.rotate(-1)
        if mode == "Phrygian": #scala frigia
            intervals.rotate(-2)
        if mode == "Lydian": #scala lidia
            intervals.rotate(-3)
        if mode == "Mixolydian": #scala misolidia
            intervals.rotate(-4)
        if mode == "Min" or mode == "Aeolian": #scala minore
            intervals.rotate(-5)
        if mode == "Locrian": #scala locria
            intervals.rotate(-6)

        for i in range(1, 8):
            self.mode_intervals.append(self.mode_intervals[i-1] + intervals[i-1])
        #trova la tonica della scala tra la lista di tutte le note
        for i, n in enumerate(Key.all_notes):
            if self.note == n[:-1]:
                break
        for j in range(8): #coprire un'intera ottava
            self.scale_notes.append(Key.all_notes[i + self.mode_intervals[j]])
        for i in range(7):
            self.distinct_notes.append((self.scale_notes[i])[:-1])
        for i in range(1, 8):
            note1 = self.distinct_notes[i-1]
            note2 = self.distinct_notes[(i+1)%7]
            note3 = self.distinct_notes[(i+3)%7]
            #note4 = self.distinct_notes[(i+5)%7] #accordo di settima
            chord = note1+"3" + ' ' + note2+"3" + ' ' + note3+"3"
            self.chords[i] = chord


class Composer:

    def __init__(self, bars=4, min_duration=16, key=Key(note="C", mode="Maj"), bad_intervals=[2, 11, 10, 1, 6], weight_notes_inchord = 0.3,
                 weight_notes_inscale = 0.65, weight_intervals = 0.2, weight_numNotes = 0.15, chromatic = True,
                 succession = [1, 1, 6, 7],
                 weight_pitchUp = 0.25,
                 weight_pitchDown = 0.25,
                 weight_sustain = 0.4,
                 weight_rest = 0.1):

        self.chromatic = chromatic #modalità di composizione
        if self.chromatic: #composizione in modalità cromatica, in cui tutte le note sono disponibili, non solo quelle della scala (consigliata per comporre Jazz)
            self.sustain_value = len(Key.all_notes)  #il valore che nella rappresentazione genetica indica il sustain (aumento di durata della nota precedente)
            self.notes = Key.all_notes #le note da utilizzare nella composizione, in questo caso tutte
            self.weight_notes_inscale = weight_notes_inscale #peso del criterio basato su quante note siano in scala (usato in modalità cromatica)
        else: #composizione con esclusivamente note della scala
            self.sustain_value = len(key.scale_notes)
            self.notes = key.scale_notes #le note da utilizzare nella composizione, in questo caso solo quelle della scala
            self.weight_notes_inscale = 0 #peso del criterio basato su quante note siano in scala (usato in modalità cromatica)

        self.rest_value = -1 #il valore che nella rappresentazione genetica indica la pausa
        self.key = key #la scala su cui deve essere basata la melodia
        self.bars = bars #il numero di battute
        self.min_duration = min_duration #la durata dell'elemento musicale più piccolo, da esprimere in potenze di 2
        self.min_duration_music21 = 1/(min_duration/4) #la min duration da associare a Music21 che ragiona a quarterlength
        self.num_music_elements = bars * min_duration #numero di geni (note, sustain, pause) in un individuo/melodia
        self.bad_intervals = bad_intervals #intervalli considerati non piacevoli, usati nel calcolo della fitness function
        self.weight_notes_inchord = weight_notes_inchord #peso del criterio basato su quante note della melodia corrispondano all'accordo suonato in quel momento
        self.weight_intervals = weight_intervals #peso del criterio basato su quanti intervalli siano piacevoli
        self.weight_numNotes = weight_numNotes #peso del criterio basato sul numero di note totali della composizione
        self.succession = succession #successione che indica l'ordine in cui suonare gli accordi della scala, la successione si ripete ogni 4 battute
        self.weight_pitchDown = weight_pitchDown #probabilità di abbassare la frequenza della nota nella mutation function
        self.weight_pitchUp = weight_pitchUp #probabilità di aumentare la frequenza della nota nella mutation function
        self.weight_sustain = weight_sustain #probabilità di allungare la durata della nota/pausa precedente nella mutation function
        self.weight_rest = weight_rest #probabilità di trasformare una nota/sustain in una pausa nella mutation function


    def in_scale(self, note: int):
        """Verifica se una data nota, rappresentata geneticamente, è contenuta nella scala scelta.
        Metodo usato nella modalità cromatica"""
        #da rappresentazione genetica a lettera della nota
        note_letter = self.notes[note]
        note_letter = note_letter[:-1]
        if note_letter in self.key.distinct_notes:
            return True
        else:
            return False


    def in_chord(self, note: int, bar_idx: int):
        """Verifica se una data nota, rappresentata geneticamente, è contenuta nell'accordo suonato alla battuta di indice bar_idx"""
        #da rappresentazione genetica a stringa dell'accordo con le frequenze delle note
        grade = self.succession[bar_idx%4] #la successione si ripete ogni 4 battute
        chord = self.key.chords[grade]
        #vengono rimosse le ottave per lasciare esclusivamente le note dell'accordo
        chord_noPitch = ''.join([i for i in chord if not i.isdigit()])
        listChord = chord_noPitch.split(' ')
        #da rappresentazione genetica a lettera della nota
        noteLetter = self.notes[note]
        noteLetter = noteLetter[:-1]

        if noteLetter in listChord:
            return True
        else:
            return False


    def toMusic21(self, solution) -> music21.stream.Stream:
        """Trasforma una rappresentazione genetica di una melodia in uno stream riproducibile di Music21"""
        stream = music21.stream.Stream()
        for i in range(len(solution)):
            gene = solution[i]
            if gene != self.sustain_value and gene != self.rest_value:
                note = music21.note.Note(self.notes[gene])
            if gene == self.rest_value:
                note = music21.note.Rest()
            if gene == self.sustain_value:
                continue

            note.duration.quarterLength = self.min_duration_music21
            if i+1 != len(solution):
                j = i+1
                next_gene = solution[j]
                while(next_gene == self.sustain_value):
                    note.duration.quarterLength += self.min_duration_music21
                    if j+1 != len(solution):
                        j += 1
                        next_gene = solution[j]
                    else:
                        break

            stream.append(note)

        return stream


    def addChordsToMusic21(self, stream: music21.stream.Stream) -> music21.stream.Stream:
        """Aggiunge accordi secondo la successione ad una melodia rappresentata come stream Music21"""
        for bar_idx in range(self.bars):
            chord = music21.chord.Chord(self.key.chords[self.succession[bar_idx%4]]) #la successione si ripete ogni 4 battute
            chord.quarterLength = 4 #l'accordo viene suonato per l'intera battuta
            stream.insert(float(4*bar_idx), chord)

        stream.clef = music21.clef.TrebleClef() #per visualizzare la composizione in chiave di violino
        return stream


    def toGeneticFromXML(self, filename: str) -> list:
        """Effettua il parsing di un file .musicxml solo melodia, passando per Music21, per creare un individuo genetico"""
        stream = music21.converter.parse(filename) #da file .musicxml a stream di Music21
        list = []
        stream = stream[1]
        for i in range(1, self.bars+1):
            bar = stream[i]
            for note in bar.getElementsByClass(['Note', 'Rest']):
                duration = note.duration.quarterLength - self.min_duration_music21
                if isinstance(note, music21.note.Note):
                    name = note.name + str(note.octave)
                    gene = self.notes.index(name)
                else:
                    gene = self.rest_value

                list.append(gene)

                while(duration != 0):
                    list.append(self.sustain_value)
                    duration -= self.min_duration_music21

        return list


    def toGeneticFromStream(self, stream: music21.stream.Stream) -> list:
        """Effettua il parsing di uno stream Music21 solo melodia per creare un individuo genetico"""
        list = []
        for note in stream.getElementsByClass(['Note', 'Rest']):
            duration = note.duration.quarterLength - self.min_duration_music21
            if isinstance(note, music21.note.Note):
                name = note.name + str(note.octave)
                gene = self.notes.index(name)
            else:
                gene = self.rest_value

            list.append(gene)

            while(duration != 0):
                list.append(self.sustain_value)
                duration -= self.min_duration_music21

        return list





