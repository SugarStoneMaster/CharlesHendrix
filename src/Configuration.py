class Key:
    pass


class CMajKey(Key):
    name = 'CMaj'
    name_relative = 'AMin'
    allowed_notes = ['C', 'D', 'E', 'F', 'G', 'A', 'B']


class ComposerConfig:

    def __init__(self, break_value=0, notes={'1': 'C4', '2': 'C#4', '3': 'D4', '4': 'D#4', '5': 'E4', '6': 'F4',
                                                 '7': 'F#4', '8': 'G4', '9': 'G#4', '10': 'A4', '11': 'A#4', '12': 'B4', '13': 'C5'},
                 repeat_value=14, bars=4, min_duration=16, key=CMajKey, bad_intervals=[2, 11, 10, 1, 6]):
        self.break_value = break_value
        self.notes = notes
        self.repeat_value = repeat_value
        self.bars = bars
        self.min_duration = min_duration
        self.num_notes = bars * min_duration
        self.key = key
        self.bad_intervals = bad_intervals

    def in_scale(self, note: int):
        noteLetter = self.notes[str(note)]
        noteLetter = noteLetter[:-1]
        if noteLetter in self.key.allowed_notes:
            return True
        else:
            return False








