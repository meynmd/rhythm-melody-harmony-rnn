from itertools import *
import os
import math
import random
import sys
import glob
import pickle
import torch
import music21

ValidFraction = 0.2
TestFraction = 0.2

music21.environment.UserSettings()['warnings'] = 0

class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []


    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]


    def __len__(self):
        return len(self.idx2word)


class Corpus(object):
    def __init__(self, corpus_dir = None, composer = None, meter = 4,
                 save = ('train-corpus.m21', 'valid-corpus.m21', 'test-corpus.m21'),
                 dict_save = ('dict_dur', 'dict_mel', 'dict_har')):

        self.duration_dictionary = Dictionary()
        self.melodic_dictionary = Dictionary()
        self.harmonic_dictionary = Dictionary()
        self.chords_dictionary = Dictionary()
        self.meter = meter

        if composer is not None or corpus_dir is not None:
            if corpus_dir is not None:
                print('Loading corpus from directory: {}'.format(corpus_dir))

            else:
                print('Loading corpus for composer: {}'.format(composer))
                train, valid, test = self.read_corpus(composer)

            self.duration_train, self.melodic_train, self.harmonic_train = self.encode(train)
            self.duration_valid, self.melodic_valid, self.harmonic_valid = self.encode(valid)
            self.duration_test, self.melodic_test, self.harmonic_test = self.encode(test)

            with open(save[0], 'wb') as trainfile:
                pickle.dump((self.duration_train, self.melodic_train, self.harmonic_train), trainfile)
            with open(save[1], 'wb') as validfile:
                pickle.dump((self.duration_valid, self.melodic_valid, self.harmonic_valid), validfile)
            with open(save[2], 'wb') as testfile:
                pickle.dump((self.duration_test, self.melodic_test, self.harmonic_test), testfile)
            with open(dict_save[0], 'wb') as dictfile:
                pickle.dump(self.duration_dictionary, dictfile)
            with open(dict_save[1], 'wb') as dictfile:
                pickle.dump(self.melodic_dictionary, dictfile)
            with open(dict_save[2], 'wb') as dictfile:
                pickle.dump(self.harmonic_dictionary, dictfile)

            print('(Duration, Melodic, Harmonic) sizes')
            print(len(self.duration_train), len(self.melodic_train), len(self.harmonic_train))
            print(len(self.duration_valid), len(self.melodic_valid), len(self.harmonic_valid))
            print(len(self.duration_test), len(self.melodic_test), len(self.harmonic_test))

        else:
            self.duration_train, self.melodic_train, self.harmonic_train = self.unpickle_corpus(save[0])
            self.duration_valid, self.melodic_valid, self.harmonic_valid = self.unpickle_corpus(save[1])
            self.duration_test, self.melodic_test, self.harmonic_test = self.unpickle_corpus(save[2])
            with open(dict_save[0], 'r') as fp:
                self.duration_dictionary = pickle.load(fp)
            with open(dict_save[1], 'r') as fp:
                self.melodic_dictionary = pickle.load(fp)
            with open(dict_save[2], 'r') as fp:
                self.harmonic_dictionary = pickle.load(fp)

    def read_corpus(self,
            composer,
            max_size = 10000,
            save = ('train-corpus.m21', 'valid-corpus.m21', 'test-corpus.m21')):

        print('Reading corpus from file...')
        self.corpus = music21.corpus.getComposer(composer)

        if len(self.corpus) < 1:
            print('Error: cannot load any works by composer {}'.format(composer))
            exit(1)

        train, valid, test = self.allocate_corpus()
        train = self.parse_corpus(train, max_size)
        valid = self.parse_corpus(valid, max_size)
        test = self.parse_corpus(test, max_size)

        return train, valid, test


    def parse_corpus(self, files, max_size):
        corpus = []
        for f in files:
            s = music21.corpus.parse(f)
            if s.parts[0].measure(1).barDuration.quarterLength == self.meter:
                corpus.append(s)
                print(s.metadata.title)
        if len(corpus) <= max_size:
            return corpus
        else:
            return corpus[: max_size]


    def unpickle_corpus(self, load):
        with open(load, 'r') as corpus_file:
            duration, melodic, harmonic = pickle.load(corpus_file)
        return duration, melodic, harmonic



    def old_allocate_corpus(self):
        corpus = self.corpus
        train, test, valid = [], [], []
        num_valid, num_test = len(corpus) // ValidFraction, len(corpus) // TestFraction
        for file in corpus:
            score = music21.corpus.parse(file)
            if len(valid) < num_valid and random.random() < ValidFraction:
                valid.append(score)
            elif len(valid) < num_valid and random.random() < TestFraction:
                test.append(score)
            else:
                train.append(score)
        return train, valid, test


    def allocate_corpus(self):
        corpus = self.corpus
        num_valid = int(math.floor(len(corpus) * ValidFraction))
        num_test = int(math.floor(len(corpus) * TestFraction))
        print(num_valid, num_test)
        test = corpus[: num_test]
        valid = corpus[num_test : num_test + num_valid]
        train = corpus[num_test + num_valid :]
        return train, valid, test


    def encode(self, corpus, rhythmic_unit = 1):
        dur_enc, mel_enc = [], []
        for score in corpus:
            for part in score.parts:
                measures = part.getElementsByClass("Measure")
                for m in measures:
                    dur_enc.append(self.enc_measure_durations(m))
                    mel_enc += self.enc_pitch_melodic(m)

        return torch.LongTensor(dur_enc), \
               torch.LongTensor(mel_enc), \
               torch.LongTensor(self.enc_pitch_harmonic(corpus))


    def enc_measure_durations(self, measure):
        # if necessary, add encoding to dict, then return the encoding
        notes = measure.getElementsByClass(["Note", "Rest", "Chord"])
        durations = [n.duration.quarterLength for n in notes]
        return self.duration_dictionary.add_word(tuple(durations))


    def enc_pitch_melodic(self, measure):
        notes = measure.getElementsByClass(["Note", "Rest", "Chord"])
        keysig = measure.keySignature
        if keysig is not None:
            trans_interval = (12 - keysig.asKey().getTonic().pitchClass) % 12
        else:
            trans_interval = 0
        measure_enc = []
        for n in notes:
            if type(n) == music21.note.Note or type(n) == music21.chord.Chord:
                pitch = self.enc_note_pitch(n.transpose(trans_interval))
            else:
                pitch = ()
            measure_enc.append(self.melodic_dictionary.add_word(pitch))
        return measure_enc


    def enc_pitch_harmonic(self, corpus):
        chord_seq = []
        for score in corpus:
            measures = score.chordify()

            # assume for simplicity that the key signature does not change
            m1 = measures[1]
            if m1.keySignature is not None:
                transpose_interval = 12 - m1.keySignature.asKey().getTonic().pitchClass
            else:
                transpose_interval = 0

            for chord in measures.recurse().getElementsByClass('Chord'):
                chord_pitches = tuple(sorted(list(set(
                    [(p.pitchClass + transpose_interval) % 12 for p in chord.pitches]
                ))))
                chord_seq.append(chord_pitches)

            chord_seq.append(())

        return [self.harmonic_dictionary.add_word(c) for c in chord_seq]


    def enc_note_pitch(self, n):
        note_type = type(n)
        if note_type == music21.chord.Chord:
            # return tuple([p.nameWithOctave for p in n.pitches])
            return tuple(n.pitchClasses)
        elif note_type == music21.note.Note:
            # return (n.pitch.nameWithOctave,)
            return (n.pitch.pitchClass,)
        else:
            return ()






