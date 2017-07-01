from itertools import *
from multiprocessing import Process, Queue
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
    def __init__(self, composer = None, meter = 4):
        self.duration_dictionary = Dictionary()
        self.melodic_dictionary = Dictionary()
        self.harmonic_dictionary = Dictionary()
        self.meter = meter

        if composer != None:
            train, valid, test = self.read_corpus(composer)
        else:
            train, valid, test = self.unpickle_corpus()

        self.duration_train, self.melodic_train, self.harmonic_train = self.encode(train)
        self.duration_valid, self.melodic_valid, self.harmonic_valid = self.encode(valid)
        self.duration_test, self.melodic_test, self.harmonic_test = self.encode(test)
        print(len(self.harmonic_dictionary), len(self.harmonic_train))


    def read_corpus(self,
            composer,
            max_size = 300,
            save = ('train-corpus.m21', 'valid-corpus.m21', 'test-corpus.m21')
    ):
        print('Reading corpus from file...')
        self.corpus = music21.corpus.getComposer(composer)
        # if len(self.corpus) > max_size:
        #     self.corpus = self.corpus[:max_size]
        if len(self.corpus) < 1:
            print('Error: cannot load any works by composer {}'.format(composer))
            exit(1)

        train, valid, test = self.allocate_corpus()
        train = self.parse_corpus(train, max_size)
        valid = self.parse_corpus(valid, max_size)
        test = self.parse_corpus(test, max_size)
        # with open(save[0], 'wb') as trainfile:
        #     pickle.dump(train, trainfile)
        # with open(save[1], 'wb') as validfile:
        #     pickle.dump(valid, validfile)
        # with open(save[2], 'wb') as testfile:
        #     pickle.dump(test, testfile)
        return train, valid, test


    def parse_corpus(self, files, max_size):
        corpus = []

        for f in files:
            s = music21.corpus.parse(f)
            if s.parts[0].measure(1).barDuration.quarterLength == self.meter:
                corpus.append(s)
        if len(corpus) <= max_size:
            return corpus
        else:
            return corpus[: max_size]


    def unpickle_corpus(self,
            load = ('train-corpus.m21', 'valid-corpus.m21', 'test-corpus.m21')
    ):
        with open(load[0], 'r') as trainfile:
            train = pickle.load(trainfile)
        with open(load[0], 'r') as validfile:
            valid = pickle.load(validfile)
        with open(load[0], 'r') as testfile:
            test = pickle.load(testfile)
        return train, valid, test


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
        test = corpus[: num_test]
        valid = corpus[num_test : num_test + num_valid]
        train = corpus[num_test + num_valid :]
        return train, valid, test


    def encode(self, corpus, rhythmic_unit = 1):
        print('Encoding rhythmic data into tensors...')
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
        measure_enc = []
        for n in notes:
            encoding = self.enc_note_pitch(n)
            measure_enc.append(self.melodic_dictionary.add_word(encoding))
        return measure_enc


    # need all parts at once, so this is a bit messier
    def enc_pitch_harmonic(self, corpus):
        # chord_enc = []
        # for score in corpus:
        #     measures = score.chordify()
        #     for chord in measures.recurse().getElementsByClass('Chord'):
        #         chord_enc += [p.nameWithOctave for p in chord.pitches] + ['<EOC>']
        # return [self.harmonic_dictionary.add_word(ch) for ch in chord_enc]
        h_pitches = []
        for score in corpus:
            measures = score.chordify()
            for chord in measures.recurse().getElementsByClass('Chord'):
                pitches = set([p.pitchClass for p in chord.pitches])
                for perm in permutations(pitches):
                    h_pitches += [p for p in perm] + ['<EOC>']
        return [self.harmonic_dictionary.add_word(c) for c in h_pitches]


    def enc_note_pitch(self, n):
        note_type = type(n)
        if note_type == music21.chord.Chord:
            return tuple([p.nameWithOctave for p in n.pitches])
            # return tuple(n.pitchClasses)
        elif note_type == music21.note.Note:
            return (n.pitch.nameWithOctave,)
        else:
            return ()




c = Corpus('bach')

