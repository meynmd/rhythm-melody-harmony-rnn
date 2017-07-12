###############################################################################
# Music Language Modeling
#
# based on Language Modeling on Penn Tree Bank
#
# This file generates new sentences sampled from the language model
#
###############################################################################

import argparse
import sys
import math
import random
from itertools import *
import numpy
from scipy import stats
import torch
from torch.autograd import Variable
import music21
import data

parser = argparse.ArgumentParser(description='PyTorch PTB Language Model')

# Model parameters.
parser.add_argument('--data', type=str, default='./data',
                    help='location of the data corpus')
parser.add_argument('--duration', type=str, default='./dur-model.pt',
                    help='duration model checkpoint to use')
parser.add_argument('--melodic', type=str, default='./mel-model.pt',
                    help='melodic pitch model checkpoint to use')
parser.add_argument('--harmonic', type=str, default='./har-model.pt',
                    help='harmonic pitch model checkpoint to use')
parser.add_argument('--outf', type=str, default='',
                    help='output file for generated text')
parser.add_argument('--measures', type=int, default='100',
                    help='number of measures to generate')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--temperature', type=float, default=1.0,
                    help='temperature - higher will increase diversity')
parser.add_argument('--log-interval', type=int, default=100,
                    help='reporting interval')
parser.add_argument('--meter', type=int, default=4,
                    help='desired tactus beats per measure')
parser.add_argument('--parts', type=int, default=2,
                    help='number of parts to generate')
args = parser.parse_args()

# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    else:
        torch.cuda.manual_seed(args.seed)

if args.temperature < 1e-3:
    parser.error("--temperature has to be greater or equal 1e-3")

# load the models
with open(args.duration, 'rb') as f:
    dur_model = torch.load(f)
dur_model.eval()
with open(args.melodic, 'rb') as f:
    mel_model = torch.load(f)
mel_model.eval()
with open(args.harmonic, 'rb') as f:
    har_model = torch.load(f)
har_model.eval()

if args.cuda:
    dur_model.cuda()
    mel_model.cuda()
    har_model.cuda()
else:
    dur_model.cpu()
    mel_model.cpu()
    har_model.cpu()

# load the last saved corpus and model
# TODO: make corpus constructor take desired meter
corpus = data.Corpus()
ntokens = len(corpus.duration_dictionary)
hidden = dur_model.init_hidden(1)
input = Variable(torch.rand(1, 1).mul(ntokens).long(), volatile=True)
if args.cuda:
    input.data = input.data.cuda()

# first, generate duration output
measures_dur = []
beats_measure = args.meter
nparts = args.parts
nmeasures = args.measures
for i in range(nmeasures):
    measure = []
    for j in range(nparts):
        output, hidden = dur_model(input, hidden)
        word_weights = output.squeeze().data.div(args.temperature).exp().cpu()
        word_idx = torch.multinomial(word_weights, 1)[0]
        input.data.fill_(word_idx)
        word = corpus.duration_dictionary.idx2word[word_idx]
        if sum(word) > beats_measure:
            word = [d for d in word]
            while len(word) > beats_measure:
                word = word[:-1]
            word = tuple(word)
        if sum(word) < beats_measure:
            print('warning: too few beats in measure')
            word = tuple([d for d in word] + [beats_measure - sum(word)])

        measure.append(word)
    measures_dur.append(measure)

    if i % args.log_interval == 0:
        print('| Duration Output: Generated {}/{} measures'.format(i, args.measures))

# start constructing the output score
score = music21.stream.Score()
for j in range(nparts):
    score.append(music21.stream.Part())
for j, md in enumerate(measures_dur):
    for i, part in enumerate(md):
        out_measure = music21.stream.Measure()
        out_measure.number = j
        for dur in part:
            # place-holder notes
            note = music21.note.Note()
            note.duration.quarterLength = dur
            note.priority = i   # I'll use Note.priority to indicate which part
            out_measure.append(note)
        score.parts.elements[i].append(out_measure)


def get_chord_weights(hidden_state, prev_pitch_classes):
    prev_pcs_set = tuple(sorted(list(set(prev_pitch_classes))))
    input_tensor = torch.LongTensor(1, 1)
    if prev_pcs_set in corpus.harmonic_dictionary.word2idx.keys():
        input_tensor[0] = corpus.harmonic_dictionary.word2idx[prev_pcs_set]
    else:
        input_tensor[0] = random.randint(len(corpus.harmonic_dictionary))

    input = Variable(input_tensor, volatile=True)
    if args.cuda:
        input.data = input.data.cuda()

    output, hidden_state = har_model(input, hidden_state)
    weights = output.squeeze().data.div(args.temperature).cpu()   #.exp().cpu()
    return weights, hidden_state


def get_melodic_weights(hidden_state, prev_pitch_class):
    input_tensor = torch.LongTensor(1, 1)
    input_tensor[0] = corpus.melodic_dictionary.word2idx[(prev_pitch_class,)]
    input = Variable(input_tensor, volatile=True)
    if args.cuda:
        input.data = input.data.cuda()

    output, hidden_state = mel_model(input, hidden_state)
    weights = output.squeeze().data.div(args.temperature)   #.exp().cpu()

    return weights, hidden_state


# generate the pitches
h_hidden = har_model.init_hidden(1)
m_hidden = [mel_model.init_hidden(1) for i in range(nparts)]
pitchset_random = []
while len(pitchset_random) < 3:
    pitchset_random = corpus.harmonic_dictionary.idx2word[random.randint(
        0, len(corpus.harmonic_dictionary.idx2word))]
current_pitch_classes = [random.choice(pitchset_random) for i in range(nparts)]
current_notes = [None for i in range(nparts)]
last_chord = None
chosen_chord = None
chord_weights = None
melodic_weights = [[] for i in range(nparts)]
overlaps = score.getOverlaps()

if nparts > 4:
    octaves = [random.randint(2, 5) for i in range(max(len(v) for v in overlaps.values()))]
    octaves[0] = 5
else:
    octaves = [5,4,4,3]

for i, (offset, notes) in enumerate(sorted(overlaps.items())):
    # generate a distribution for the melodic notes of each part
    for j, note in enumerate(notes):
        part_idx = note.priority
        if current_pitch_classes[part_idx] is not None:
            melodic_weights[part_idx], m_hidden[part_idx] = get_melodic_weights(
                m_hidden[part_idx], current_pitch_classes[part_idx]
            )
        else:
            melodic_weights[part_idx] = torch.Tensor(
                [-1. for i in range(len(corpus.melodic_dictionary))]
            )

        mw = torch.exp(melodic_weights[part_idx])
        sum_weights = mw.sum()
        melodic_weights[part_idx] = torch.div(mw, sum_weights)

    # generate a distribution for the next chord
    if last_chord is None:
        chord_idx = random.randint(0, len(corpus.harmonic_dictionary.idx2word))
        chosen_chord = corpus.harmonic_dictionary.idx2word[chord_idx]
        last_chord = chosen_chord
    else:
        chord_weights, new_hidden = get_chord_weights(h_hidden, last_chord)
        h_hidden = new_hidden

        chord_weights = torch.exp(chord_weights)
        sum_weights = chord_weights.sum()
        chord_weights = torch.div(chord_weights, sum_weights)

        # choose the next chord
        chosen_chord_idx = torch.multinomial(chord_weights, 1)[0]
        chosen_chord = corpus.harmonic_dictionary.idx2word[chosen_chord_idx]
        last_chord = chosen_chord
        print(chosen_chord)

    chord_tone_idxs = [corpus.melodic_dictionary.word2idx[(pc,)] for pc in chosen_chord]
    for note in notes:
        part_idx = note.priority
        for j, w in enumerate(melodic_weights[part_idx]):
            if j not in chord_tone_idxs:
                melodic_weights[part_idx][j] = 0.

        # renormalize
        sum_weights = melodic_weights[part_idx].sum()
        if sum_weights == 0.:
            print('zero')
            continue
        melodic_weights[part_idx] = torch.div(melodic_weights[part_idx], sum_weights)

        if type(note) == music21.note.Note:
            # figure out what is the best chord pitch to assign this note
            part_idx = note.priority
            chosen_pitch_idx = torch.multinomial(melodic_weights[part_idx], 1)[0]
            chosen_pitch = corpus.melodic_dictionary.idx2word[chosen_pitch_idx]
            if len(chosen_pitch) == 1:
                note.pitch.pitchClass = chosen_pitch[0]
                # determine the octave
                note.octave = octaves[part_idx]
                if current_notes[part_idx] is not None:
                    interval = music21.interval.notesToInterval(current_notes[part_idx], note).cents / 100
                else:
                    interval = 0

                while abs(interval) > 9:
                    if interval > 9:
                        if note.octave > 1:
                            note.octave -= 1
                        else:
                            break
                    elif interval < -9:
                        if note.octave < 5:
                            note.octave += 1
                        else:
                            break
                    interval = music21.interval.notesToInterval(current_notes[part_idx], note).cents / 100
                octaves[part_idx] = note.octave
                current_notes[part_idx] = note
                current_pitch_classes[part_idx] = note.pitch.pitchClass

            elif len(chosen_pitch) == 0:
                note = music21.note.Rest()
            else:
                note = music21.chord.Chord(list(chosen_pitch))

        elif type(note) == music21.chord.Chord:
            chord_pitches = [corpus.melodic_dictionary.idx2word[idx] for idx in note_idxs]
            if len(note) > 0:
                for n in note:
                    n.pitch.pitchClass = random.choice(chord_pitches)
                    n.pitch.pitchClass = octaves[part_idx]




score.show()



