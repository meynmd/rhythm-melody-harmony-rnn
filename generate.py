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

# TODO: make corpus constructor take desired meter
corpus = data.Corpus('beethoven')
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
        if sum(word) != beats_measure:
            word = tuple([d for d in word] + [beats_measure - sum(word)])
        measure.append(word)
    measures_dur.append(measure)
    print (measure)

    if i % args.log_interval == 0:
        print('| Duration Output: Generated {}/{} measures'.format(i, args.measures))

# generate one line of melody to start with
ntokens = len(corpus.melodic_dictionary)
hidden = mel_model.init_hidden(1)
input = Variable(torch.rand(1, 1).mul(ntokens).long(), volatile=True)
if args.cuda:
    input.data = input.data.cuda()

npitches = sum([sum([len(p) for p in m]) for m in measures_dur])
melodic_pitches = []
for i in range(npitches):
    output, hidden = mel_model(input, hidden)
    word_weights = output.squeeze().data.div(args.temperature).exp().cpu()
    word_idx = torch.multinomial(word_weights, 1)[0]
    input.data.fill_(word_idx)
    word = corpus.melodic_dictionary.idx2word[word_idx]
    melodic_pitches.append(word)

# start constructing the output score
score = music21.stream.Stream()
topline = music21.stream.Stream()
out_parts = [music21.stream.Part() for j in range(nparts)]
for md in measures_dur:
    for i, part in enumerate(md):
        out_measure = music21.stream.Measure()
        for dur in part:
            if i == 0:
                next_pitch = melodic_pitches.pop(0)
                if len(next_pitch) == 1:
                    note = music21.note.Note(next_pitch[0])
                elif len(next_pitch) == 0:
                    note = music21.note.Rest()
                else:
                    note = music21.chord.Chord(list(next_pitch))
            else:
                note = music21.note.Note()

            note.duration.quarterLength = dur
            out_measure.append(note)
        out_parts[i].append(out_measure)

for p in out_parts:
     score.append(p)

if nparts < 2:
    score.show()
    exit(0)


def find_overlapping(measures, this_note):
    this_note_offset = this_note.offset
    overlapping = []
    for m in measures:
        # preceding = takewhile(
        #     lambda x: x.offset <= this_note.offset,
        #     [n for n in m.getElementsByClass(["Note", "Rest", "Chord"])]
        # )
        over_note = None
        for n in m.getElementsByClass(["Note", "Rest", "Chord"]):
            if n.offset <= this_note.offset:
                over_note = n
            else:
                break
        if type(over_note) == music21.note.Note:
            overlapping.append(over_note.pitch.pitchClass)
        elif type(over_note) == music21.chord.Chord:
            overlapping += [p.pitchClass for p in over_note.pitches]
    return overlapping


def generate_chord(existing_pitches, num_to_gen):
    ntokens = len(corpus.harmonic_dictionary)
    out_chords = []
    hidden = har_model.init_hidden(1)
    for pitch in existing_pitches:
        p_idx = corpus.harmonic_dictionary.word2idx[pitch]
        # input = Variable(torch.LongTensor([p_idx]), volatile=True)
        # input = Variable(torch.rand(1, 1).mul(ntokens).long(), volatile=True)
        start_note = torch.LongTensor(1, 1)
        start_note[0] = p_idx
        input = Variable(start_note, volatile=True)
        print (input)
        if args.cuda:
            input.data = input.data.cuda()

        chord_pitches = []
        for i in range(num_to_gen):
            output, hidden = har_model(input, hidden)
            word_weights = output.squeeze().data.div(args.temperature).exp().cpu()
            word_idx = torch.multinomial(word_weights, 1)[0]
            input.data.fill_(word_idx)
            word = corpus.harmonic_dictionary.idx2word[word_idx]
            if word == '<EOC>':
                break
            out_chords += [word_idx]

        # if len(chord_pitches) > 0:
        #     out_chords.append(chord_pitches)

    return out_chords


# fill in the other parts
current_octave = [random.randint(1, 3) for i in range(nparts)]
last_notes = [None for i in range(nparts)]
for j in range(nmeasures):
    for i in range(1, nparts):
        for k, note in enumerate(score[i][j]):
            over_pitches = find_overlapping([score[m][j] for m in range(i)], note)
            # run the model to find candidate pitches
            candidate_chords = generate_chord(over_pitches, 4)
            print('note: {}\n\tcandidates: {}'.format(note, candidate_chords))
            if len(candidate_chords) > 0:
                chosen_pitch = stats.mode(numpy.array(candidate_chords))[0][0]
                note.pitch.pitchClass = corpus.harmonic_dictionary.idx2word[chosen_pitch]
                if last_notes[i] != None:
                    if note.pitch.pitchClass - last_notes[i] > 9:
                        current_octave[i] = max(current_octave[i] - 1, 1)
                    elif note.pitch.pitchClass - last_notes[i] < 9:
                        current_octave[i] = min(current_octave[i] + 1, 5)
                note.pitch.octave = current_octave[i]
                last_notes[i] = note.pitch.pitchClass
            else:
                r = music21.note.Rest()
                r.duration.quarterLength = note.duration.quarterLength
                score[i][j].replace(note, r)

score.show()




