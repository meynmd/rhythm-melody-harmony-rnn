"""
training script for duration-melodic-harmonic music language model
based on main.py from PyTorch word language model (Penn Treebank) example code
https://github.com/pytorch/examples.git
"""

import argparse
import time
import math
import torch
import torch.nn as nn
from torch.autograd import Variable

import data
import model

parser = argparse.ArgumentParser(
    description='Music Language Model, based on PyTorch PennTreeBank RNN/LSTM Language Model'
)
parser.add_argument('--data', type=str, default='./data',
                    help='location of the data corpus')
parser.add_argument('--model', type=str, default='LSTM',
                    help='type of recurrent net (RNN_TANH, RNN_RELU, LSTM, GRU)')
parser.add_argument('--emsize', type=int, default=200,
                    help='size of word embeddings')
parser.add_argument('--nhid', type=int, default=200,
                    help='number of hidden units per layer')
parser.add_argument('--nlayers', type=int, default=2,
                    help='number of layers')
parser.add_argument('--lr', type=float, default=20,
                    help='initial learning rate')
parser.add_argument('--clip', type=float, default=0.25,
                    help='gradient clipping')
parser.add_argument('--epochs', type=int, default=40,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=20, metavar='N',
                    help='batch size')
parser.add_argument('--bptt', type=int, default=35,
                    help='sequence length')
parser.add_argument('--dropout', type=float, default=0.2,
                    help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--tied', action='store_true',
                    help='tie the word embedding and softmax weights')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--log-interval', type=int, default=200, metavar='N',
                    help='report interval')
parser.add_argument('--dursave', type=str,  default='dur-model.pt',
                    help='path to save the final rhythm model')
parser.add_argument('--melsave', type=str,  default='mel-model.pt',
                    help='path to save the final melodic pitch model')
parser.add_argument('--harsave', type=str,  default='har-model.pt',
                    help='path to save the final harmonic pitch model')
parser.add_argument('--composer', type=str,  default=None,
                    help='composer to train on')
parser.add_argument('--corpus', type=str,  default=None,
                    help='directory containing files to train on')
args = parser.parse_args()


eval_batch_size = 10
criterion = nn.CrossEntropyLoss()
lr = args.lr


def batchify(data, bsz):
    # Work out how cleanly we can divide the dataset into bsz parts.
    nbatch = data.size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    data = data.view(bsz, -1).t().contiguous()
    if args.cuda:
        data = data.cuda()
    return data


def repackage_hidden(h):
    """Wraps hidden states in new Variables, to detach them from their history."""
    if type(h) == Variable:
        return Variable(h.data)
    else:
        return tuple(repackage_hidden(v) for v in h)


def get_batch(source, i, evaluation=False):
    seq_len = min(args.bptt, len(source) - 1 - i)
    data = Variable(source[i:i+seq_len], volatile=evaluation)
    target = Variable(source[i+1:i+1+seq_len].view(-1))
    return data, target


def evaluate(data_source, data_dict, model_eval):
    # Turn on evaluation mode which disables dropout.
    model_eval.eval()
    total_loss = 0
    ntokens = len(data_dict)
    hidden = model_eval.init_hidden(eval_batch_size)
    for i in range(0, data_source.size(0) - 1, args.bptt):
        data, targets = get_batch(data_source, i, evaluation=True)
        output, hidden = model_eval(data, hidden)
        output_flat = output.view(-1, ntokens)
        total_loss += len(data) * criterion(output_flat, targets).data
        hidden = repackage_hidden(hidden)
    return total_loss[0] / len(data_source)


def train(data_train, data_dict, model_train, epoch):
    # Turn on training mode which enables dropout.
    model_train.train()
    total_loss = 0
    start_time = time.time()
    ntokens = len(data_dict)
    hidden = model_train.init_hidden(args.batch_size)
    for batch, i in enumerate(range(0, data_train.size(0) - 1, args.bptt)):
        data, targets = get_batch(data_train, i)
        # Starting each batch, we detach the hidden state from how it was previously produced.
        # If we didn't, the model_train would try backpropagating all the way to start of the dataset.
        hidden = repackage_hidden(hidden)
        model_train.zero_grad()
        output, hidden = model_train(data, hidden)
        loss = criterion(output.view(-1, ntokens), targets)
        loss.backward()

        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        torch.nn.utils.clip_grad_norm(model_train.parameters(), args.clip)
        for p in model_train.parameters():
            p.data.add_(-lr, p.grad.data)

        total_loss += loss.data

        if batch % args.log_interval == 0 and batch > 0:
            cur_loss = total_loss[0] / args.log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.2f} | ms/batch {:5.2f} | '
                    'loss {:5.2f} | ppl {:8.2f}'.format(
                epoch, batch, len(data_train) // args.bptt, lr,
                elapsed * 1000 / args.log_interval, cur_loss, math.exp(cur_loss)))
            total_loss = 0
            start_time = time.time()


def run_training(train_data, val_data, test_data, model_train, data_dict, lr, save, epochs):
    best_val_loss = None
    try:
        for epoch in range(1, epochs + 1):
            epoch_start_time = time.time()
            train(train_data, data_dict, model_train, epoch)
            val_loss = evaluate(val_data, data_dict, model_train)
            print('-' * 89)
            print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                  'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                             val_loss, math.exp(val_loss)))
            print('-' * 89)
            # Save the model if the validation loss is the best we've seen so far.
            if not best_val_loss or val_loss < best_val_loss:
                with open(save, 'wb') as f:
                    torch.save(model_train, f)
                best_val_loss = val_loss
            else:
                # Anneal the learning rate if no improvement has been seen in the validation dataset.
                lr /= 4.0
    except KeyboardInterrupt:
        print('-' * 89)
        print('Exiting from training early')


def test(test_data, data_dict, test_model):
    test_loss = evaluate(test_data, data_dict, test_model)
    print('=' * 89)
    print('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
        test_loss, math.exp(test_loss)))
    print('=' * 89)


###############################################################################
#                                                                             #
# the training script                                                         #
#                                                                             #
###############################################################################

# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    else:
        torch.cuda.manual_seed(args.seed)

# load the data
if args.corpus is not None:
    corpus = data.Corpus(corpus_dir=args.corpus)
elif args.composer is not None:
    corpus = data.Corpus(composer=args.composer)
else:
    corpus = data.Corpus()

###############################################################################
# Rhythm model
###############################################################################

dur_train_data = batchify(corpus.duration_train, args.batch_size)
dur_val_data = batchify(corpus.duration_valid, eval_batch_size)
dur_test_data = batchify(corpus.duration_test, eval_batch_size)

# Build the model
n_dur_tokens = len(corpus.duration_dictionary)
rhythm_model = model.RNNModel(
    args.model, n_dur_tokens, args.emsize, args.nhid, args.nlayers, args.dropout, args.tied)
if args.cuda:
    rhythm_model.cuda()

# Train
run_training(dur_train_data, dur_val_data, dur_test_data,
             rhythm_model, corpus.duration_dictionary, lr,
             args.dursave, 40)

# Load the best saved model.
with open(args.dursave, 'rb') as f:
    rhythm_model = torch.load(f)

test(dur_test_data, corpus.duration_dictionary, rhythm_model)

###############################################################################
# melodic model
###############################################################################

lr = args.lr
mel_train_data = batchify(corpus.melodic_train, args.batch_size)
mel_val_data = batchify(corpus.melodic_valid, eval_batch_size)
mel_test_data = batchify(corpus.melodic_test, eval_batch_size)

# Build the model
n_mel_tokens = len(corpus.melodic_dictionary)
melodic_model = model.RNNModel(
    args.model, n_mel_tokens, args.emsize, args.nhid, args.nlayers, args.dropout, args.tied)
if args.cuda:
    melodic_model.cuda()

# Train
run_training(mel_train_data, mel_val_data, mel_test_data,
             melodic_model, corpus.melodic_dictionary, lr,
             args.melsave, args.epochs)

# Load the best saved model.
with open(args.melsave, 'rb') as f:
    melodic_model = torch.load(f)

test(mel_test_data, corpus.melodic_dictionary, melodic_model)

###############################################################################
# harmonic model
###############################################################################

lr = args.lr
har_train_data = batchify(corpus.harmonic_train, args.batch_size)
har_val_data = batchify(corpus.harmonic_valid, eval_batch_size)
har_test_data = batchify(corpus.harmonic_test, eval_batch_size)

# Build the model
n_h_tokens = len(corpus.harmonic_dictionary)
harmonic_model = model.RNNModel(
    args.model, n_h_tokens, args.emsize, args.nhid, args.nlayers, args.dropout, args.tied)
if args.cuda:
    harmonic_model.cuda()

# Train
run_training(har_train_data, har_val_data, har_test_data,
             harmonic_model, corpus.harmonic_dictionary, lr,
             args.harsave, args.epochs)

# Load the best saved model.
with open(args.harsave, 'rb') as f:
    harmonic_model = torch.load(f)

test(har_test_data, corpus.harmonic_dictionary, harmonic_model)
