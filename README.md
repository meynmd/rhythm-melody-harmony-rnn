# rhythm-melody-harmony-rnn

This music generator uses RNNs to create separate duration, melodic and harmonic language models trained on a specified corpus.
The neural netword code is based on the PyTorch word language model (Penn Treebank) example code at 
https://github.com/pytorch/examples.git

Required dependencies:
  PyTorch (torch)
  Music21 (https://github.com/cuthbertLab/music21/releases)
  
You should also have a music editor installed so you can view the result when it is generated.

train.py
  Use train.py to train the model. Specify a directory containing files on which to train (in any format supported by Music21, such
  as .xml or .krn) or specify the composer to search in the Music21 corpus. 
  
  For example, if you have your corpus files in ./data:
    python train.py --corpus data
  Or to search for Bach in the Music21 corpus:
    python train.py --composer bach

  If you have a CUDA device, use the --cuda option.

generate.py
  To generate music from the trained model, just specify how many measures to generate, how many parts, and a random seed if you 
  want. For now, generate.py will just use the last saved model. For example, to generate 60 measures in four parts using CUDA,
    python generate.py --measures 60 --parts 4 --seed 3063 --cuda
