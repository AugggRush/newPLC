import os
import torch
import torch.utils.data
import random
import argparse
import json
from librosa.filters import mel as librosa_mel_fn
import sys
from scipy.io.wavfile import read

from mel2samp import TacotronSTFT, files_to_list, load_wav_to_torch, MAX_WAV_VALUE

# this file is going to reset the Dataset class 
# for new training target. Given previous five frames audio and its mel spectrum, 
# the model is going to Map the samples of the preceding two frames, the current 
# frame and the next two frames to the simple Gaussian distribution.
# ===============================================================================================
# ===============================================================================================
# input audios  : |--first frame--|--second frame--|-- ... --|--fourth frame--|--fifth frame--|...
# condition mel : |--first frame--|--second frame--|-- ... --|--fourth frame--|--fifth frame--|...

# output audios : |--third frame--|--fourth frame--|-- ... --|--sixth frame--|--seventh frame--|
# ===============================================================================================
# ===============================================================================================
# that means Dataset(condition mel, output audios)

class Mel2Samp_dislocation(torch.utils.data.Dataset):
    """
    This is the main class that calculates the spectrogram and returns the
    spectrogram, audio pair.
    """
    def __init__(self, training_files, segment_length, filter_length,
                 hop_length, win_length, sampling_rate, mel_fmin, mel_fmax):
        self.audio_files = files_to_list(training_files)
        random.seed(1234)
        random.shuffle(self.audio_files)
        self.stft = TacotronSTFT(filter_length=filter_length,
                                 hop_length=hop_length,
                                 win_length=win_length,
                                 sampling_rate=sampling_rate,
                                 mel_fmin=mel_fmin, mel_fmax=mel_fmax)
        self.postpone_length = 2 * filter_length
        self.segment_length = segment_length
        self.sampling_rate = sampling_rate

    def get_mel(self, audio):
        audio_norm = audio / MAX_WAV_VALUE
        audio_norm = audio_norm.unsqueeze(0)
        audio_norm = torch.autograd.Variable(audio_norm, requires_grad=False)
        melspec = self.stft.mel_spectrogram(audio_norm)
        melspec = torch.squeeze(melspec, 0)
        return melspec

    def __getitem__(self, index):
        # Read audio
        filename = self.audio_files[index]
        audio, sampling_rate = load_wav_to_torch(filename)
        if sampling_rate != self.sampling_rate:
            raise ValueError("{} SR doesn't match target {} SR".format(
                sampling_rate, self.sampling_rate))

        # Take mel segment and audio segemnt
        if audio.size(0) >= self.segment_length+self.postpone_length:
            max_audio_start = audio.size(0) - self.segment_length - self.postpone_length
            audio_start = random.randint(0, max_audio_start)
            audio4mel = audio[audio_start:audio_start+self.segment_length]
            audio4targ = audio[audio_start+self.postpone_length:audio_start+self.segment_length+self.postpone_length]
        else:
            audio = torch.nn.functional.pad(audio, (0, self.segment_length + self.postpone_length - audio.size(0)), 'constant').data
            audio4mel = audio[0:self.segment_length]
            audio4targ = audio[self.postpone_length:self.segment_length+self.postpone_length]

        mel = self.get_mel(audio4mel)
        audio4targ = audio4targ / MAX_WAV_VALUE

        return (mel, audio4targ)

    def __len__(self):
        return len(self.audio_files)