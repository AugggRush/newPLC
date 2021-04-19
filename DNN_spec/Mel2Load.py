import os
import random
import argparse
import json
import torch
import torch.utils.data
from librosa.filters import mel as librosa_mel_fn
import sys
from scipy.io.wavfile import read

import mel2samp as ms 

class Mel2Load(torch.utils.data.Dataset):
    """
    This is the main class that calculates the spectrogram and returns the
    long spectrogram, short spectrogram pair.
    long spectrogram comprises of 11 frames, short spectrogram consists of 12th and 13th frames.
    """
    def __init__(self, training_files, num_frame, filter_length,
                hop_length, win_length, sampling_rate, mel_fmin, mel_fmax):
        self.audio_files = ms.files_to_list(training_files)
        random.seed(4321)
        random.shuffle(self.audio_files)
        self.stft = ms.TacotronSTFT(filter_length=filter_length,
                                 hop_length=hop_length,
                                 win_length=win_length,
                                 sampling_rate=sampling_rate,
                                 mel_fmin=mel_fmin, mel_fmax=mel_fmax)
        self.segment = num_frame * hop_length + win_length
        self.sampling_rate = sampling_rate

    def get_mel(self, audio):
        audio_norm = audio / ms.MAX_WAV_VALUE
        audio_norm = audio_norm.unsqueeze(0)
        audio_norm = torch.autograd.Variable(audio_norm, requires_grad=False)
        melspec = self.stft.mel_spectrogram(audio_norm)
        melspec = torch.squeeze(melspec, 0)
        return melspec
    
    def __getitem__(self, index):
        # Read audio
        filename = self.audio_files[index]
        audio, sampling_rate = ms.load_wav_to_torch(filename)
        if sampling_rate != self.sampling_rate:
            raise ValueError("{} SR doesn't match target {} SR".format(
                sampling_rate, self.sampling_rate))

        # Take segment
        if audio.size(0) >= self.segment:
            max_audio_start = audio.size(0) - self.segment
            audio_start = random.randint(0, max_audio_start)
            audio = audio[audio_start:audio_start+self.segment]
        else:
            audio = torch.nn.functional.pad(audio, (self.segment - audio.size(0), 0), 'constant').data


        mel = self.get_mel(audio)
        feed_mel = mel[:,:-2]
        targ_mel = mel[:,-2:]
        
        return (feed_mel, targ_mel)

    def __len__(self):
        return len(self.audio_files)