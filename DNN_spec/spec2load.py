import os
import random
import argparse
import json
import torch
import torch.utils.data
import sys
from scipy.io.wavfile import read
from tqdm import tqdm
import mel2samp as ms 

class spec2load(torch.utils.data.Dataset):
    """
    This is the main class that calculates the spectrogram and returns the
    long spectrogram, short spectrogram pair.
    long spectrogram comprises of 11 frames, short spectrogram consists of 12th and 13th frames.
    """
    def __init__(self, training_files, num_frame, filter_length,
                hop_length, win_length, sampling_rate, mel_fmin, mel_fmax):
        self.audio_files = ms.files_to_list(training_files)
        self.all_length = torch.tensor(0, dtype=torch.long)
        self.hop_length = hop_length
        self.win_length = win_length
        random.seed(4321)
        random.shuffle(self.audio_files)
        self.stft = ms.TacotronSTFT(filter_length=filter_length,
                                 hop_length=self.hop_length,
                                 win_length=self.win_length,
                                 sampling_rate=sampling_rate,
                                 mel_fmin=mel_fmin, mel_fmax=mel_fmax)
        self.segment = num_frame * self.hop_length + self.win_length
        self.sampling_rate = sampling_rate

    def load_buffer(self):
        num_files = len(self.audio_files)
        for i in tqdm(range(num_files)):
            filename = self.audio_files[i]
            audio, sampling_rate = ms.load_wav_to_torch(filename)
            if sampling_rate != self.sampling_rate:
                raise ValueError("{} SR doesn't match target {} SR".format(
                    sampling_rate, self.sampling_rate))
            self.all_length += audio.size(0)
            if i == 0:
                self.all_audio = audio
            else:
                self.all_audio = torch.cat((self.all_audio, audio), 0)
        print("All audio has been loaded, totally length: {}".format(self.all_length))
    
    def get_spec(self, audio_data):
        audio_norm = audio_data / ms.MAX_WAV_VALUE
        audio_norm = audio_norm.unsqueeze(0)
        audio_norm = torch.autograd.Variable(audio_norm, requires_grad=False)
        spec = self.stft.mel_spectrogram(audio_norm)
        spec = torch.squeeze(spec, 0)
        return spec
    
    def __getitem__(self, index):

        # Take segment
        start_index = index * self.hop_length

        audio_segment = self.all_audio[start_index:start_index+self.segment]

        spec = self.get_spec(audio_segment)
        feed_spec = spec[:,:-2]
        targ_spec = spec[:,-2:]
        
        return (feed_spec, targ_spec)

    def __len__(self):

        num_segments = torch.floor((self.all_length - self.segment) / self.hop_length)
        # print("There is totally {} segments in the dataset".format(num_segments))
        return num_segments.long()