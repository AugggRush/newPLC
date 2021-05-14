import argparse
import json
import random
import os
import torch
import sys
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
sys.path.append("./")

from DNN_spec.DNNnet import DNNnet
from DNN_spec.linear2load import linear2load
import mel2samp as ms

def DNN_test(checkpoint_path, filename):
    model = DNNnet(**DNN_net_config)
    checkpoint_dict = torch.load(checkpoint_path, map_location='cpu')
    model_for_loading = checkpoint_dict['model']
    model.load_state_dict(model_for_loading.state_dict())
    print("Loaded checkpoint '{}' " .format(
        checkpoint_path))

    model.eval()

    Data_gen = linear2load(**DNN_data_config)

    audio, sr = ms.load_wav_to_torch(filename)
    # Take segment
    if audio.size(0) >= 2240:
        max_audio_start = audio.size(0) - 2240
        audio_start = random.randint(0, max_audio_start)
        audio = audio[audio_start:audio_start+2240]
    else:
        audio = torch.nn.functional.pad(audio, (2240 - audio.size(0), 0), 'constant').data


    linear = Data_gen.get_mel(audio)
    feed_linear = linear[:,:-2]
    targ_linear = linear[:,-2:]
    feed_linear = feed_linear.unsqueeze(0)
    gener_linear = model.forward(feed_linear)
    
    # gener_linear = gener_linear.squeeze().view(-1, 2).T
    gener_linear = gener_linear.detach().numpy()
    targ_linear = torch.cat((targ_linear[:,0], targ_linear[:,1]), 0)
    targ_linear = targ_linear.numpy()
    # targ_mel_db = librosa.power_to_db(targ_mel[0], ref=np.max)
    # gener_mel_db = librosa.power_to_db(gener_mel[0], ref=np.max)
    plt.figure()

    plt.plot(targ_linear)
    # librosa.display.specshow(targ_mel, x_axis='time', y_axis='mel')

    plt.plot(gener_linear[0], 'r')
    # librosa.display.specshow(gener_mel, x_axis='time', y_axis='mel')

    plt.show()

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='DNN_spec/config.json',
                        help='JSON file for configuration')

    args = parser.parse_args()

    # Parse configs.  Globals nicer in this case
    with open(args.config) as f:
        data = f.read()
    config = json.loads(data)
    global DNN_data_config
    DNN_data_config = config["DNN_data_config"]
    global DNN_net_config
    DNN_net_config = config["DNN_net_config"]
    ckp = "G:/程序/waveglow-modified/DNN_checkpoints/MEL/DNN_net_24"
    tap = "F:/DATA/LJSpeech-1/LJSpeech-1.0-16k/LJ050-0168.wav"
    DNN_test(ckp, tap)