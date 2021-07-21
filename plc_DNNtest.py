import os
from scipy.io.wavfile import write
import torch
import json
import numpy as np
import matplotlib.pyplot as plt
import librosa
import soundfile
from mel2samp import files_to_list, MAX_WAV_VALUE
from denoiser import Denoiser
import mel2samp as ms
from DNN_spec.DNNnet import DNNnet
from DNN_spec.spec2load import spec2load
from Packlossfunction import frame2wav, win_generate
# This file is going to recover the received packet(one frame) lossed audio.
# Once the losed packet(one frame) was detected, the nueral net work is going to generate 
# three previous frames, current losed frame and one frame futher, totally five frame raw 
# audio.
# Then the WSOLA method was used to construct fluent audio
# If continuous packets loss, process recursively.

def pl_detect(state_file):
    """
    Assume there is a log.txt recording the audio packets receiving state
    at the decoder.
    Frame received marked as 1, and frame lost marked as 0.
    And the log.txt was named after the audio file name.
    """
    with open(state_file, encoding='utf-8') as f:
        pl_state = f.readlines()
        state = []
        for bool_state in pl_state:
            state.append(int(bool_state))
        print("packet loss state for audio {}:".format(state_file))
        print(state)
        return state

def load_frames_to_torch(frame_filename):
    numpy_frames = np.load(frame_filename)
    print("Successfully load packet lossed frames from path: {}".format(frame_filename))
    return torch.from_numpy(numpy_frames).float()

def extra_prevAudio(pl_frames, pl_index, win_length, hop_length):
    """
    For simulation, assuming decoded audio is received, the audio feeding to the neural
    plc system is clipped from pl_audio while packet loss detected(pl_state[pl_index] = 1).
    """
    if pl_index == 0:
        feed_frames = torch.zeros((11, win_length))
    else:
        if pl_index >= 11:
            feed_frames = pl_frames[pl_index-11:pl_index]
        else: 
            pad_zeros = torch.zeros((11-pl_index, win_length))
            feed_frames = torch.cat((pad_zeros, pl_frames[:pl_index]), dim=0)
 
    feed_audio = frame2wav(feed_frames.numpy(), hop_length)

    return torch.from_numpy(feed_audio).float()

def extra_truespec(true_frames, pl_index, win_length, hop_length):

    if pl_index == 0:
        feed_frames = torch.zeros((13, win_length))
    else:
        if pl_index >= 11:
            feed_frames = true_frames[pl_index-11:pl_index+2]
        else: 
            pad_zeros = torch.zeros((11-pl_index, win_length))
            feed_frames = torch.cat((pad_zeros, true_frames[:pl_index+2]), dim=0)
 
    feed_audio = frame2wav(feed_frames.numpy(), hop_length)
    true_audio = torch.from_numpy(feed_audio).float()
    Data_gen = spec2load(**DNN_data_config)
    spec = Data_gen.get_spec(true_audio)
    true_spec = spec.unsqueeze(0)

    return true_spec

def DNN_stretch(feed_audio, DNN_path):
    assert os.path.isfile(DNN_path)
    checkpoint_dict = torch.load(DNN_path, map_location='cpu')
    model = DNNnet(**DNN_net_config)
    iteration = checkpoint_dict['iteration']
    model_for_loading = checkpoint_dict['model']
    model.load_state_dict(model_for_loading.state_dict())
    print("Loaded checkpoint '{}' (iteration {})" .format(
           DNN_path, iteration))

    model.eval()

    Data_gen = spec2load(**DNN_data_config)

    spec = Data_gen.get_spec(feed_audio)
    feed_spec = spec.unsqueeze(0)
    gener_spec = model.forward(feed_spec)

    out_mel = torch.stack([gener_spec[:, :spec.size(0)], \
        gener_spec[:, spec.size(0):]], dim=-1)

    com_mel = torch.cat((feed_spec, out_mel), dim=2)

    return com_mel

def inference_plc(mel, waveglow_path, sigma, sampling_rate, is_fp16,
         denoiser_strength):

    waveglow = torch.load(waveglow_path)['model']
    waveglow = waveglow.remove_weightnorm(waveglow)
    waveglow.cuda().eval()
    if is_fp16:
        from apex import amp
        waveglow, _ = amp.initialize(waveglow, [], opt_level="O3")
   
    if denoiser_strength > 0:
        denoiser = Denoiser(waveglow).cuda()
    mel = torch.autograd.Variable(mel.cuda())
    mel = mel.half() if is_fp16 else mel
    with torch.no_grad():
        audio = waveglow.infer(mel, sigma=sigma)
        if denoiser_strength > 0:
            audio = denoiser(audio, denoiser_strength)
        audio = audio * MAX_WAV_VALUE
    audio = audio.squeeze()

    return audio

def audio_mend(audio_file, win_length, hop_length, \
        DNN_path, waveglow_path, sigma, sampling_rate, is_fp16,
        denoiser_strength, comparation):

    state_file = audio_file[:-3]+'txt'
    pl_state = pl_detect(state_file)
    pl_frames = load_frames_to_torch(audio_file)

    window = torch.from_numpy(win_generate(win_length)).float()
    gain_ = window * window
    win_adjust = torch.cat([gain_[:hop_length]+gain_[hop_length:], gain_[hop_length:]+gain_[:hop_length]], dim=-1)
    win_adjust = np.sqrt(win_length*win_adjust)
    gain_factor = 1
    for index, state in enumerate(pl_state):
        
        if state == 1:
            gain_factor = 1
            print("Number {} frame is not loss, skip to next".format(index))
            continue
        else:
            print("Number {} frame is loss, compensation start:".format(index))
            prev_audio = extra_prevAudio(pl_frames, index, win_length, hop_length)
            #Mel comparation part
            true_Mel = extra_truespec(comparation, index, win_length, hop_length)
            mel = DNN_stretch(prev_audio, DNN_path)
            true_Mel2plot = true_Mel.squeeze(0).transpose(0, 1)
            mel2plot = mel.squeeze(0).transpose(0, 1).detach()
            plt.figure('mel compare '+str(index))
            plt.plot(true_Mel2plot[-2:].flatten(), 'b')
            plt.plot(mel2plot[-2:].flatten(), 'y')
            
            ##############################################
            glowaudio = inference_plc(mel, waveglow_path, sigma, sampling_rate, is_fp16, denoiser_strength)
            #power_factor = power_adjust(prev_audio, glowaudio[:-(win_length)])

            likely_probe = prev_audio[-hop_length:]
            audio_patch = glowaudio[-(2*hop_length+win_length):]# * power_factor
            audio_patch = audio_patch.cpu()
            max_cor = -ms.MAX_WAV_VALUE
            max_i = hop_length
            for i in range(2*hop_length):
                cor = torch.dot(likely_probe/ms.MAX_WAV_VALUE, audio_patch[i:i+hop_length]/ms.MAX_WAV_VALUE)
                if cor >= max_cor:
                    max_cor = cor
                    max_i = i
                else:
                    continue
            print("The most similar segment index is found at {}, has value {}".format(max_i, max_cor))
            candidate_frame = audio_patch[max_i:max_i+win_length]
            candidate_frame = candidate_frame * window / win_adjust
            plt.figure('frame_compare'+str(index))
            plt.plot(comparation[index])
            plt.plot(candidate_frame, 'y')
            plt.show()
            pl_frames[index] = candidate_frame #* gain_factor
            print("Number {} frame`s compensation complete.".format(index))
            gain_factor = gain_factor - 0.2
    
    compensate_audio = frame2wav(pl_frames.numpy(), hop_length)

    # return compensate_audio


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-c1', '--config1', type=str, default='config.json',
                        help='JSON file for configuration')
    parser.add_argument('-c2', '--config2', type=str, default='DNN_spec/config.json',
                        help='JSON file for configuration')
    # parser.add_argument('-a', "--audio_file", default="../pl_wave/LJ050-0223/LJ050-0223_10percent_pl.npy")
    parser.add_argument('-dp', '--DNN_path', default="../DNN_checkpoints/MEL/DNN_net_28",
                    help='Path to DNN checkpoint with model')
    parser.add_argument('-wp', '--waveglow_path', default="../waveglow_checkpoints/waveglow_10000",
                        help='Path to waveglow decoder checkpoint with model')
    # parser.add_argument('-o', "--output_dir", default="../pl_wave/")
    parser.add_argument("-s", "--sigma", default=0.8, type=float)
    parser.add_argument("--sampling_rate", default=16000, type=int)
    parser.add_argument("--is_fp16", action="store_true")
    parser.add_argument("-d", "--denoiser_strength", default=0.0, type=float,
                        help='Removes model bias. Start with 0.1 and adjust')

    args = parser.parse_args()

    filename = 'pl_files.txt'
    with open(filename, encoding='utf-8') as f:
        files = f.readlines()
        files = [f.rstrip() for f in files]
    num_files = len(files)
    # Parse configs.  Globals nicer in this case
    with open(args.config1) as f1:
        data = f1.read()
    config1 = json.loads(data)
    global waveglow_config
    waveglow_config = config1["waveglow_config"]

    with open(args.config2) as f2:
        data = f2.read()
    config2 = json.loads(data)

    global DNN_data_config
    DNN_data_config = config2["DNN_data_config"]
    global DNN_net_config
    DNN_net_config = config2["DNN_net_config"]
    DNN_check_files = os.listdir('../DNN_checkpoints/MEL/')


    audio_files = []
    for whole_files in os.listdir(files[1]):
        if whole_files[-3:] == "npy":
            audio_files.append(whole_files)

    # for check_item in DNN_check_files:
    # DNN_path = os.path.join('../DNN_checkpoints/MEL/', check_item)
    savepath = os.path.join(files[1], 'DNN_28_wg_10000')
    if not os.path.exists(savepath):
        os.mkdir(savepath)
    # for temp_item in audio_files[:20]:
    #     file_item = os.path.join(files[1], temp_item)
        
    # pl_audio, _ = librosa.load('D:/VCwork-Py/waveglow-modified/LJSpeech-1.0-16k/LJ050-0223.wav', sr=16000, mono=True)
    # pl_audio = pl_audio * ms.MAX_WAV_VALUE
    # origin_audio, _ = librosa.load('../pl_wave/LJ050-0223/LJ050-0223_0percent_pl.wav', sr=16000, mono=True)
    # origin_audio = origin_audio * ms.MAX_WAV_VALUE
    file_item = '../pl_wave/plRate_30/LJ050-0265_30percent_pl.npy'
    origin_frames = load_frames_to_torch('../pl_wave/plRate_0/LJ050-0265_0percent_pl.npy')
    audio_mend(file_item, DNN_data_config['win_length'], DNN_data_config['hop_length'], \
        args.DNN_path, args.waveglow_path, args.sigma, args.sampling_rate, args.is_fp16, args.denoiser_strength, origin_frames)



        # plt.figure('audio_compare')
        # plt.plot(origin_audio, 'b')
        # plt.plot(pl_audio, 'y')
        # plt.plot(plc_audio, 'r')
        # plt.show()
        # result_path = os.path.join(savepath, temp_item[:-4] + 'DNN_28_wg_10000.wav')
        # soundfile.write(result_path, plc_audio/32768.0, args.sampling_rate)
        # print('Updated wav file at {}'.format(savepath))