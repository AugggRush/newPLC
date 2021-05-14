import os
import librosa
import soundfile
dataFilepath = "./Mel2test/"
# tarFilepath = "F:/DATA/LJSpeech-1/LJSpeech-1.0-16k/"
filenames = os.listdir(dataFilepath)

# for item in filenames:
#     print(item)
#     temp = os.path.join(dataFilepath, item)
#     audio, sr = librosa.load(temp, sr=22050)
#     print("origin samplerate is: {}".format(sr))
#     audio_16 = librosa.resample(audio, sr, 16000)
#     print("target sample rate is: {}".format(16000))
#     soundfile.write(os.path.join(tarFilepath, item), audio_16, 16000)
#     print("successfully resample audio {}".format(item))

# print(len(filenames))
# with open("train_files.txt", 'w+') as f:
#     for item in filenames[:-100]:
#         f.write('../LJSpeech-1.0-16k/'+item+'\n')
# with open("test_files.txt", 'w+') as f:
#     for item in filenames[-4:]:
#         f.write('../LJSpeech-1.0-16k/'+item+'\n')
with open("mel_files.txt", 'w+') as f:
    for item in filenames:
        f.write('Mel2test/'+item+'\n')