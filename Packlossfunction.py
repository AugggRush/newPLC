import numpy as np
import random
import os
import librosa
import soundfile
from scipy.io.wavfile import read
import matplotlib.pyplot as plt


def packlossf(PackLoss, theo_pack_loss_maxlen, speech):#, speech

	theo_pack_loss_rate = PackLoss /100 
	
	total_packs = len(speech)
	if theo_pack_loss_maxlen == 0:
		q = 1 - theo_pack_loss_rate
	else:
		q = 1 / theo_pack_loss_maxlen

	p = (theo_pack_loss_rate*q)/(1 - theo_pack_loss_rate)

	check = 100
	while check >= 5 :
		good = 1
		packets = []
		pack_num = 1
		while pack_num <= total_packs:
			if good == 1:
				packets.append(good) 
				good = int (random.random() > p  ) #判断是否大于p，如果大于p，则返回1，否则返回0，相当于突发丢包得概率是p
			elif good == 0:
				packets.append(good) 
				good = int (random.random() > (1- q)) #判断是否大于1-q，如果大于1-q，则返回1，否则返回0，相当于连续丢包得概率是1-q
			else:
				print('error\n')
				break
			pack_num = pack_num + 1
		# fid = open('Loss_Pattern_py.txt','w')
		# # print(packets)
		# fid.writelines(str(packets))
		# fid.close()
		
		received_packs = np.sum(packets)
		act_pack_loss_rate = 1 - received_packs/total_packs
		check = abs(theo_pack_loss_rate - act_pack_loss_rate) / theo_pack_loss_rate * 100
		print(check)
					

	# print(theo_pack_loss_rate)
	act_pack_loss_rate = 1 - received_packs/total_packs
	# print(act_pack_loss_rate)
	speech_PL = np.array(speech)

	for iframe in range(len(speech_PL)):
		if packets[iframe] == 0:
			speech_PL[iframe,:] = 0



	return packets, theo_pack_loss_rate, speech_PL

def win_generate(winLen):

	bins = np.arange(0.5, winLen, dtype=np.float)
	win = np.sin(bins / winLen * np.pi)
	return win

def wav2frame(audio, framelen, hoplen):
	num_frames = int(np.ceil(len(audio)/hoplen))
	# framecount = 0
	frames = np.zeros([int(num_frames), framelen])
	win = win_generate(framelen)
	audio_pd = np.pad(audio, [0, int((num_frames-1) * hoplen + framelen - len(audio))],
                'constant')
	win_adjust = np.zeros(int((num_frames - 1) * hoplen + framelen), dtype=float)
	for i in range(num_frames):
		win_adjust[i*hoplen:i*hoplen+framelen] = win_adjust[i*hoplen:i*hoplen+framelen] + (win * win)
	win_adjust = np.sqrt(framelen*win_adjust)
	for t in range(num_frames):
		pieceFrame = audio_pd[t*hoplen:t*hoplen+framelen] * win / win_adjust[t*hoplen:t*hoplen+framelen]
		frames[t] = pieceFrame
	# while len(audio_pd) > framelen:
	# 	pieceFrame = np.array(audio_pd[:framelen]*win)
	# 	frames[framecount] = pieceFrame
	# 	audio_pd = audio_pd[hoplen:]
	# 	framecount = framecount + 1
	# print(framecount)
	return frames

def frame2wav(frames, hoplen, wavlen = -1):

	raw, col = frames.shape
	win = win_generate(col)
	win_adjust = np.zeros(int((raw - 1) * hoplen + col), dtype=float)
	for i in range(raw):
		win_adjust[i*hoplen:i*hoplen+col] = win_adjust[i*hoplen:i*hoplen+col] + (win * win)
	win_adjust = np.sqrt(win_adjust / col)
	placeholder = np.zeros(int((raw - 1) * hoplen + col), dtype = np.float)
	for iframe in range(raw):
		placeholder[iframe*hoplen:iframe*hoplen+col] = placeholder[iframe*hoplen:iframe*hoplen+col] + \
		                                               frames[iframe] * win / win_adjust[iframe*hoplen:iframe*hoplen+col]
	if wavlen == -1:
	    audio = placeholder		
	else:
	    audio = placeholder[:wavlen]

	return audio



def activatePL(filename, 
               sample_rate,
			   framelen,
			   hoplen,
               PackLoss, 
			   theo_pack_loss_maxlen, 
			   savepath):

	sampling_rate, audio = read(filename)
	frames = wav2frame(audio, framelen, hoplen)

	pl_label, plRate, plFrame = packlossf(PackLoss, theo_pack_loss_maxlen, frames)
	
	plAudio = frame2wav(plFrame, hoplen, len(audio))

	strName = str.split(filename, '/')[-1][:-4]

	saveAudio = os.path.join(savepath, strName+'_'+str(int(plRate*100))+'percent_pl.wav')
	saveLabel = os.path.join(savepath, strName+'_'+str(int(plRate*100))+'percent_pl.txt')
	np.savetxt(saveLabel, np.array(pl_label), fmt='%d')
	# fid = open(saveLabel,'w')
	# # print(packets)
	# fid.writelines(str(pl_label))
	# fid.close()
	np.save(saveAudio[:-4], plFrame)
	soundfile.write(saveAudio, plAudio/32768.0, sample_rate)
	print('Updated wav file at {}'.format(filename))


	return saveAudio, saveLabel



# def pl_detection(filename,
#                 savename,
#                 sample_rate,
#                 frame_dur = 0.02,   # frame duration 20ms, same as one packet length
#                 hop_dur = 0.01):
#     '''Detectes first one lossed packet and send to recover'''
#     '''Return boolean isLoss to determin wither packets loss in the file and append wave'''

#     frame_len = int(frame_dur * sample_rate)
#     hop_len = int(hop_dur * sample_rate)
#     audio, _ = librosa.load(filename, sr=sample_rate, mono=True)
#     originLen = len(audio)
#     originwave = audio
#     num_frames = 1 + int(np.ceil(float(np.abs(len(originwave) - frame_len)) / hop_len))
#     pl_labels = np.loadtxt(filename[:-4]+'.txt', dtype=int)

#     print(pl_labels)
#     if num_frames != len(pl_labels):
#         raise ValueError('packets loss label does not match the audios    '
#                         'the frame number of this audio wave is {}    '.format(num_frames),
#                         'but the number of labels is {}     '.format(len(pl_labels)),
#                         'Please check it')

#     audio = np.pad(audio, [0, frame_len - hop_len],
#                 'constant')
#     seed_Len = 0
#     for index in range(num_frames):
#         if pl_labels[index] == 0:
#             print('packet loss encountered at frame {}'.format(index))
#             isloss = 1
#             pl_labels[index] = 1
#             break
#         else:
#             print('There is no packet loss encountered at frame{}'.format(index))
#             isloss = 0
#     seed_Len = index * hop_len

#     if isloss:
#         wavform = originwave[:seed_Len]
#     else:
#         wavform = originwave[:originLen]

#     write_wav(wavform, sample_rate, savename)

#     np.savetxt(filename[:-4]+'.txt', np.array(pl_labels), fmt='%d')

#     return isloss, originwave[:seed_Len], originwave[seed_Len+frame_len:originLen]

if __name__ == '__main__':
    
	audioname, labelname = activatePL('D:/VCwork-Py/waveglow-modified/LJSpeech-1.0-16k/LJ001-0096.wav',
	                                  16000,
									  320, 160, 10, 2, 'D:/VCwork-Py/waveglow-modified/pl_wave/')
	plaudio, _ = librosa.load(audioname, sr=16000, mono=True)
	pl_label = np.loadtxt(labelname, dtype=int)
	plframe = wav2frame(plaudio, 320, 160)
	print(plframe.shape)
	print(pl_label.shape)
	# winValue = win_generate(1024)



	audio, _ = librosa.load('D:/VCwork-Py/waveglow-modified/LJSpeech-1.0-16k/LJ001-0096.wav', sr=16000, mono=True)
	# rec = frame2wav(wav2frame(audio, 512, 128), 128, len(audio))
	plt.subplot(211)
	plt.plot(audio)
	plt.subplot(212)
	plt.plot(plaudio-audio)
	plt.show()


