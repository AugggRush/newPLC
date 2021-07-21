import numpy as np
import random
import os
import librosa
import soundfile
from scipy.io.wavfile import read
import matplotlib.pyplot as plt


def packlossf(PackLoss, speech_frames):#, speech

	desire_pack_loss_rate = PackLoss /100 
	
	PG = 0
	PB = 0.5
	Gamma = 0.5
    # P =(1- gamma )( 1 - (PB-FER)/(PB-PG)))
	p = (1 - Gamma)*(1-( PB - desire_pack_loss_rate)/(PB - PG ))
    # % Q =(1- gamma ) (PB-FER)/(PB-PG))
	q = (1 - Gamma)*   ( PB - desire_pack_loss_rate)/(PB - PG )
	
	try:
		if(p > 1-q):
			raise ValueError("p > 1-q, 不符合Gilbert概率模型")
	except ValueError as e:
		print("引发异常：",repr(e))

	num_packets = speech_frames.shape[0]

    

	check = 100
	while check >= 10 :
		channel_good = 1
		packets = []
		pack_num = 1
		while pack_num <= num_packets:
			if channel_good == 1:
				packet_good = channel_good
				packets.append(packet_good) 
				#判断是否大于p，如果大于p，则信道状态返回1，否则返回0，相当于信道变坏概率是p
				channel_good = int (random.random() > p) 
			elif channel_good == 0:
				#判断是否大于PB，如果大于PB，则返回1 表示不丢，否则返回0，相当于信道坏的状况下丢包概率是PB
				packet_good = int (random.random() > PB)
				packets.append(packet_good) 
				#判断是否大于1-q，如果大于1-q，则返回1，否则返回0，相当于信道连续变坏的概率是1-q
				channel_good = int (random.random() > (1- q)) 
			else:
				print('error\n')
				break
			pack_num = pack_num + 1

		received_packs = np.sum(packets)
		act_pack_loss_rate = 1 - received_packs/num_packets
		check = abs(desire_pack_loss_rate - act_pack_loss_rate) / desire_pack_loss_rate * 100
		# print(check)
					

	print("desire_pack_loss_rate: {}".format(desire_pack_loss_rate))
	theo_pack_loss_rate = (q/(1 - Gamma))*PG + (p/(1 - Gamma))*PB
	act_pack_loss_rate = 1 - received_packs/num_packets
	print("theo_pack_loss_rate: {}".format(theo_pack_loss_rate))
	print("act_pack_loss_rate: {}".format(act_pack_loss_rate))
	speech_frames_PL = np.array(speech_frames)

	for iframe in range(len(speech_frames_PL)):
		if packets[iframe] == 0:
			speech_frames_PL[iframe,:] = 0

	return packets, desire_pack_loss_rate, speech_frames_PL

def win_generate(winLen):

	bins = np.arange(0.5, winLen, dtype=float)
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
	placeholder = np.zeros(int((raw - 1) * hoplen + col), dtype = float)
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
			   savepath):

	sampling_rate, audio = read(filename)
	frames = wav2frame(audio, framelen, hoplen)

	pl_label, plRate, plFrame = packlossf(PackLoss, frames)
	
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
	print('Updated wav file of {}'.format(filename))


	return saveAudio, saveLabel



if __name__ == '__main__':

	filename = 'test_files.txt'
	with open(filename, encoding='utf-8') as f:
		files = f.readlines()
		files = [f.rstrip() for f in files]
	random.seed(222)
	random.shuffle(files)
	num_files = len(files)
	for rate in [0, 30]:
		if not os.path.exists('../libri-temp/plRate_'+str(rate)):
			os.makedirs('../libri-temp/plRate_'+str(rate))
			savePath = '../libri-temp/plRate_'+str(rate)
		else:
			savePath = '../libri-temp/plRate_'+str(rate)
		for i in range(1):
			# filename = files[i]
			filename = 'F:/DATA/LibriSpeex/Train-100/8468-294887-0001.wav'
			audioname, labelname = activatePL(filename,
											16000,
											320, 160, rate, savePath)
			# plaudio, _ = librosa.load(audioname, sr=16000, mono=True)
			# pl_label = np.loadtxt(labelname, dtype=int)
			# plframe = wav2frame(plaudio, 320, 160)
			# print(plframe.shape)
			# print(pl_label.shape)
			# # winValue = win_generate(1024)



			# audio, _ = librosa.load('D:/VCwork-Py/waveglow-modified/LJSpeech-1.0-16k/LJ050-0223.wav', sr=16000, mono=True)
			# # rec = frame2wav(wav2frame(audio, 512, 128), 128, len(audio))
			# plt.subplot(211)
			# plt.plot(audio)
			# plt.subplot(212)
			# plt.plot(plaudio-audio)
			# plt.show()


