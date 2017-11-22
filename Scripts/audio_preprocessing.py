from os.path import join
import librosa
import librosa.display
import numpy as np
import scipy.signal
import scipy.io.wavfile
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale
import math


def round_up_to_even(x):
    return int(math.ceil(x / 2.) * 2)
    
def next_pow2(x):
    return 2**(x-1).bit_length()
    
def apply_window(frame):

	M = frame.size
	window = scipy.signal.hann(M)
	
	return(frame*window)

def overlapp_add_reconstruction( frame1_windowed, frame2_windowed ):
	
	M = frame2_windowed.size
	R = M/2
	output = np.zeros(frame1_windowed.size + R)
	output[:frame1_windowed.size] = frame1_windowed
	output[(frame1_windowed.size-R):] = output[(frame1_windowed.size-R):] + frame2_windowed
	
	return(output)

def load_audio_files( audio_folder_path, chapter_names, noise_names ):
	
	chapters = {}
	for chapter_name in chapter_names:
		file_path = join(audio_folder_path, chapter_name + ".wav")
		fs, audio_time_series = scipy.io.wavfile.read(file_path)
		chapters[ chapter_name ] = (audio_time_series, fs)
	
	noise = {}
	for noise_name in noise_names:
		file_path = join(audio_folder_path, noise_name + ".wav")
		fs, audio_time_series = scipy.io.wavfile.read(file_path)
		noise[ noise_name ] = (audio_time_series, fs)
	
	return chapters, noise

def concatenate_audio( names, dict ):

	arrays_to_concatenate = []
	fs = dict[names[0]][1]
	for name in names:
		arrays_to_concatenate.append(dict[name][0])

	return(np.concatenate(arrays_to_concatenate) , fs)
def combine_clean_and_noise(audio_time_series_train, audio_time_series_noise, snr_db):
	if( audio_time_series_train.size <= audio_time_series_noise.size ):
		audio_time_series_noise = audio_time_series_noise[0:audio_time_series_train.size] 
	else:
		audio_time_series_train = audio_time_series_train.size[0:audio_time_series_noise.size]
	
	A_train = np.mean(np.absolute(audio_time_series_train))
	A_noise = np.mean(np.absolute(audio_time_series_noise))
	A_noise_targ = np.sqrt( (A_train**2) / (10**(snr_db/10.)) )
	
	scaling_coeff = A_noise_targ / A_noise
	
	print(A_train, A_noise, A_noise_targ, scaling_coeff)
	audio_time_series_noise = scaling_coeff * audio_time_series_noise
	
	combined_result = audio_time_series_train + audio_time_series_noise
	
	return(combined_result)

def downsample( audio, orig_sr, targ_sr):
		audio = audio.astype('float')
		audio_downsampled = librosa.resample(audio, orig_sr, targ_sr)
		audio_downsampled = audio_downsampled.astype('int16')
		return audio_downsampled, targ_sr
		
def generate_frames( audio_time_series_train, fs, frame_time, lag = 0.5 ):
	frame_length = round_up_to_even( frame_time * fs ) 
	total_time_steps = int((audio_time_series_train.size / (frame_length * lag)) - 1)
	
	x_train = np.zeros( shape = (frame_length, total_time_steps) )

	for i in range(0, (total_time_steps - 1)):
		x_train[:, i] =  apply_window( audio_time_series_train[ i*(frame_length / 2) : (i*(frame_length / 2) + frame_length) ] )
	return(x_train)
	
def generate_train_features( x_train ):
	n = next_pow2(x_train.shape[0])
	x_train_features = np.zeros( shape = (n,  x_train.shape[1]))
	for i in range(0, x_train.shape[1]):
		x_train_features[:, i] = np.abs(np.fft.fft( a = x_train[:, i], n = n ))
	return(x_train_features)
	
def scale_features( x_train_features, mu, std ):
	
	x_train_features = (x_train_features - mu) / float(std)
		
	return(x_train_features.T)
	
def rebuild_audio_from_indices( predicted_time_series_indices, x_train ):
	output = x_train[:, predicted_time_series_indices[0]]
	
	for i in predicted_time_series_indices[1:]:
		output = overlapp_add_reconstruction(output, x_train[:, i])
		
	return(output.astype('int16'))
	
def rebuild_audio( x_test ):
	output = x_test[0, :]
	for i in xrange(1, x_test.shape[0]-1):
		output = overlapp_add_reconstruction(output, x_test[i, :])
		
	return(output.astype('int16'))