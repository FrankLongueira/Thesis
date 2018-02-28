import audio_preprocessing as ap
import cnn_model as cnn
import numpy as np
import scipy.io.wavfile
import os
from subprocess import call

print("Getting paths to audio files...")
cwd = os.getcwd()
parent_cwd = os.path.abspath(os.path.join(cwd, os.pardir))
audio_folder_path = parent_cwd + "/Audio_Files/"

print("Creating training, validation, and test sets...")
snr_db = 5
frame_time = 0.020
"""
# Generate training set
audio_time_series_train, fs = ap.load_audio( audio_folder_path, audio_filename = "Chapter1.wav")
audio_time_series_train_noise, fs = ap.load_audio( audio_folder_path, audio_filename = "Chapter1_Babble.wav")
audio_time_series_train_noisy = ap.combine_clean_and_noise(audio_time_series_train, audio_time_series_train_noise, snr_db)
train_mu = np.mean( audio_time_series_train )
train_std = np.std( audio_time_series_train )

train_clean = ap.generate_input(  audio_time_series_train, fs, frame_time, train_mu, train_std )
train_noisy = ap.generate_input(  audio_time_series_train_noisy, fs, frame_time, train_mu, train_std )

# Generate validation set
audio_time_series_validation, fs = ap.load_audio( audio_folder_path, audio_filename = "Chapter2_5_Min.wav")
audio_time_series_validation_noise, fs = ap.load_audio( audio_folder_path, audio_filename = "Chapter2_5_Min_Babble.wav")
audio_time_series_validation_noisy = ap.combine_clean_and_noise(audio_time_series_validation, audio_time_series_validation_noise, snr_db)

validation_clean = ap.generate_input(  audio_time_series_validation, fs, frame_time, train_mu, train_std )
validation_noisy = ap.generate_input(  audio_time_series_validation_noisy, fs, frame_time, train_mu, train_std )

# Generate test set
audio_time_series_test, fs = ap.load_audio( audio_folder_path, audio_filename = "Chapter3_1_Min.wav")
audio_time_series_test_noise, fs = ap.load_audio( audio_folder_path, audio_filename = "Chapter3_1_Min_Babble.wav")
audio_time_series_test_noisy = ap.combine_clean_and_noise(audio_time_series_test, audio_time_series_test_noise, snr_db)

test_clean = ap.generate_input(  audio_time_series_test, fs, frame_time, train_mu, train_std )
test_noisy = ap.generate_input(  audio_time_series_test_noisy, fs, frame_time, train_mu, train_std )

print("Preparing neural network for training...")
input_shape = (train_noisy.shape[1], 1)
epochs = 150
batch_size = 100
filter_size_per_hidden_layer = [0.005, 0.005, 0.005, 0.005, 0.005]
filter_size_output_layer = 0.005
num_filters_per_hidden_layer = [25, 25, 50, 50, 100]
"""
#	model = cnn.create_model( input_shape, num_filters_per_hidden_layer, map(int, list(np.array(filter_size_per_hidden_layer)*fs)), int(filter_size_output_layer*fs) )
#	model, history = cnn.train_model( 	model = model, 
#										train_inputs = train_noisy, 
#										train_labels = train_clean, 
#										epochs = epochs,
#										batch_size = batch_size,
#										validation_inputs = validation_noisy,
#										validation_labels = validation_clean,
#										filepath = model_save_path)
#	cnn.save_model(model, model_save_path)

#scipy.io.wavfile.write( filename = parent_cwd + "/Audio_Files/Test_Files/" + "CleanValidation_1min.wav", rate = fs, data = audio_time_series_validation.astype('int16')[0:(60*fs)])
#scipy.io.wavfile.write( filename = parent_cwd + "/Audio_Files/Test_Files/" + "NoisyValidation_5dB_1min.wav", rate = fs, data = audio_time_series_validation_noisy.astype('int16')[0:(60*fs)])

#model_names = ["Model25", "Model26", "Model27", "Model30", "Model31", "Model33", "Model38", "Model41", "Model53_PReLU", "Model_65"]

#model_names = ["Model53_PReLU", "Model_65"]
#for model_name in model_names:
#	model_save_path = parent_cwd + "/Saved_Models/" + model_name
#	model = cnn.load_model_(model_save_path)

#	print("Getting CNN output for noisy test set input...")
#	test_filtered_frames = (train_std * cnn.get_output_multiple_batches(model, validation_noisy)) + train_mu
#	print("Perfectly reconstructing filtered test set audio & saving to memory...")
#	test_filtered = ap.rebuild_audio( test_filtered_frames )
#	scipy.io.wavfile.write( filename = parent_cwd + "/Audio_Files/Test_Files/" + model_name + "_FilteredValidation.wav", rate = fs, data = test_filtered)
#	scipy.io.wavfile.write( filename = parent_cwd + "/Audio_Files/Test_Files/" + model_name + "_FilteredValidation_1min.wav", rate = fs, data = test_filtered[0:(60*fs)])
os.chdir(parent_cwd + "/Audio_Files/Test_Files")
#call( ["./PESQ", "+16000", "CleanValidation_1min.wav", model_name + "_FilteredValidation_1min.wav"] )
call( ["./PESQ", "+16000", "CleanValidation_1min.wav", "NoisyValidation_5dB_1min.wav"] )

#	summary_stats_filename = parent_cwd + "/Saved_Models/Model_Descriptions.txt"
#	cnn.summary_statistics( summary_stats_filename, model_name, history, frame_time, snr_db, 
#						 	num_filters_per_hidden_layer, filter_size_per_hidden_layer, filter_size_output_layer,
#						 	epochs, batch_size)
	
