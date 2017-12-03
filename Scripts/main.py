import audio_preprocessing as ap
import dca_model as dcam
import numpy as np
import scipy.io.wavfile
import os 

print("Getting paths to audio files...")
cwd = os.getcwd()
parent_cwd = os.path.abspath(os.path.join(cwd, os.pardir))
audio_folder_path = parent_cwd + "/Audio_Files"

print("Loading audio files...")
chapter_names = ["Chapter1", "Chapter2_5_Min"]
noise_names = ["Chapter1_Babble", "Chapter2_Babble_Train_5_Min", "Chapter2_Babble_Testing_5_Min"]
chapters, noise = ap.load_audio_files( audio_folder_path, chapter_names, noise_names )

print("Creating training & test sets...")
training_chapter_names = ["Chapter1"]
training_noise_names = ["Chapter1_Babble"]
audio_time_series_train, fs = ap.concatenate_audio( training_chapter_names, chapters )
train_mu = np.mean( audio_time_series_train )
train_std = np.std( audio_time_series_train )

audio_time_series_train_noise, fs = ap.concatenate_audio( training_noise_names, noise )

snr_db = 5
audio_time_series_train_noisy = ap.combine_clean_and_noise(audio_time_series_train, audio_time_series_train_noise, snr_db)

x_train = ap.generate_frames( audio_time_series_train, fs, frame_time = 0.020 )
x_train_scaled = ap.scale_features( x_train, train_mu, train_std )
x_train_scaled_input = np.reshape(x_train_scaled, (x_train_scaled.shape[0], x_train_scaled.shape[1], 1))

x_train_noisy = ap.generate_frames( audio_time_series_train_noisy, fs, frame_time = 0.020 )
x_train_noisy_scaled = ap.scale_features( x_train_noisy, train_mu, train_std )
x_train_noisy_scaled_input = np.reshape(x_train_noisy_scaled, (x_train_noisy_scaled.shape[0], x_train_noisy_scaled.shape[1], 1))

test_chapter_names = ["Chapter2_5_Min"]
test_train_noise_names = ["Chapter2_Babble_Train_5_Min"]
test_test_noise_names = ["Chapter2_Babble_Testing_5_Min"]
audio_time_series_test, fs = ap.concatenate_audio( test_chapter_names, chapters )
audio_time_series_test_noise_train, fs = ap.concatenate_audio( test_train_noise_names, noise )
audio_time_series_test_noise_test, fs = ap.concatenate_audio( test_test_noise_names, noise )

audio_time_series_test = audio_time_series_test[0:60*fs]

audio_time_series_test_noisy_train = ap.combine_clean_and_noise(audio_time_series_test, audio_time_series_test_noise_train, snr_db)
audio_time_series_test_noisy_test = ap.combine_clean_and_noise(audio_time_series_test, audio_time_series_test_noise_test, snr_db)

x_test_noisy_train = ap.generate_frames( audio_time_series_test_noisy_train, fs, frame_time = 0.020 )
x_test_noisy_train_scaled = ap.scale_features( x_test_noisy_train, train_mu, train_std )
x_test_noisy_train_scaled_input = np.reshape(x_test_noisy_train_scaled, (x_test_noisy_train_scaled.shape[0], x_test_noisy_train_scaled.shape[1], 1))

x_test_noisy_test = ap.generate_frames( audio_time_series_test_noisy_test, fs, frame_time = 0.020 )
x_test_noisy_test_scaled = ap.scale_features( x_test_noisy_test, train_mu, train_std )
x_test_noisy_test_scaled_input = np.reshape(x_test_noisy_test_scaled, (x_test_noisy_test_scaled.shape[0], x_test_noisy_test_scaled.shape[1], 1))

print("Preparing neural network for training...")
input_shape = (x_train_noisy.shape[0], 1)
filter_size = int(0.005*fs)
model = dcam.create_model( input_shape, filter_size )
epochs = 1
batch_size = 100
model = dcam.train_model( model = model, inputs = x_train_noisy_scaled_input, labels = x_train_scaled_input, epochs = epochs, batch_size = batch_size )

print( "Saving (Loading) trained model..." )
model_save_path = parent_cwd + "/Saved_Models/Model1"
dcam.save_model(model, model_save_path)
#load_path = parent_cwd + "/Saved_Models/Model1"
#model = dcam.load_model_(load_path)

print("Getting CNN output for noisy test set inputs...")
x_test_train_encoded_flattened = (train_std * dcam.get_output_multiple_batches(model, x_test_noisy_train_scaled_input)) + train_mu
x_test_test_encoded_flattened = (train_std * dcam.get_output_multiple_batches(model, x_test_noisy_test_scaled_input)) + train_mu

print("Perfectly reconstructing filtered test set audio & saving to memory...")
test_train_set_audio_rebuilt = ap.rebuild_audio( x_test_train_encoded_flattened )
test_test_set_audio_rebuilt = ap.rebuild_audio( x_test_test_encoded_flattened )

scipy.io.wavfile.write( filename = parent_cwd + "/Audio_Files/Noisy_Validation.wav", rate = fs, data = audio_time_series_test_noisy_train.astype('int16'))
scipy.io.wavfile.write( filename = parent_cwd + "/Audio_Files/Filtered_Validation.wav", rate = fs, data = test_train_set_audio_rebuilt)

scipy.io.wavfile.write( filename = parent_cwd + "/Audio_Files/Noisy_Test.wav", rate = fs, data = audio_time_series_test_noisy_test.astype('int16'))
scipy.io.wavfile.write( filename = parent_cwd + "/Audio_Files/Filtered_Test.wav", rate = fs, data = test_test_set_audio_rebuilt)