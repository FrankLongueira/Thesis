import audio_preprocessing as ap
import dca_model as dcam
import clustering as clus
import numpy as np
import scipy.io.wavfile
import os 
import pickle

# Get path locating audio files
cwd = os.getcwd()
parent_cwd = os.path.abspath(os.path.join(cwd, os.pardir))
audio_folder_path = parent_cwd + "/Audio_Files"

# Load audio files and store into dictionary for ease of access
chapter_names = ["Chapter1", "Chapter2"]
noise_names = ["Chapter1_Babble", "Chapter2_Babble"]

print("Loading audio files...")
chapters, noise = ap.load_audio_files( audio_folder_path, chapter_names, noise_names )

# Concatenate audio from training chapters into one long vector
# Create training set of windowed, overlapping frames. Each column is a frame. 
# Then scale for computational purposes.

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

test_chapter_names = ["Chapter2"]
test_noise_names = ["Chapter2_Babble"]
audio_time_series_test, fs = ap.concatenate_audio( test_chapter_names, chapters )
audio_time_series_test_noise, fs = ap.concatenate_audio( test_noise_names, noise )
audio_time_series_test = audio_time_series_train[0:60*fs]
audio_time_series_test_noise = audio_time_series_train_noise[0:60*fs]

audio_time_series_test_noisy = ap.combine_clean_and_noise(audio_time_series_test, audio_time_series_test_noise, snr_db)

x_test_noisy = ap.generate_frames( audio_time_series_test_noisy, fs, frame_time = 0.020 )
x_test_noisy_scaled = ap.scale_features( x_test_noisy, train_mu, train_std )
x_test_noisy_scaled_input = np.reshape(x_test_noisy_scaled, (x_test_noisy_scaled.shape[0], x_test_noisy_scaled.shape[1], 1))

# Build Neural Network
print("Preparing neural network for training...")
input_shape = (x_train_noisy.shape[0], 1)
filter_size = int(0.010*fs)

model = dcam.create_model( input_shape, filter_size )

# Train Neural Network
epochs = 50
batch_size = 100
model = dcam.train_model( model = model, inputs = x_train_noisy_scaled_input, labels = x_train_scaled_input, epochs = epochs, batch_size = batch_size )

# Save/load model
model_save_path = parent_cwd + "/Saved_Models/Current_CNN_Model"
dcam.save_model(model, model_save_path)
#load_path = parent_cwd + "/Saved_Models/Current_CNN_Model"
#model = dcam.load_model_(load_path)

# Cluster training utterances using smallest encoded layer. 
# Then match test set utterances with closest utterances in training utterance embedding
print("Encoding & flattening training/test sets...")
#x_train_encoded_flattened = clus.encode_and_flatten(model, x_train_scaled_input)
x_test_encoded_flattened = (train_std * clus.encode_and_flatten(model, x_test_noisy_scaled_input)) + train_mu
	
#print("Matching test set with closest utterances in encoded space...")
#x_test_prediction_indices = np.ravel( clus.KNN_routine(x_train_encoded_flattened, x_test_encoded_flattened, n_jobs = 3))

# Use training utterances to reconstruct test set audio
# Save audio to .wav file

print("Rebuilding test set audio file & saving to memory...")
#test_set_audio_rebuilt = ap.rebuild_audio_from_indices(x_test_prediction_indices, x_train)	
test_set_audio_rebuilt = ap.rebuild_audio( x_test_encoded_flattened )

scipy.io.wavfile.write( filename = parent_cwd + "/Audio_Files/Output_Test_Noisy.wav", rate = fs, data = audio_time_series_test_noisy.astype('int16'))
scipy.io.wavfile.write( filename = parent_cwd + "/Audio_Files/Output_Test.wav", rate = fs, data = test_set_audio_rebuilt)