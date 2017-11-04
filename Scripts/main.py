import audio_preprocessing as ap
import dca_model as dcam
import clustering as clus
import numpy as np
import scipy.io.wavfile
import os 

# Get path locating audio files
cwd = os.getcwd()
parent_cwd = os.path.abspath(os.path.join(cwd, os.pardir))
audio_folder_path = parent_cwd + "/Audio_Files"

# Load audio files
chapter_names = ["Chapter1", "Chapter2"]
noise_names = []
chapters, noise = ap.load_audio_files( audio_folder_path, chapter_names, noise_names )

# Concatenate audio for training sets
training_chapter_names = ["Chapter1"]
audio_time_series_train, fs = ap.concatenate_audio( training_chapter_names, chapters )
audio_time_series_train = audio_time_series_train[0:6*fs] # Take snippet of training data for testing purposes
#audio_downsampled, fs_new = ap.downsample( audio_time_series_train, fs, 16000)

# Create training set of windowed, overlapping frames. Each column is a frame.
frame_time = 0.015
x_train = ap.generate_train_time_series( audio_time_series_train, fs, frame_time )

# Scale the training set by the maximal element to aid in training/computation of neural network
x_train_scaled = ap.scale_features( x_train, is_time_series = True )

# Build Neural Network
input_shape = (x_train_scaled.shape[0], 1)
num_filters = 20
filter_size = int(0.003*fs)
pool_size = 4
model = dcam.create_model( input_shape, num_filters, filter_size, pool_size )

# Train Neural Network
epochs = 1
batch_size = 100
x_train_scaled = np.reshape(x_train_scaled, (x_train_scaled.shape[1], x_train_scaled.shape[0], 1))
model = dcam.train_model( model = model, inputs = x_train_scaled, labels = x_train_scaled, epochs = epochs, batch_size = batch_size )

# Save/load model
save_path = parent_cwd + "/Saved_Models/Current_Model"
dcam.save_model(model, save_path)
#load_path = "/Users/franklongueira/Desktop/Thesis/Saved_Models/Current_Model"
#model = dcam.load_model_(load_path)

# Cluster training utterances using smallest encoded layer. 
# Then match test set utterances with closest utterances in training utterance embedding
# Use training utterances to reconstruct test set audio
K = int( x_train.shape[1] )
test_set_scaled = x_train_scaled
test_set_prediction_indices = clus.cluster_and_match_routine(K, model, x_train_scaled, test_set_scaled)
test_set_audio_rebuilt = ap.rebuild_audio(test_set_prediction_indices, x_train)	

scipy.io.wavfile.write( filename = parent_cwd + "/Audio_Files/Output_Test.wav", rate = fs, data = test_set_audio_rebuilt)