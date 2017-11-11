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
chapter_names = ["Chapter1"]
noise_names = []

print("Loading audio files...")
chapters, noise = ap.load_audio_files( audio_folder_path, chapter_names, noise_names )

# Concatenate audio from training chapters into one long vector
# Create training set of windowed, overlapping frames. Each column is a frame. 
# Then scale for computational purposes.

print("Creating training & test sets...")
training_chapter_names = ["Chapter1"]
audio_time_series_train, fs = ap.concatenate_audio( training_chapter_names, chapters )
x_train = ap.generate_frames( audio_time_series_train, fs, frame_time = 0.015 )
x_train_scaled = ap.scale_features( x_train, is_time_series = True )
x_train_scaled = np.reshape(x_train_scaled, (x_train_scaled.shape[1], x_train_scaled.shape[0], 1))

test_chapter_names = ["Chapter1"]
audio_time_series_test, fs = ap.concatenate_audio( test_chapter_names, chapters )
audio_time_series_test = audio_time_series_test[0:60*fs]
x_test = ap.generate_frames( audio_time_series_test, fs, frame_time = 0.015 )
x_test_scaled = ap.scale_features( x_test, is_time_series = True )
x_test_scaled = np.reshape(x_test_scaled, (x_test_scaled.shape[1], x_test_scaled.shape[0], 1))

# Build Neural Network
print("Preparing neural network for training...")
input_shape = (x_train.shape[0], 1)
num_filters = 20
filter_size = int(0.003*fs)
pool_size = 4
model = dcam.create_model( input_shape, num_filters, filter_size, pool_size )

# Train Neural Network
epochs = 1
batch_size = 100
model = dcam.train_model( model = model, inputs = x_train_scaled, labels = x_train_scaled, epochs = epochs, batch_size = batch_size )

# Save/load model
model_save_path = parent_cwd + "/Saved_Models/Current_CNN_Model"
dcam.save_model(model, model_save_path)
#load_path = parent_cwd + "/Saved_Models/Current_CNN_Model"
#model = dcam.load_model_(load_path)

# Cluster training utterances using smallest encoded layer. 
# Then match test set utterances with closest utterances in training utterance embedding
K = int( x_train.shape[1] / 100 )

print("Encoding & flattening training/test sets...")
x_train_encoded_flattened = clus.encode_and_flatten(model, x_train_scaled)
x_test_encoded_flattened = clus.encode_and_flatten(model, x_test_scaled)

print("K-Means clustering encoded training set...")
cluster_model = clus.create_kmeans_model( K, x_train_encoded_flattened )
kmeans_save_path = parent_cwd + "/Saved_Models/Current_KMeans_Model"
pickle.dump(cluster_model, open(kmeans_save_path, 'wb'))

print("Finding K-exemplars for encoded training set...")
K_exemplar_indices = clus.find_K_exemplars( x_train_encoded_flattened, cluster_model)

print("Matching test set with closest utterances in encoded space...")
x_test_prediction_indices = clus.match_routine(cluster_model, K_exemplar_indices, x_test_encoded_flattened)

# Use training utterances to reconstruct test set audio
# Save audio to .wav file

print("Rebuilding test set audio file & saving to memory...")
test_set_audio_rebuilt = ap.rebuild_audio(x_test_prediction_indices, x_train)	
scipy.io.wavfile.write( filename = parent_cwd + "/Audio_Files/Output_Test.wav", rate = fs, data = test_set_audio_rebuilt)