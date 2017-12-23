import audio_preprocessing as ap
import dca_model as dcam
import numpy as np
import scipy.io.wavfile
import os 

print("Getting paths to audio files...")
cwd = os.getcwd()
parent_cwd = os.path.abspath(os.path.join(cwd, os.pardir))
audio_folder_path = parent_cwd + "/Audio_Files/"

print("Creating training, validation, and test sets...")
snr_db = 5
frame_time = 0.020

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
audio_time_series_test, fs = ap.load_audio( audio_folder_path, audio_filename = "Chapter3_5_Min.wav")
audio_time_series_test_noise, fs = ap.load_audio( audio_folder_path, audio_filename = "Chapter3_5_Min_Babble.wav")
audio_time_series_test_noisy = ap.combine_clean_and_noise(audio_time_series_test, audio_time_series_test_noise, snr_db)

test_clean = ap.generate_input(  audio_time_series_test, fs, frame_time, train_mu, train_std )
test_noisy = ap.generate_input(  audio_time_series_test_noisy, fs, frame_time, train_mu, train_std )

print("Preparing neural network for training...")
input_shape = (train_noisy.shape[1], 1)
filter_size_per_hidden_layer = [int(0.005*fs), int(0.005*fs)]
num_filters_per_hidden_layer = [256, 256]
filter_size_output_layer = int(0.005*fs)
model = dcam.create_model( input_shape, num_filters_per_hidden_layer, filter_size_per_hidden_layer, filter_size_output_layer )
epochs = 125
batch_size = 100
model_name = "Model_Testing"
model_save_path = parent_cwd + "/Saved_Models/" + model_name

model, history = dcam.train_model( 	model = model, 
							train_inputs = train_noisy, 
							train_labels = train_clean, 
							epochs = epochs, 
							batch_size = batch_size,
							validation_inputs = validation_noisy,
							validation_labels = validation_clean,
							filepath = model_save_path)

print(history.history['val_loss'])
print( "Saving (Loading) trained model..." )
#model_save_path = parent_cwd + "/Saved_Models/" + model_name
#dcam.save_model(model, model_save_path)
#load_path = parent_cwd + "/Saved_Models/" + model_name
#model = dcam.load_model_(load_path)

print("Getting CNN output for noisy test set input...")
test_filtered_frames = (train_std * dcam.get_output_multiple_batches(model, test_noisy)) + train_mu

print("Perfectly reconstructing filtered test set audio & saving to memory...")
test_filtered = ap.rebuild_audio( test_filtered_frames )

scipy.io.wavfile.write( filename = parent_cwd + "/Audio_Files/" + model_name + "_NoisyTest.wav", rate = fs, data = audio_time_series_test_noisy.astype('int16'))
scipy.io.wavfile.write( filename = parent_cwd + "/Audio_Files/" + model_name + "_FilteredTest.wav", rate = fs, data = test_filtered)

print("Computing and printing summary statistics:")
print("\n")
dcam.summary_statistics( model_name, history, frame_time, snr_db, 
						 num_filters_per_hidden_layer, filter_size_per_hidden_layer, filter_size_output_layer,
						 epochs, batch_size)