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
snr_db = -5
frame_time = 0.020

# Generate training set
audio_time_series_train, fs = ap.load_audio( audio_folder_path, audio_filename = "Chapter1.wav")
audio_time_series_train_noise, fs = ap.load_audio( audio_folder_path, audio_filename = "Chapter1_Babble.wav")
#audio_time_series_train, fs = ap.load_audio( audio_folder_path, audio_filename = "TriciaG_FineTune_25-30mins.wav")
#audio_time_series_train_noise, fs = ap.load_audio( audio_folder_path, audio_filename = "Chapter2_5_Min_Babble.wav")
audio_time_series_train_noisy = ap.combine_clean_and_noise(audio_time_series_train, audio_time_series_train_noise, snr_db)
train_mu = np.mean( audio_time_series_train )
train_std = np.std( audio_time_series_train )

"""
train_clean = ap.generate_input(  audio_time_series_train, fs, frame_time, train_mu, train_std )
train_noisy = ap.generate_input(  audio_time_series_train_noisy, fs, frame_time, train_mu, train_std )


# Generate validation set
audio_time_series_validation, fs = ap.load_audio( audio_folder_path, audio_filename = "Chapter2_5_Min.wav")
scipy.io.wavfile.write( filename = parent_cwd + "/Audio_Files/Test_Files/" + "CleanValidation_5min_" + str(snr_db) + "dB.wav", rate = fs, data = audio_time_series_validation.astype('int16'))
scipy.io.wavfile.write( filename = parent_cwd + "/Audio_Files/Test_Files/" + "CleanValidation_1min_" + str(snr_db) + "dB.wav", rate = fs, data = audio_time_series_validation.astype('int16')[0:(60*fs)])

audio_time_series_validation_noise, fs = ap.load_audio( audio_folder_path, audio_filename = "Chapter2_5_Min_Babble.wav")

audio_time_series_validation_noisy = ap.combine_clean_and_noise(audio_time_series_validation, audio_time_series_validation_noise, snr_db)
scipy.io.wavfile.write( filename = parent_cwd + "/Audio_Files/Test_Files/" + "NoisyValidation_5min_" + str(snr_db) + "dB.wav", rate = fs, data = audio_time_series_validation_noisy.astype('int16'))
scipy.io.wavfile.write( filename = parent_cwd + "/Audio_Files/Test_Files/" + "NoisyValidation_1min_" + str(snr_db) + "dB.wav", rate = fs, data = audio_time_series_validation_noisy.astype('int16')[0:(60*fs)])

validation_clean = ap.generate_input(  audio_time_series_validation, fs, frame_time, train_mu, train_std )
validation_noisy = ap.generate_input(  audio_time_series_validation_noisy, fs, frame_time, train_mu, train_std )
"""
# Generate test set
#audio_time_series_test, fs = ap.load_audio( audio_folder_path, audio_filename = "Chapter3_5_Min.wav")
audio_time_series_test, fs = ap.load_audio( audio_folder_path, audio_filename = "TriciaG_5min.wav")
#scipy.io.wavfile.write( filename = parent_cwd + "/Audio_Files/Test_Files/" + "CleanTest_Tricia_5min_" + str(snr_db) + "dB.wav", rate = fs, data = audio_time_series_test.astype('int16'))
#scipy.io.wavfile.write( filename = parent_cwd + "/Audio_Files/Test_Files/" + "CleanTest_Tricia_1min_" + str(snr_db) + "dB.wav", rate = fs, data = audio_time_series_test.astype('int16')[0:(60*fs)])

audio_time_series_test_noise, fs = ap.load_audio( audio_folder_path, audio_filename = "Chapter3_5_Min_Babble.wav")
audio_time_series_test_noisy = ap.combine_clean_and_noise(audio_time_series_test, audio_time_series_test_noise, snr_db)
#scipy.io.wavfile.write( filename = parent_cwd + "/Audio_Files/Test_Files/" + "NoisyTest_Tricia_5min_" + str(snr_db) + "dB.wav", rate = fs, data = audio_time_series_test_noisy.astype('int16'))
#scipy.io.wavfile.write( filename = parent_cwd + "/Audio_Files/Test_Files/" + "NoisyTest_Tricia_1min_" + str(snr_db) + "dB.wav", rate = fs, data = audio_time_series_test_noisy.astype('int16')[0:(60*fs)])

test_clean = ap.generate_input(  audio_time_series_test, fs, frame_time, train_mu, train_std )
test_noisy = ap.generate_input(  audio_time_series_test_noisy, fs, frame_time, train_mu, train_std )
"""
print("Preparing neural network for training...")
input_shape = (train_noisy.shape[1], 1)
epochs = 5
batch_size = 100
filter_size_per_hidden_layer = [0.005, 0.005, 0.005, 0.005, 0.005]
filter_size_output_layer = 0.005
num_filters_per_hidden_layer = [12, 25, 50, 100, 200]
patience = 20
"""
#model_name = "Model53_" + str(snr_db) + "dB"
model_name = "Model53_5dB"
model_save_path = parent_cwd + "/Saved_Models/" + model_name
model = cnn.load_model_(model_save_path)

#model_name_new = "Model53_" + "-5" + "dB_FINETUNED"
#model_save_path_new = parent_cwd + "/Saved_Models/" + model_name_new

#model = cnn.load_model_(model_save_path_new)

"""
model = cnn.create_model( input_shape, num_filters_per_hidden_layer, map(int, list(np.array(filter_size_per_hidden_layer)*fs)), int(filter_size_output_layer*fs) )

model, history = cnn.train_model( 	model = model, 
									train_inputs = train_noisy, 
									train_labels = train_clean, 
									epochs = epochs,
									batch_size = batch_size,
									validation_inputs = validation_noisy,
									validation_labels = validation_clean,
									filepath = model_save_path_new,
									patience = patience)
									
model_finetuned, history = cnn.train_model_finetune(  	model = model, 
														train_inputs = train_noisy, 
														train_labels = train_clean, 
														epochs = epochs,
														batch_size = batch_size)
"""

#cnn.save_model(model_finetuned, model_save_path_new)
#model = cnn.load_model_(model_save_path)

#model_name = model_name_new

print("Getting CNN output for noisy test set input...")
test_filtered_frames = (train_std * cnn.get_output_multiple_batches(model, test_noisy)) + train_mu

print("Perfectly reconstructing filtered test set audio & saving to memory...")
test_filtered = ap.rebuild_audio( test_filtered_frames )
scipy.io.wavfile.write( filename = parent_cwd + "/Audio_Files/Test_Files/" + model_name + "_FilteredTest_SameModel5dB_Test" + str(snr_db) + "dB_5min.wav", rate = fs, data = test_filtered)
scipy.io.wavfile.write( filename = parent_cwd + "/Audio_Files/Test_Files/" + model_name + "_FilteredTest_SameModel5dB_Test" + str(snr_db) + "dB_1min.wav", rate = fs, data = test_filtered[0:(60*fs)])

os.chdir(parent_cwd + "/Audio_Files/Test_Files")
#call( ["./PESQ", "+16000", "CleanTest_Tricia_1min_" + str(snr_db) + "dB.wav", "NoisyTest_Tricia_1min_" + str(snr_db) + "dB.wav"] )
#call( ["./PESQ", "+16000", "CleanTest_Tricia_1min_" + str(snr_db) + "dB.wav", model_name + "_FilteredTest_Trained5dB_Test" + str(snr_db) + "1min.wav"] )
#call( ["./PESQ", "+16000", "CleanTest_1min_" + str(snr_db) + "dB.wav", "NoisyTest_1min_" + str(snr_db) + "dB.wav"] )
call( ["./PESQ", "+16000", "CleanTest_Tricia_1min_" + str(snr_db) + "dB.wav", model_name + "_FilteredTest_SameModel5dB_Test" + str(snr_db) + "dB_1min.wav"] )
"""
summary_stats_filename = parent_cwd + "/Saved_Models/Model_Descriptions.txt"
cnn.summary_statistics( summary_stats_filename, model_name, history, frame_time, snr_db, 
						 	num_filters_per_hidden_layer, filter_size_per_hidden_layer, filter_size_output_layer,
						 	epochs, batch_size)
"""