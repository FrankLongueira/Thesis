from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv1D, MaxPooling1D, UpSampling1D
from keras.layers.normalization import BatchNormalization
from keras.layers import LSTM
from keras.models import load_model
from keras import backend as K
from keras.layers import Activation
import numpy as np
from keras.callbacks import ModelCheckpoint
from keras.callbacks import History 
from keras.callbacks import EarlyStopping

def create_model(input_shape, num_filters_per_hidden_layer, filter_size_per_hidden_layer, filter_size_output_layer ):
	
	model = Sequential()
	
	model.add(Conv1D(filters = num_filters_per_hidden_layer[0], kernel_size = filter_size_per_hidden_layer[0], padding='same', input_shape = input_shape))
	model.add(BatchNormalization())
	model.add(Activation('relu'))

	for num_filters, filter_size in zip(num_filters_per_hidden_layer[1:], filter_size_per_hidden_layer[1:]): 
		model.add(Conv1D(filters = num_filters, kernel_size = filter_size, padding='same'))
		model.add(BatchNormalization())
		model.add(Activation('relu'))
	
	model.add(Conv1D(1, kernel_size = filter_size_output_layer, padding='same'))
	
	return(model)
	
def train_model( model, train_inputs, train_labels, epochs, batch_size, validation_inputs, validation_labels, filepath ):

	model.compile(optimizer = 'adam', loss='mean_squared_error')
	
	checkpointer = ModelCheckpoint(filepath = filepath, monitor = "val_loss", verbose = 1, mode = 'min', save_best_only = True)
	early_stopping = EarlyStopping(monitor = 'val_loss', min_delta = 0, patience = 20, verbose = 1, mode='auto')

	history = model.fit(	train_inputs, train_labels,
            				epochs = epochs,
                			batch_size = batch_size,
                			shuffle = True,
                			validation_data = (validation_inputs, validation_labels),
                			callbacks = [checkpointer, early_stopping])
    
	model = load_model(filepath)

	return(model, history)

def save_model( model, save_path ):
	model.save(save_path)

def load_model_( load_path ):
	model = load_model( load_path )
	return(model)

def predict_model( model, inputs ):
	predictions = model.predict(inputs, batch_size = None, verbose=0, steps=None)
	return(predictions)
	
def get_output( model, new_input ):
	get_output = K.function([model.layers[0].input, K.learning_phase()],
                                  	  [model.layers[ len(model.layers) - 1 ].output])
	layer_output = get_output([new_input, 0])[0]
	
	return(layer_output)
	
def get_output_multiple_batches(model, input_frames, batch_size = 100):
	
	batches_output_frames_holder = []
	for i in xrange(0, input_frames.shape[0], batch_size):
		batch_input_frames = input_frames[ i:i+batch_size , :, : ]
		batch_output_frames = get_output( model, batch_input_frames )
		batch_output_frames = np.reshape(batch_output_frames, (batch_output_frames.shape[0], -1)) 
		batches_output_frames_holder.append(batch_output_frames)
		
	output_frames_concatenated = np.concatenate( batches_output_frames_holder, axis = 0 )
	
	return(output_frames_concatenated)
	
def summary_statistics( model_name, history, frame_time, snr_db, 
						num_filters_per_hidden_layer, filter_size_per_hidden_layer, filter_size_output_layer,
						epochs, batch_size):
	best_val_loss = min( history.history["val_loss"] )				
	best_epoch_index = history.history["val_loss"].index( best_val_loss )
	best_train_loss = history.history["loss"][ best_epoch_index ]
	
	print( "\tFCNN Name: " + model_name )
	print( "\tNumber of Filters Per Hidden Layer: " + ','.join(map(str, num_filters_per_hidden_layer)) ) 
	print( "\tFilter Size Per Hidden Layer: " + ','.join(map(str, filter_size_per_hidden_layer)) )
	print( "\tFilter Size for Output Layer: " + str( filter_size_output_layer ) )
	print( "\tFrame Time: " + frame_time*1000 + str( " ms" ) )
	print( "\tTotal Epochs: " + str(epochs) )
	print( "\tBatch Size: " + str(batch_size) + " examples"
	print( "\tSNR: " + str( snr_db ) + str( " dB" ) )
	print( "\tBest Epoch: " + str(  best_epoch_index + 1 ) )
	print( "\tValidation Loss: " + str( best_val_loss ) )
	print( "Training Loss: " + str( best_train_loss ) )
	