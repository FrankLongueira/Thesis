from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv1D, MaxPooling1D, UpSampling1D
from keras.layers.normalization import BatchNormalization
from keras.layers import LSTM
from keras.models import load_model
from keras import backend as K
from keras.layers import Activation

def create_model(input_shape, filter_size):
	
	model = Sequential()
	
	model.add(Conv1D(filters = 256, kernel_size = filter_size, padding='same', input_shape = input_shape))
	model.add(BatchNormalization())
	model.add(Activation('relu'))

	model.add(Conv1D(filters = 256, kernel_size = filter_size, padding='same', input_shape = input_shape))
	model.add(BatchNormalization())
	model.add(Activation('relu'))
	
	model.add(Conv1D(1, kernel_size  = filter_size, padding='same'))
	
	return(model)
	
def train_model( model, inputs, labels, epochs, batch_size ):

	model.compile(optimizer = 'adam', loss='mean_squared_error')
	model.fit(	inputs, labels,
            	epochs = epochs,
                batch_size = batch_size,
                shuffle = False ) 
                #validation_data=(x_test_noisy, x_test)
                #tensorboard --logdir=/tmp/autoencoder
                #callbacks=[TensorBoard(log_dir='/tmp/tb', histogram_freq=0, write_graph=False)]    
	return(model)

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
	
	batches_frames_output_holder = []
	for i in xrange(0, input_frames.shape[0], batch_size):
		batch_input_frames = input_frames[ i:i+batch_size , :, : ]
		batch_output_frames = get_output( model, batch_input_frames )
		batch_output_frames_flattened = np.reshape(batch_output_frames, (batch_output_frames.shape[0], -1)) 
		batches_output_frames_holder.append(batch_output_frames_flattened)
		
	output_frames_concatenated = np.concatenate( batches_frames_encoded_holder, axis = 0 )
	
	return(output_frames_concatenated)