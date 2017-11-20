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
	
	# Encode image into down-sampled latent space
	#model.add(Conv1D(filters = 12, kernel_size = filter_size, activation='relu', padding='same', input_shape = input_shape))
	#model.add(BatchNormalization())
	
	#model.add(Conv1D(filters = 20, kernel_size = filter_size, activation='relu', padding='same'))
	#model.add(MaxPooling1D(pool_size = 4, padding='same'))
	
	#model.add(BatchNormalization())
	model.add(Conv1D(filters = 64, kernel_size = filter_size, padding='same', input_shape = input_shape))
	model.add(BatchNormalization())
	model.add(Activation('relu'))
	#model.add(MaxPooling1D(pool_size = 4, padding='same'))
	
	### 
	#model.add(BatchNormalization())
	#model.add(Conv1D(filters = 40, kernel_size = filter_size, activation='relu', padding='same'))
	###
	
	#model.add(BatchNormalization())
	#model.add(Conv1D(filters = 30, kernel_size = filter_size, activation='relu', padding='same'))
	#model.add(UpSampling1D(size = 4))
	
	#model.add(BatchNormalization())
	#model.add(Conv1D(filters = 20, kernel_size = filter_size, activation='relu', padding='same'))
	#model.add(UpSampling1D(size = 4))
	
	#model.add(BatchNormalization())
	#model.add(Conv1D(filters = 12, kernel_size = filter_size, activation='relu', padding='same'))

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
	
def get_intermediary_output( model, new_input ):
	get_output = K.function([model.layers[0].input, K.learning_phase()],
                                  	  [model.layers[ len(model.layers) - 1 ].output])
	layer_output = get_output([new_input, 0])[0]
	
	return(layer_output)