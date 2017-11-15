import numpy as np
import audio_preprocessing as ap
import dca_model as dcam
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier

def find_K_exemplars( training_set_encoded_flatted, cluster_model ):
	K_exemplar_indices = np.argmin( cluster_model.transform(training_set_encoded_flatted), axis = 0  )
	return(K_exemplar_indices)
			
def get_corresponding_times_series_indices( new_data, cluster_model, K_exemplar_indices ):
	cluster_indices = cluster_model.predict(new_data)
	time_series_indices = K_exemplar_indices[ cluster_indices ]
	return(time_series_indices)

def encode_and_flatten(model, frames_to_encode, batch_size = 100):
	
	batches_frames_encoded_holder = []
	for i in xrange(0, frames_to_encode.shape[0], batch_size):
		batch_frames_to_encode = frames_to_encode[ i:i+batch_size , :, : ]
		batch_frames_encoded = dcam.get_intermediary_output( model, batch_frames_to_encode )
		batch_frames_encoded_flattened = np.reshape(batch_frames_encoded, (batch_frames_encoded.shape[0], -1)) 
		batches_frames_encoded_holder.append(batch_frames_encoded_flattened)
		
	frames_encoded_flattened = np.concatenate( batches_frames_encoded_holder, axis = 0 )
	
	return(frames_encoded_flattened)
	
def create_kmeans_model( K, training_frames_encoded_flattened ):

	cluster_model = KMeans(n_clusters = K, verbose = 1, n_init = 3).fit(training_frames_encoded_flattened)
	
	return(cluster_model)

def match_routine(cluster_model, K_exemplar_indices, test_frames_encoded_flattened):
	
	test_set_prediction_indices = get_corresponding_times_series_indices( test_frames_encoded_flattened, cluster_model, K_exemplar_indices )

	return(test_set_prediction_indices)
	

def KNN_routine( training_frames_encoded_flattened, test_frames_encoded_flattened, n_jobs = 1):
	KNN_model = KNeighborsClassifier(n_neighbors = 1, n_jobs = n_jobs).fit( X = training_frames_encoded_flattened, y = np.zeros((training_frames_encoded_flattened.shape[0], )) )
	test_set_prediction_indices = KNN_model.kneighbors( X = test_frames_encoded_flattened, return_distance = False )
	return(test_set_prediction_indices)
	
