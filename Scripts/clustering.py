import numpy as np
import audio_preprocessing as ap
import dca_model as dcam
from sklearn.cluster import KMeans

def find_K_exemplars( training_set_encoded_flatted, cluster_model ):
	K_exemplar_indices = np.argmin( cluster_model.transform(training_set_encoded_flatted), axis = 0  )
	return(K_exemplar_indices)
			
def get_corresponding_times_series_indices( new_data, cluster_model, K_exemplar_indices ):
	cluster_indices = cluster_model.predict(new_data)
	time_series_indices = K_exemplar_indices[ cluster_indices ]
	return(time_series_indices)
	
def cluster_and_match_routine(K, model, training_set_scaled, test_set_scaled):
	
	training_set_encoded = dcam.get_intermediary_output( model, training_set_scaled )
	training_set_encoded_flattened = np.reshape(training_set_encoded, (training_set_encoded.shape[0], -1))

	cluster_model = KMeans(n_clusters = K).fit(training_set_encoded_flattened)
	K_exemplar_indices = find_K_exemplars(training_set_encoded_flattened, cluster_model)

	test_set_encoded = dcam.get_intermediary_output( model, test_set_scaled )
	test_set_encoded_flattened = np.reshape(test_set_encoded, (test_set_encoded.shape[0], -1))
	test_set_prediction_indices = get_corresponding_times_series_indices( test_set_encoded_flattened, cluster_model, K_exemplar_indices )

	return(test_set_prediction_indices)
