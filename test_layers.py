from testing_environment import get_steering_vector, add_steering_vectors_hook, test_steering, model, tokenizer
import random


LAYER_RANGE = 34
COEF_RANGE = 15

def grid_search(num_iterations, layer_range, coeff_range):
    best_scores = {} # key: score, value: hyperparameters 
    # track: 
    # score: key float value 
    # hyperparameters: tuple (layer, coefficient, response_id?)
    # responses corresponding to hyperparameter pairing: store in json file: result_id
    
    for l in range(layer_range):
        steering_vec = get_steering_vector(model, tokenizer, layer_idx=l)
        for c in range(1, coeff_range, 2):
            avg_post, post_responses, result_id = test_steering()
            
            best_scores[avg_post] = (l, c, result_id)
            
            
    
    
    
    # priority queue
    
