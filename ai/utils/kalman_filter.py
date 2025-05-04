# utils/kalman_filter.py
def apply_kalman_filtering(current_confidences, kalman_filters):
    """
    Apply Kalman filtering to smooth confidence values
    """
    smoothed_confidences = {}
    
    for gesture, confidence in current_confidences.items():
        # Get current Kalman filter state
        kalman = kalman_filters[gesture]
        
        # Prediction step
        prediction = kalman['estimate']
        prediction_error = kalman['error'] + kalman['process_noise']
        
        # Update step
        kalman_gain = prediction_error / (prediction_error + kalman['measurement_noise'])
        estimate = prediction + kalman_gain * (confidence - prediction)
        error = (1 - kalman_gain) * prediction_error
        
        # Update filter state
        kalman['estimate'] = estimate
        kalman['error'] = error
        
        # Store smoothed confidence
        smoothed_confidences[gesture] = estimate
    
    return smoothed_confidences