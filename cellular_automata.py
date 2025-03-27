import numpy as np
import logging

logger = logging.getLogger(__name__)

def predict_spread(current_state, days):
    """
    Predict the spread of Ganoderma infection using a Cellular Automata model
    
    Args:
        current_state: List of dictionaries with current infection data (lat, lng, level)
        days: Number of days to predict into the future
    
    Returns:
        Dictionary with prediction data for visualization
    """
    logger.info(f"Predicting spread for {days} days based on {len(current_state)} infection points")
    
    # In a real implementation, this would:
    # 1. Convert the infection points into a grid
    # 2. Define the cellular automata rules
    # 3. Run the simulation for the specified number of days
    # 4. Return the resulting infection spread
    
    # For this implementation, we'll simulate the spread by:
    # - Keeping all original points
    # - Adding new points around existing ones
    # - Increasing infection levels over time
    
    # Copy the current state
    prediction = current_state.copy()
    
    # Define the maximum number of new points to add in total
    max_new_points = min(100, len(current_state) * days // 10)
    
    # Add new points based on the number of days
    new_points_count = min(max_new_points, int(len(current_state) * days * 0.1))
    
    for _ in range(new_points_count):
        # Select a random existing point as the source of spread
        source_point = prediction[np.random.randint(0, len(prediction))]
        
        # Create a new point nearby
        spread_distance = 0.005 * np.sqrt(days)  # Further spread over time
        lat_offset = (np.random.random() - 0.5) * spread_distance
        lng_offset = (np.random.random() - 0.5) * spread_distance
        
        # New infection level is based on source but decreases with distance
        distance = np.sqrt(lat_offset**2 + lng_offset**2)
        new_level = max(0.1, source_point['level'] * (1 - distance/spread_distance))
        
        # Add the new point
        new_point = {
            'lat': source_point['lat'] + lat_offset,
            'lng': source_point['lng'] + lng_offset,
            'level': new_level
        }
        prediction.append(new_point)
    
    # Increase infection levels of existing points over time
    for point in prediction:
        # Increase by up to 20% based on days, but cap at 1.0
        point['level'] = min(1.0, point['level'] * (1 + days * 0.005))
    
    # Create timeframes for visualization
    timeframes = []
    for day in range(0, days + 1, max(1, days // 5)):  # 5 timeframes
        if day == 0:
            # Initial state
            timeframes.append({
                'day': day,
                'points': current_state.copy()
            })
        else:
            # Calculate intermediate state
            ratio = day / days
            points_at_timeframe = []
            
            # Include original points with increased infection
            for point in current_state:
                p = point.copy()
                p['level'] = min(1.0, p['level'] * (1 + day * 0.005))
                points_at_timeframe.append(p)
            
            # Add new points proportional to timeframe
            new_points_at_timeframe = int(new_points_count * ratio)
            for i in range(new_points_at_timeframe):
                if len(prediction) > len(current_state) + i:
                    points_at_timeframe.append(prediction[len(current_state) + i])
            
            timeframes.append({
                'day': day,
                'points': points_at_timeframe
            })
    
    return {
        'initial_state': current_state,
        'final_state': prediction,
        'timeframes': timeframes
    }
