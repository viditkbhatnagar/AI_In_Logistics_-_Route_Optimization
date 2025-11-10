"""
Real-time data simulation utilities for dynamic routing
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
from config import TRAFFIC_MULTIPLIERS, WEATHER_MULTIPLIERS


def simulate_realtime_traffic_update(base_traffic_df, hour, incident_probability=0.1):
    """Simulate a real-time traffic update (e.g., accident, congestion)"""
    traffic_update = base_traffic_df.copy()
    
    # Check if incident occurs
    if random.random() < incident_probability:
        # Create traffic incident
        affected_hours = [hour, (hour + 1) % 24]
        for h in affected_hours:
            mask = traffic_update['hour'] == h
            if mask.any():
                idx = traffic_update[mask].index[0]
                # Increase congestion
                new_congestion = min(1.0, traffic_update.loc[idx, 'congestion_factor'] + 0.3)
                traffic_update.loc[idx, 'congestion_factor'] = new_congestion
                
                # Update traffic level
                if new_congestion > 0.8:
                    traffic_update.loc[idx, 'traffic_level'] = 'severe'
                elif new_congestion > 0.6:
                    traffic_update.loc[idx, 'traffic_level'] = 'heavy'
                else:
                    traffic_update.loc[idx, 'traffic_level'] = 'moderate'
                
                # Update speed multiplier
                traffic_update.loc[idx, 'speed_multiplier'] = 1 - new_congestion * 0.6
                traffic_update.loc[idx, 'average_speed_kmh'] = 50 * traffic_update.loc[idx, 'speed_multiplier']
    
    return traffic_update


def simulate_realtime_weather_update(base_weather_df, change_probability=0.2):
    """Simulate a real-time weather change (e.g., sudden rain, storm)"""
    weather_update = base_weather_df.copy()
    
    # Check if weather changes
    if random.random() < change_probability:
        # Get current weather
        current = weather_update.iloc[-1].copy()
        
        # Simulate weather change
        weather_conditions = ['clear', 'cloudy', 'rain', 'snow', 'storm']
        current_idx = weather_conditions.index(current['condition'])
        
        # Weather can get worse or better
        if random.random() < 0.7:  # 70% chance it gets worse
            new_idx = min(len(weather_conditions) - 1, current_idx + random.randint(1, 2))
        else:  # 30% chance it gets better
            new_idx = max(0, current_idx - 1)
        
        new_condition = weather_conditions[new_idx]
        
        # Update weather
        weather_update.iloc[-1, weather_update.columns.get_loc('condition')] = new_condition
        
        if new_condition in ['rain', 'snow', 'storm']:
            weather_update.iloc[-1, weather_update.columns.get_loc('precipitation_mm')] = \
                np.random.uniform(5, 20) if new_condition == 'rain' else \
                np.random.uniform(10, 30) if new_condition == 'snow' else \
                np.random.uniform(15, 40)
        
        if new_condition == 'storm':
            weather_update.iloc[-1, weather_update.columns.get_loc('wind_speed_kmh')] = \
                np.random.uniform(40, 60)
    
    return weather_update


def simulate_demand_surge(base_demand_df, surge_probability=0.15, surge_multiplier=1.5):
    """Simulate a sudden demand surge (e.g., flash sale, event)"""
    demand_update = base_demand_df.copy()
    
    # Check if surge occurs
    if random.random() < surge_probability:
        # Find today's forecast
        today = datetime.now().date()
        today_mask = demand_update['date'].dt.date == today
        
        if today_mask.any():
            idx = demand_update[today_mask].index[0]
            original_demand = demand_update.loc[idx, 'forecasted_orders']
            new_demand = int(original_demand * surge_multiplier)
            demand_update.loc[idx, 'forecasted_orders'] = new_demand
            # Reduce confidence due to unexpected surge
            demand_update.loc[idx, 'confidence_level'] = max(0.5, demand_update.loc[idx, 'confidence_level'] - 0.2)
    
    return demand_update


def calculate_route_with_realtime_conditions(route, distance_matrix, vehicle_info, orders_df, 
                                            delivery_locations, traffic_data, weather_data,
                                            current_hour=9):
    """Calculate route metrics considering real-time conditions"""
    from utils.route_optimizer import calculate_route_cost
    
    # Get current traffic condition
    current_traffic = traffic_data[traffic_data['hour'] == current_hour]
    if len(current_traffic) > 0:
        traffic_level = current_traffic.iloc[0]['traffic_level']
        traffic_mult = TRAFFIC_MULTIPLIERS.get(traffic_level, 1.0)
    else:
        traffic_mult = 1.0
    
    # Get current weather
    if len(weather_data) > 0:
        weather_condition = weather_data.iloc[-1]['condition']
        weather_mult = WEATHER_MULTIPLIERS.get(weather_condition, 1.0)
    else:
        weather_mult = 1.0
    
    # Calculate route cost with real-time multipliers
    cost_info = calculate_route_cost(
        route, distance_matrix, vehicle_info, orders_df,
        traffic_mult, weather_mult
    )
    
    return cost_info, traffic_mult, weather_mult

