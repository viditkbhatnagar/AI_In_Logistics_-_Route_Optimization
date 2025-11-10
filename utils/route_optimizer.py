"""
Core route optimization utilities and cost calculation functions
"""

import numpy as np
import pandas as pd
from geopy.distance import geodesic
from config import (
    FUEL_COST_PER_LITER, FUEL_CONSUMPTION_PER_KM, DRIVER_COST_PER_HOUR,
    VEHICLE_MAINTENANCE_PER_KM, AVERAGE_SPEED_KMH, TRAFFIC_MULTIPLIERS,
    WEATHER_MULTIPLIERS
)


def calculate_distance(lat1, lon1, lat2, lon2):
    """Calculate distance between two coordinates in kilometers"""
    return geodesic((lat1, lon1), (lat2, lon2)).kilometers


def calculate_distance_matrix(locations_df):
    """Calculate distance matrix for all locations"""
    n = len(locations_df)
    distance_matrix = np.zeros((n, n))
    
    for i in range(n):
        for j in range(n):
            if i != j:
                loc1 = locations_df.iloc[i]
                loc2 = locations_df.iloc[j]
                distance_matrix[i][j] = calculate_distance(
                    loc1['latitude'], loc1['longitude'],
                    loc2['latitude'], loc2['longitude']
                )
    
    return distance_matrix


def calculate_route_cost(route, distance_matrix, vehicle_info, orders_df, 
                         traffic_multiplier=1.0, weather_multiplier=1.0):
    """
    Calculate total cost for a route
    
    Parameters:
    - route: list of delivery indices
    - distance_matrix: distance matrix
    - vehicle_info: dict with vehicle details
    - orders_df: DataFrame with order information
    - traffic_multiplier: multiplier for traffic conditions
    - weather_multiplier: multiplier for weather conditions
    """
    if len(route) == 0:
        return 0
    
    total_distance = 0
    total_time = 0
    
    # Calculate distance and time for route
    for i in range(len(route) - 1):
        from_idx = route[i]
        to_idx = route[i + 1]
        distance = distance_matrix[from_idx][to_idx]
        
        # Apply traffic and weather multipliers
        adjusted_distance = distance * traffic_multiplier * weather_multiplier
        total_distance += adjusted_distance
        
        # Calculate time (distance / speed)
        time_hours = adjusted_distance / (AVERAGE_SPEED_KMH * traffic_multiplier)
        total_time += time_hours
    
    # Add delivery time for each stop
    delivery_time_hours = len(route) * 10 / 60  # 10 minutes per delivery
    total_time += delivery_time_hours
    
    # Calculate costs
    fuel_cost = total_distance * vehicle_info['fuel_efficiency_l_per_km'] * FUEL_COST_PER_LITER
    driver_cost = total_time * DRIVER_COST_PER_HOUR
    maintenance_cost = total_distance * VEHICLE_MAINTENANCE_PER_KM
    vehicle_cost = total_distance * vehicle_info.get('cost_per_km', 1.0)
    
    total_cost = fuel_cost + driver_cost + maintenance_cost + vehicle_cost
    
    return {
        'total_cost': total_cost,
        'total_distance': total_distance,
        'total_time': total_time,
        'fuel_cost': fuel_cost,
        'driver_cost': driver_cost,
        'maintenance_cost': maintenance_cost,
        'vehicle_cost': vehicle_cost,
        'n_deliveries': len(route)
    }


def generate_baseline_route(delivery_locations, depot_location, orders_df):
    """Generate a simple baseline route (nearest neighbor heuristic)"""
    # Reset index to ensure we have 0-based sequential indices
    delivery_locations_reset = delivery_locations.reset_index(drop=True)
    
    # Calculate distances from depot
    distances_from_depot = []
    for idx in range(len(delivery_locations_reset)):
        loc = delivery_locations_reset.iloc[idx]
        dist = calculate_distance(
            depot_location['latitude'], depot_location['longitude'],
            loc['latitude'], loc['longitude']
        )
        distances_from_depot.append((idx, dist))
    
    # Sort by distance and create route (using positional indices 0, 1, 2, ...)
    distances_from_depot.sort(key=lambda x: x[1])
    route = [idx for idx, _ in distances_from_depot]
    
    return route


def calculate_route_metrics(route, distance_matrix, vehicle_info, orders_df, delivery_locations,
                           depot_idx, traffic_data=None, weather_data=None, depot_location=None):
    """Calculate comprehensive metrics for a route"""
    if len(route) == 0:
        return {}
    
    # Get traffic and weather multipliers
    traffic_mult = 1.0
    weather_mult = 1.0
    
    if traffic_data is not None and len(traffic_data) > 0:
        # Use average traffic level
        avg_traffic = traffic_data['traffic_level'].mode()[0] if len(traffic_data) > 0 else 'light'
        traffic_mult = TRAFFIC_MULTIPLIERS.get(avg_traffic, 1.0)
    
    if weather_data is not None and len(weather_data) > 0:
        # Use most recent weather condition
        latest_weather = weather_data.iloc[-1]['condition'] if len(weather_data) > 0 else 'clear'
        weather_mult = WEATHER_MULTIPLIERS.get(latest_weather, 1.0)
    
    # Calculate route cost
    cost_info = calculate_route_cost(
        route, distance_matrix, vehicle_info, orders_df,
        traffic_mult, weather_mult
    )
    
    # Calculate SLA compliance
    sla_violations = 0
    current_time = 9.0  # Start at 9 AM (typical start time)
    
    # Calculate travel time considering traffic and weather
    base_speed = AVERAGE_SPEED_KMH / traffic_mult / weather_mult
    
    for i, delivery_idx in enumerate(route):
        # Calculate travel time to this delivery
        if i == 0:
            # First delivery: travel from depot
            if depot_location is not None:
                # Calculate distance from depot to first delivery
                first_loc = delivery_locations.iloc[delivery_idx]
                travel_distance = calculate_distance(
                    depot_location['latitude'], depot_location['longitude'],
                    first_loc['latitude'], first_loc['longitude']
                )
            else:
                # Fallback: use average distance from first delivery to others
                if len(distance_matrix) > delivery_idx and len(distance_matrix[delivery_idx]) > 0:
                    # Use minimum non-zero distance as approximation
                    distances_from_first = [d for d in distance_matrix[delivery_idx] if d > 0]
                    travel_distance = min(distances_from_first) if distances_from_first else 5.0  # Default 5km
                else:
                    travel_distance = 5.0  # Default 5km
        else:
            # Travel from previous delivery
            prev_idx = route[i-1]
            if prev_idx < len(distance_matrix) and delivery_idx < len(distance_matrix[prev_idx]):
                travel_distance = distance_matrix[prev_idx][delivery_idx]
            else:
                travel_distance = 0
        
        # Travel time in hours
        travel_time = travel_distance / base_speed if base_speed > 0 else 0
        current_time += travel_time
        
        # Check SLA compliance
        order = orders_df[orders_df['delivery_id'] == delivery_locations.iloc[delivery_idx]['delivery_id']]
        if len(order) > 0:
            order = order.iloc[0]
            # Check if delivery is within time window
            time_window_start = order.get('time_window_start', 0)
            time_window_end = order.get('time_window_end', 24)
            
            # Only count as violation if outside window (with small buffer for rounding)
            if current_time < time_window_start - 0.1 or current_time > time_window_end + 0.1:
                sla_violations += 1
        
        # Add delivery time (10 minutes = 10/60 hours)
        current_time += 10 / 60
    
    # Calculate utilization
    total_weight = orders_df.loc[orders_df['delivery_id'].isin(
        [delivery_locations.iloc[i]['delivery_id'] for i in route]
    ), 'weight_kg'].sum()
    
    total_volume = orders_df.loc[orders_df['delivery_id'].isin(
        [delivery_locations.iloc[i]['delivery_id'] for i in route]
    ), 'volume_m3'].sum()
    
    weight_utilization = total_weight / vehicle_info['capacity_kg'] if vehicle_info['capacity_kg'] > 0 else 0
    volume_utilization = total_volume / vehicle_info['capacity_vol'] if vehicle_info['capacity_vol'] > 0 else 0
    
    metrics = {
        **cost_info,
        'sla_violations': sla_violations,
        'sla_compliance_rate': 1 - (sla_violations / len(route)) if len(route) > 0 else 1.0,
        'weight_utilization': min(weight_utilization, 1.0),
        'volume_utilization': min(volume_utilization, 1.0),
        'deliveries_per_hour': len(route) / cost_info['total_time'] if cost_info['total_time'] > 0 else 0,
        'traffic_multiplier': traffic_mult,
        'weather_multiplier': weather_mult
    }
    
    return metrics


def optimize_route_assignment(orders_df, vehicles_df, delivery_locations, distance_matrix):
    """Assign orders to vehicles based on capacity and proximity"""
    assignments = {}
    remaining_orders = orders_df.copy().reset_index(drop=True)
    
    # Create a mapping from delivery_id to location coordinates for faster lookup
    delivery_id_to_location = {}
    for idx, loc in delivery_locations.iterrows():
        delivery_id_to_location[loc['delivery_id']] = {
            'latitude': loc['latitude'],
            'longitude': loc['longitude']
        }
    
    for _, vehicle in vehicles_df.iterrows():
        vehicle_orders = []
        vehicle_weight = 0
        vehicle_volume = 0
        
        if len(remaining_orders) == 0:
            assignments[vehicle['vehicle_id']] = []
            continue
        
        # Calculate distances to depot for remaining orders
        distances = []
        for _, order_row in remaining_orders.iterrows():
            delivery_id = order_row['delivery_id']
            if delivery_id in delivery_id_to_location:
                loc_info = delivery_id_to_location[delivery_id]
                dist = calculate_distance(
                    vehicle['depot_latitude'], vehicle['depot_longitude'],
                    loc_info['latitude'], loc_info['longitude']
                )
            else:
                dist = float('inf')  # Very large distance if location not found
            distances.append(dist)
        
        # Add distance column
        remaining_orders = remaining_orders.copy()
        remaining_orders['distance_to_depot'] = distances
        
        # Sort by priority and distance
        remaining_orders = remaining_orders.sort_values(
            by=['priority', 'distance_to_depot'],
            ascending=[False, True]
        ).reset_index(drop=True)
        
        # Assign orders to vehicle based on capacity
        orders_to_remove = []
        for idx, order in remaining_orders.iterrows():
            if (vehicle_weight + order['weight_kg'] <= vehicle['capacity_kg'] and
                vehicle_volume + order['volume_m3'] <= vehicle['capacity_vol']):
                
                vehicle_orders.append(order['delivery_id'])
                vehicle_weight += order['weight_kg']
                vehicle_volume += order['volume_m3']
                orders_to_remove.append(idx)
        
        assignments[vehicle['vehicle_id']] = vehicle_orders
        
        # Remove assigned orders
        if len(orders_to_remove) > 0:
            remaining_orders = remaining_orders.drop(index=orders_to_remove).reset_index(drop=True)
    
    return assignments

