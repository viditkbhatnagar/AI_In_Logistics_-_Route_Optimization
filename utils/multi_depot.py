"""
Multi-depot and pickup-delivery optimization
"""

import numpy as np
import pandas as pd
from utils.route_optimizer import calculate_distance_matrix, calculate_route_cost


def optimize_multi_depot_routing(orders_df, vehicles_df, delivery_locations, depots):
    """Optimize routing with multiple depots"""
    # Calculate distance from each delivery to each depot
    depot_assignments = {}
    
    for _, order in orders_df.iterrows():
        delivery_id = order['delivery_id']
        delivery_loc = delivery_locations[delivery_locations['delivery_id'] == delivery_id]
        
        if len(delivery_loc) > 0:
            delivery_loc = delivery_loc.iloc[0]
            min_distance = float('inf')
            best_depot = None
            
            for _, depot in depots.iterrows():
                from geopy.distance import geodesic
                distance = geodesic(
                    (delivery_loc['latitude'], delivery_loc['longitude']),
                    (depot['latitude'], depot['longitude'])
                ).kilometers
                
                if distance < min_distance:
                    min_distance = distance
                    best_depot = depot['depot_id']
            
            if best_depot not in depot_assignments:
                depot_assignments[best_depot] = []
            depot_assignments[best_depot].append(delivery_id)
    
    return depot_assignments


def optimize_pickup_delivery(orders_df, delivery_locations, vehicles_df, depots):
    """Optimize routes with pickup and delivery pairs"""
    # Separate pickups and deliveries
    pickups = orders_df[orders_df['order_type'] == 'pickup'] if 'order_type' in orders_df.columns else pd.DataFrame()
    deliveries = orders_df[orders_df['order_type'] == 'delivery'] if 'order_type' in orders_df.columns else orders_df
    
    # Match pickup-delivery pairs
    pairs = []
    for _, delivery in deliveries.iterrows():
        # Find corresponding pickup (simplified - in real scenario, use order_id matching)
        if len(pickups) > 0:
            pickup = pickups.iloc[0]  # Simplified matching
            pairs.append({
                'pickup_id': pickup.get('delivery_id', ''),
                'delivery_id': delivery['delivery_id'],
                'priority': delivery['priority']
            })
        else:
            pairs.append({
                'pickup_id': None,
                'delivery_id': delivery['delivery_id'],
                'priority': delivery['priority']
            })
    
    # Sort by priority
    pairs.sort(key=lambda x: x['priority'])
    
    return pairs

