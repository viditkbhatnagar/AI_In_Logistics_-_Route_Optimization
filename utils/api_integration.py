"""
Mock API integration examples
Simulates integration with external services like Google Maps, HERE Maps
"""

import pandas as pd
import numpy as np
from geopy.distance import geodesic


class MockGoogleMapsAPI:
    """Mock Google Maps API integration"""
    
    def __init__(self, api_key=None):
        self.api_key = api_key or "mock_key"
        self.call_count = 0
    
    def get_directions(self, origin, destination, mode='driving'):
        """Get directions between two points"""
        self.call_count += 1
        
        # Simulate API call delay
        import time
        time.sleep(0.01)  # Simulate network delay
        
        # Calculate distance
        distance_km = geodesic(
            (origin['lat'], origin['lng']),
            (destination['lat'], destination['lng'])
        ).kilometers
        
        # Simulate route with traffic
        duration_minutes = (distance_km / 50.0) * 60  # Assume 50 km/h average
        
        return {
            'distance_km': distance_km,
            'duration_minutes': duration_minutes,
            'route': [
                {'lat': origin['lat'], 'lng': origin['lng']},
                {'lat': destination['lat'], 'lng': destination['lng']}
            ],
            'api_calls': self.call_count
        }
    
    def get_distance_matrix(self, origins, destinations):
        """Get distance matrix"""
        matrix = []
        for origin in origins:
            row = []
            for dest in destinations:
                result = self.get_directions(origin, dest)
                row.append(result['distance_km'])
            matrix.append(row)
        return matrix
    
    def get_geocoding(self, address):
        """Geocode an address"""
        self.call_count += 1
        # Mock geocoding - return random coordinates
        return {
            'lat': np.random.uniform(40.0, 41.0),
            'lng': np.random.uniform(-74.0, -73.0),
            'formatted_address': address
        }
    
    def get_eta(self, origin, destination, departure_time=None):
        """Get estimated time of arrival"""
        result = self.get_directions(origin, destination)
        return {
            'eta_minutes': result['duration_minutes'],
            'distance_km': result['distance_km']
        }


class MockHEREMapsAPI:
    """Mock HERE Maps API integration"""
    
    def __init__(self, api_key=None):
        self.api_key = api_key or "mock_key"
    
    def calculate_route(self, waypoints, vehicle_type='truck'):
        """Calculate route through waypoints"""
        total_distance = 0
        total_time = 0
        
        for i in range(len(waypoints) - 1):
            distance = geodesic(
                (waypoints[i]['lat'], waypoints[i]['lng']),
                (waypoints[i+1]['lat'], waypoints[i+1]['lng'])
            ).kilometers
            
            total_distance += distance
            total_time += (distance / 50.0) * 60  # minutes
        
        return {
            'total_distance_km': total_distance,
            'total_time_minutes': total_time,
            'waypoints': waypoints,
            'vehicle_type': vehicle_type
        }


def demonstrate_api_integration(delivery_locations, depots):
    """Demonstrate API integration"""
    google_api = MockGoogleMapsAPI()
    here_api = MockHEREMapsAPI()
    
    depot = depots.iloc[0]
    origin = {'lat': depot['latitude'], 'lng': depot['longitude']}
    
    results = {
        'google_maps': [],
        'here_maps': []
    }
    
    # Test with first 5 deliveries
    for idx in range(min(5, len(delivery_locations))):
        loc = delivery_locations.iloc[idx]
        destination = {'lat': loc['latitude'], 'lng': loc['longitude']}
        
        # Google Maps
        google_result = google_api.get_directions(origin, destination)
        results['google_maps'].append({
            'delivery_id': loc['delivery_id'],
            'distance_km': google_result['distance_km'],
            'duration_minutes': google_result['duration_minutes']
        })
        
        # HERE Maps
        waypoints = [origin, destination]
        here_result = here_api.calculate_route(waypoints)
        results['here_maps'].append({
            'delivery_id': loc['delivery_id'],
            'distance_km': here_result['total_distance_km'],
            'duration_minutes': here_result['total_time_minutes']
        })
    
    return results, google_api.call_count

