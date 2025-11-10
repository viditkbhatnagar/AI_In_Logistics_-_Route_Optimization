"""
Simulated Annealing for Vehicle Routing Problem
Fast implementation optimized for demo performance
"""

import numpy as np
import random
import math
from config import TRAFFIC_MULTIPLIERS, WEATHER_MULTIPLIERS, AVERAGE_SPEED_KMH


class SimulatedAnnealingVRP:
    """Simulated Annealing for VRP"""
    
    def __init__(self, distance_matrix, orders_df, delivery_locations, vehicle_info,
                 traffic_data=None, weather_data=None,
                 initial_temp=1000.0, cooling_rate=0.95, iterations=200):
        self.distance_matrix = distance_matrix
        self.orders_df = orders_df
        self.delivery_locations = delivery_locations
        self.vehicle_info = vehicle_info
        self.initial_temp = initial_temp
        self.cooling_rate = cooling_rate
        self.iterations = iterations
        self.n_deliveries = len(delivery_locations)
        
        # Traffic and weather multipliers
        self.traffic_multipliers_by_hour = {}
        if traffic_data is not None and len(traffic_data) > 0:
            for _, row in traffic_data.iterrows():
                hour = int(row['hour'])
                self.traffic_multipliers_by_hour[hour] = TRAFFIC_MULTIPLIERS.get(row['traffic_level'], 1.0)
        
        if weather_data is not None and len(weather_data) > 0:
            current_weather = weather_data.iloc[-1]['condition']
            self.weather_multiplier = WEATHER_MULTIPLIERS.get(current_weather, 1.0)
        else:
            self.weather_multiplier = 1.0
        
        self.history = {'best_fitness': [], 'current_fitness': [], 'temperature': []}
    
    def calculate_route_cost(self, route):
        """Calculate total cost for a route"""
        if len(route) == 0:
            return float('inf')
        
        total_distance = 0
        total_time = 0
        current_hour = 9
        
        for i in range(len(route) - 1):
            from_idx = route[i]
            to_idx = route[i + 1]
            distance = self.distance_matrix[from_idx][to_idx]
            
            traffic_mult = self.traffic_multipliers_by_hour.get(current_hour % 24, 1.0)
            adjusted_distance = distance * traffic_mult * self.weather_multiplier
            total_distance += adjusted_distance
            
            effective_speed = AVERAGE_SPEED_KMH / traffic_mult / self.weather_multiplier
            time_hours = adjusted_distance / effective_speed
            total_time += time_hours
            current_hour += int(time_hours)
        
        total_time += len(route) * 10 / 60
        
        fuel_cost = total_distance * self.vehicle_info['fuel_efficiency_l_per_km'] * 1.2
        driver_cost = total_time * 25.0
        maintenance_cost = total_distance * 0.15
        
        return fuel_cost + driver_cost + maintenance_cost
    
    def generate_neighbor(self, route):
        """Generate neighbor solution"""
        neighbor = route.copy()
        
        # Random mutation: swap two random positions
        if len(neighbor) > 1:
            i, j = random.sample(range(len(neighbor)), 2)
            neighbor[i], neighbor[j] = neighbor[j], neighbor[i]
        
        return neighbor
    
    def solve(self, initial_route=None, progress_callback=None):
        """Solve VRP using Simulated Annealing"""
        # Initialize with random or provided route
        if initial_route is None:
            current_route = list(range(self.n_deliveries))
            random.shuffle(current_route)
        else:
            current_route = initial_route.copy()
        
        current_cost = self.calculate_route_cost(current_route)
        best_route = current_route.copy()
        best_cost = current_cost
        
        temperature = self.initial_temp
        
        for iteration in range(self.iterations):
            # Generate neighbor
            neighbor_route = self.generate_neighbor(current_route)
            neighbor_cost = self.calculate_route_cost(neighbor_route)
            
            # Accept or reject
            delta = neighbor_cost - current_cost
            if delta < 0 or random.random() < math.exp(-delta / temperature):
                current_route = neighbor_route
                current_cost = neighbor_cost
                
                if current_cost < best_cost:
                    best_route = current_route.copy()
                    best_cost = current_cost
            
            # Cool down
            temperature *= self.cooling_rate
            
            # Record history
            self.history['best_fitness'].append(best_cost)
            self.history['current_fitness'].append(current_cost)
            self.history['temperature'].append(temperature)
            
            if progress_callback:
                progress_callback(iteration + 1, self.iterations, best_cost, current_cost)
        
        return best_route, self.history

