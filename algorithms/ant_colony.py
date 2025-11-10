"""
Ant Colony Optimization (ACO) for Vehicle Routing Problem
Fast implementation optimized for demo performance
"""

import numpy as np
import random
from config import TRAFFIC_MULTIPLIERS, WEATHER_MULTIPLIERS, AVERAGE_SPEED_KMH


class AntColonyVRP:
    """Ant Colony Optimization for VRP"""
    
    def __init__(self, distance_matrix, orders_df, delivery_locations, vehicle_info,
                 traffic_data=None, weather_data=None,
                 n_ants=30, alpha=1.0, beta=2.0, evaporation=0.1, iterations=50):
        self.distance_matrix = distance_matrix
        self.orders_df = orders_df
        self.delivery_locations = delivery_locations
        self.vehicle_info = vehicle_info
        self.n_ants = n_ants
        self.alpha = alpha  # Pheromone importance
        self.beta = beta    # Heuristic importance
        self.evaporation = evaporation
        self.iterations = iterations
        self.n_deliveries = len(delivery_locations)
        
        # Initialize pheromone matrix
        self.pheromone = np.ones((self.n_deliveries, self.n_deliveries))
        
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
        
        self.history = {'best_fitness': [], 'avg_fitness': []}
        self.best_route = None
        self.best_cost = float('inf')
    
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
            
            # Apply traffic multiplier
            traffic_mult = self.traffic_multipliers_by_hour.get(current_hour % 24, 1.0)
            adjusted_distance = distance * traffic_mult * self.weather_multiplier
            total_distance += adjusted_distance
            
            # Calculate time
            effective_speed = AVERAGE_SPEED_KMH / traffic_mult / self.weather_multiplier
            time_hours = adjusted_distance / effective_speed
            total_time += time_hours
            current_hour += int(time_hours)
        
        # Add delivery time
        total_time += len(route) * 10 / 60
        
        # Calculate cost
        fuel_cost = total_distance * self.vehicle_info['fuel_efficiency_l_per_km'] * 1.2
        driver_cost = total_time * 25.0
        maintenance_cost = total_distance * 0.15
        
        return fuel_cost + driver_cost + maintenance_cost
    
    def construct_solution(self):
        """Construct solution using ant colony logic"""
        routes = []
        costs = []
        
        for ant in range(self.n_ants):
            route = []
            unvisited = list(range(self.n_deliveries))
            current = random.choice(unvisited)
            route.append(current)
            unvisited.remove(current)
            
            while unvisited:
                # Calculate probabilities for next node
                probabilities = []
                for node in unvisited:
                    pheromone = self.pheromone[current][node] ** self.alpha
                    heuristic = (1.0 / (self.distance_matrix[current][node] + 0.1)) ** self.beta
                    probabilities.append(pheromone * heuristic)
                
                # Normalize probabilities
                total = sum(probabilities)
                if total > 0:
                    probabilities = [p / total for p in probabilities]
                else:
                    probabilities = [1.0 / len(unvisited)] * len(unvisited)
                
                # Select next node
                next_node = np.random.choice(unvisited, p=probabilities)
                route.append(next_node)
                unvisited.remove(next_node)
                current = next_node
            
            cost = self.calculate_route_cost(route)
            routes.append(route)
            costs.append(cost)
        
        return routes, costs
    
    def update_pheromone(self, routes, costs):
        """Update pheromone matrix"""
        # Evaporation
        self.pheromone *= (1 - self.evaporation)
        
        # Add pheromone from all ants
        for route, cost in zip(routes, costs):
            if cost > 0:
                pheromone_deposit = 1.0 / cost
                for i in range(len(route) - 1):
                    self.pheromone[route[i]][route[i + 1]] += pheromone_deposit
    
    def solve(self, progress_callback=None):
        """Solve VRP using ACO"""
        for iteration in range(self.iterations):
            routes, costs = self.construct_solution()
            
            # Update best solution
            best_idx = np.argmin(costs)
            if costs[best_idx] < self.best_cost:
                self.best_cost = costs[best_idx]
                self.best_route = routes[best_idx].copy()
            
            # Update pheromone
            self.update_pheromone(routes, costs)
            
            # Record history
            self.history['best_fitness'].append(self.best_cost)
            self.history['avg_fitness'].append(np.mean(costs))
            
            if progress_callback:
                progress_callback(iteration + 1, self.iterations, self.best_cost, np.mean(costs))
        
        return self.best_route, self.history

