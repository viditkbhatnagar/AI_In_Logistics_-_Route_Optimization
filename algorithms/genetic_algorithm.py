"""
Genetic Algorithm implementation for Vehicle Routing Problem (VRP)
"""

import numpy as np
import random
from copy import deepcopy
from config import GA_POPULATION_SIZE, GA_MUTATION_RATE, GA_CROSSOVER_RATE, GA_GENERATIONS


class GeneticAlgorithmVRP:
    """Genetic Algorithm for solving Vehicle Routing Problem"""
    
    def __init__(self, distance_matrix, orders_df, delivery_locations, vehicle_info,
                 traffic_data=None, weather_data=None,
                 population_size=GA_POPULATION_SIZE, mutation_rate=GA_MUTATION_RATE,
                 crossover_rate=GA_CROSSOVER_RATE, generations=GA_GENERATIONS):
        self.distance_matrix = distance_matrix
        self.orders_df = orders_df
        self.delivery_locations = delivery_locations
        self.vehicle_info = vehicle_info
        self.traffic_data = traffic_data
        self.weather_data = weather_data
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.generations = generations
        self.n_deliveries = len(delivery_locations)
        self.history = {'best_fitness': [], 'avg_fitness': []}
        
        # Pre-process traffic multipliers by hour
        self.traffic_multipliers_by_hour = {}
        if traffic_data is not None and len(traffic_data) > 0:
            for _, row in traffic_data.iterrows():
                hour = int(row['hour'])
                from config import TRAFFIC_MULTIPLIERS
                self.traffic_multipliers_by_hour[hour] = TRAFFIC_MULTIPLIERS.get(row['traffic_level'], 1.0)
        
        # Get weather multiplier
        from config import WEATHER_MULTIPLIERS
        if weather_data is not None and len(weather_data) > 0:
            current_weather = weather_data.iloc[-1]['condition']
            self.weather_multiplier = WEATHER_MULTIPLIERS.get(current_weather, 1.0)
        else:
            self.weather_multiplier = 1.0
    
    def create_individual(self):
        """Create a random individual (route)"""
        individual = list(range(self.n_deliveries))
        random.shuffle(individual)
        return individual
    
    def calculate_fitness(self, individual):
        """
        Calculate fitness of an individual (lower is better)
        Fitness = total cost (distance + time + penalties)
        Now considers traffic patterns by hour and weather conditions
        """
        if len(individual) == 0:
            return float('inf')
        
        total_distance = 0
        total_time = 0
        current_hour = 9  # Start at 9 AM
        
        # Calculate total distance and time with traffic/weather considerations
        for i in range(len(individual) - 1):
            from_idx = individual[i]
            to_idx = individual[i + 1]
            segment_distance = self.distance_matrix[from_idx][to_idx]
            
            # Get traffic multiplier for current hour
            traffic_mult = self.traffic_multipliers_by_hour.get(current_hour % 24, 1.0)
            
            # Apply traffic and weather multipliers to distance
            adjusted_distance = segment_distance * traffic_mult * self.weather_multiplier
            total_distance += adjusted_distance
            
            # Calculate time considering traffic (slower speed during peak hours)
            base_speed = 50.0  # km/h
            effective_speed = base_speed / traffic_mult / self.weather_multiplier
            segment_time = adjusted_distance / effective_speed
            total_time += segment_time
            
            # Update current hour
            current_hour += int(segment_time)
        
        # Add return to depot distance
        if len(individual) > 0:
            return_distance = self.distance_matrix[individual[-1]][individual[0]]
            traffic_mult = self.traffic_multipliers_by_hour.get(current_hour % 24, 1.0)
            adjusted_return = return_distance * traffic_mult * self.weather_multiplier
            total_distance += adjusted_return
            effective_speed = 50.0 / traffic_mult / self.weather_multiplier
            total_time += adjusted_return / effective_speed
        
        # Add delivery time for each stop
        total_time += len(individual) * 10 / 60  # 10 minutes per delivery
        
        # Calculate cost with adjusted distance and time
        fuel_cost = total_distance * self.vehicle_info['fuel_efficiency_l_per_km'] * 1.2
        driver_cost = total_time * 25.0
        maintenance_cost = total_distance * 0.15
        
        total_cost = fuel_cost + driver_cost + maintenance_cost
        
        # Add penalty for capacity violations
        total_weight = 0
        total_volume = 0
        for idx in individual:
            order = self.orders_df[self.orders_df['delivery_id'] == 
                                  self.delivery_locations.iloc[idx]['delivery_id']]
            if len(order) > 0:
                total_weight += order.iloc[0]['weight_kg']
                total_volume += order.iloc[0]['volume_m3']
        
        capacity_penalty = 0
        if total_weight > self.vehicle_info['capacity_kg']:
            capacity_penalty += 1000 * (total_weight - self.vehicle_info['capacity_kg'])
        if total_volume > self.vehicle_info['capacity_vol']:
            capacity_penalty += 1000 * (total_volume - self.vehicle_info['capacity_vol'])
        
        # Add penalty for SLA violations (considering actual delivery time)
        sla_penalty = 0
        delivery_time = 9  # Start at 9 AM
        for route_idx, delivery_idx in enumerate(individual):
            order = self.orders_df[self.orders_df['delivery_id'] == 
                                  self.delivery_locations.iloc[delivery_idx]['delivery_id']]
            if len(order) > 0:
                order = order.iloc[0]
                # Check if delivery is within time window
                if delivery_time < order['time_window_start'] or delivery_time > order['time_window_end']:
                    sla_penalty += 50
                # Update delivery time for next stop
                if route_idx < len(individual) - 1:
                    next_delivery_idx = individual[route_idx + 1]
                    segment_dist = self.distance_matrix[delivery_idx][next_delivery_idx]
                    traffic_mult = self.traffic_multipliers_by_hour.get(int(delivery_time) % 24, 1.0)
                    segment_time = (segment_dist * traffic_mult * self.weather_multiplier) / (50.0 / traffic_mult / self.weather_multiplier)
                    delivery_time += segment_time + 10/60
        
        fitness = total_cost + capacity_penalty + sla_penalty
        return fitness
    
    def crossover(self, parent1, parent2):
        """Order crossover (OX) for TSP/VRP"""
        if random.random() > self.crossover_rate:
            return parent1.copy(), parent2.copy()
        
        size = len(parent1)
        start, end = sorted(random.sample(range(size), 2))
        
        # Create offspring1
        child1 = [-1] * size
        child1[start:end] = parent1[start:end]
        
        remaining = [item for item in parent2 if item not in child1[start:end]]
        idx = 0
        for i in range(size):
            if child1[i] == -1:
                child1[i] = remaining[idx]
                idx += 1
        
        # Create offspring2
        child2 = [-1] * size
        child2[start:end] = parent2[start:end]
        
        remaining = [item for item in parent1 if item not in child2[start:end]]
        idx = 0
        for i in range(size):
            if child2[i] == -1:
                child2[i] = remaining[idx]
                idx += 1
        
        return child1, child2
    
    def mutate(self, individual):
        """Swap mutation"""
        if random.random() > self.mutation_rate:
            return individual
        
        mutated = individual.copy()
        i, j = random.sample(range(len(mutated)), 2)
        mutated[i], mutated[j] = mutated[j], mutated[i]
        return mutated
    
    def select_parents(self, population, fitness_scores):
        """Tournament selection"""
        tournament_size = 3
        selected = []
        
        for _ in range(2):
            tournament = random.sample(list(zip(population, fitness_scores)), tournament_size)
            winner = min(tournament, key=lambda x: x[1])
            selected.append(winner[0])
        
        return selected
    
    def evolve(self, progress_callback=None):
        """Main evolution loop"""
        # Initialize population
        population = [self.create_individual() for _ in range(self.population_size)]
        
        for generation in range(self.generations):
            # Calculate fitness
            fitness_scores = [self.calculate_fitness(ind) for ind in population]
            
            # Track history
            best_fitness = min(fitness_scores)
            avg_fitness = np.mean(fitness_scores)
            self.history['best_fitness'].append(best_fitness)
            self.history['avg_fitness'].append(avg_fitness)
            
            # Call progress callback if provided
            if progress_callback:
                progress_callback(generation + 1, self.generations, best_fitness, avg_fitness)
            
            # Create new population
            new_population = []
            
            # Elitism: keep best individual
            best_idx = np.argmin(fitness_scores)
            new_population.append(population[best_idx].copy())
            
            # Generate offspring
            while len(new_population) < self.population_size:
                # Select parents
                parent1, parent2 = self.select_parents(population, fitness_scores)
                
                # Crossover
                child1, child2 = self.crossover(parent1, parent2)
                
                # Mutate
                child1 = self.mutate(child1)
                child2 = self.mutate(child2)
                
                new_population.extend([child1, child2])
            
            # Trim to population size
            population = new_population[:self.population_size]
        
        # Return best solution
        final_fitness = [self.calculate_fitness(ind) for ind in population]
        best_idx = np.argmin(final_fitness)
        return population[best_idx], self.history
    
    def solve(self, progress_callback=None):
        """Solve the VRP problem"""
        best_route, history = self.evolve(progress_callback)
        return best_route, history

