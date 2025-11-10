"""
Comprehensive synthetic data generator for logistics and route optimization scenarios.
Generates realistic datasets including deliveries, vehicles, traffic, weather, and demand data.
"""

import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
import json
import os
from config import (
    VEHICLE_TYPES, DEFAULT_TIME_WINDOW_START, DEFAULT_TIME_WINDOW_END,
    DELIVERY_TIME_MINUTES, STANDARD_SLA_HOURS, PRIORITY_SLA_HOURS, EXPRESS_SLA_HOURS
)


class LogisticsDataGenerator:
    """Generate realistic synthetic logistics data"""
    
    def __init__(self, seed=42):
        np.random.seed(seed)
        random.seed(seed)
        self.seed = seed
    
    def generate_delivery_locations(self, n_deliveries, center_lat=40.7128, center_lon=-74.0060, 
                                   radius_km=20, city_name="New York"):
        """Generate delivery locations with coordinates"""
        locations = []
        
        # Generate realistic coordinates around center point
        for i in range(n_deliveries):
            # Use normal distribution for more realistic clustering
            lat_offset = np.random.normal(0, radius_km / 111)  # ~111 km per degree
            lon_offset = np.random.normal(0, radius_km / (111 * np.cos(np.radians(center_lat))))
            
            lat = center_lat + lat_offset
            lon = center_lon + lon_offset
            
            # Generate realistic address
            street_num = random.randint(1, 9999)
            street_names = ['Main St', 'Park Ave', 'Broadway', '5th Ave', 'Lexington Ave', 
                           'Madison Ave', 'Washington St', 'Market St', 'Oak St', 'Elm St']
            street = random.choice(street_names)
            address = f"{street_num} {street}, {city_name}"
            
            locations.append({
                'delivery_id': f'DEL_{i+1:04d}',
                'address': address,
                'latitude': round(lat, 6),
                'longitude': round(lon, 6),
                'city': city_name
            })
        
        return pd.DataFrame(locations)
    
    def generate_customer_orders(self, delivery_locations, order_date=None):
        """Generate customer orders with realistic attributes"""
        if order_date is None:
            order_date = datetime.now().date()
        
        orders = []
        priority_levels = ['standard', 'priority', 'express']
        priority_weights = [0.7, 0.25, 0.05]  # Most orders are standard
        
        for idx, loc in delivery_locations.iterrows():
            priority = np.random.choice(priority_levels, p=priority_weights)
            
            # Weight and volume based on priority (express tends to be lighter)
            if priority == 'express':
                weight = np.random.uniform(0.5, 5.0)
                volume = np.random.uniform(0.1, 1.0)
            elif priority == 'priority':
                weight = np.random.uniform(2.0, 15.0)
                volume = np.random.uniform(0.5, 3.0)
            else:
                weight = np.random.uniform(5.0, 30.0)
                volume = np.random.uniform(1.0, 8.0)
            
            # Determine SLA hours based on priority
            if priority == 'express':
                sla_hours = EXPRESS_SLA_HOURS
            elif priority == 'priority':
                sla_hours = PRIORITY_SLA_HOURS
            else:
                sla_hours = STANDARD_SLA_HOURS
            
            # Generate time window (some customers have specific windows)
            has_time_window = np.random.random() < 0.4  # 40% have specific windows
            if has_time_window:
                window_start = random.randint(9, 14)
                window_end = random.randint(window_start + 2, 18)
            else:
                window_start = DEFAULT_TIME_WINDOW_START
                window_end = DEFAULT_TIME_WINDOW_END
            
            # Order timestamp
            order_time = datetime.combine(order_date, datetime.min.time()) + \
                        timedelta(hours=random.randint(0, 6))  # Orders come in early morning
            
            orders.append({
                'order_id': f'ORD_{idx+1:05d}',
                'delivery_id': loc['delivery_id'],
                'customer_name': f'Customer_{idx+1}',
                'priority': priority,
                'weight_kg': round(weight, 2),
                'volume_m3': round(volume, 2),
                'order_timestamp': order_time,
                'sla_hours': sla_hours,
                'time_window_start': window_start,
                'time_window_end': window_end,
                'special_requirements': random.choice(['none', 'fragile', 'refrigerated', 'signature_required', 'none']),
                'estimated_delivery_time_minutes': DELIVERY_TIME_MINUTES + random.randint(-5, 10)
            })
        
        return pd.DataFrame(orders)
    
    def generate_vehicle_fleet(self, n_vehicles, depot_locations):
        """Generate vehicle fleet with realistic attributes"""
        vehicles = []
        vehicle_type_list = list(VEHICLE_TYPES.keys())
        
        for i in range(n_vehicles):
            # Assign vehicle type with distribution
            if i < n_vehicles * 0.3:
                v_type = 'small_van'
            elif i < n_vehicles * 0.7:
                v_type = 'medium_truck'
            else:
                v_type = 'large_truck'
            
            # Assign depot
            depot = random.choice(depot_locations)
            
            vehicles.append({
                'vehicle_id': f'VEH_{i+1:03d}',
                'vehicle_type': v_type,
                'capacity_kg': VEHICLE_TYPES[v_type]['capacity_kg'],
                'capacity_vol': VEHICLE_TYPES[v_type]['capacity_vol'],
                'fuel_efficiency_l_per_km': VEHICLE_TYPES[v_type]['fuel_efficiency'],
                'depot_latitude': depot['latitude'],
                'depot_longitude': depot['longitude'],
                'depot_name': depot['name'],
                'cost_per_km': round(np.random.uniform(0.8, 1.5), 2),
                'driver_name': f'Driver_{i+1}',
                'available_from': '08:00',
                'available_until': '20:00'
            })
        
        return pd.DataFrame(vehicles)
    
    def generate_traffic_data(self, n_hours=24, date=None):
        """Generate realistic traffic patterns"""
        if date is None:
            date = datetime.now().date()
        
        traffic_data = []
        hours = list(range(24))
        
        # Traffic patterns: peak hours have higher congestion
        for hour in hours:
            # Morning rush: 7-9 AM, Evening rush: 5-7 PM
            if 7 <= hour <= 9:
                base_congestion = np.random.uniform(0.6, 0.9)
                traffic_level = np.random.choice(['heavy', 'severe'], p=[0.7, 0.3])
            elif 17 <= hour <= 19:
                base_congestion = np.random.uniform(0.5, 0.8)
                traffic_level = np.random.choice(['moderate', 'heavy'], p=[0.6, 0.4])
            elif 10 <= hour <= 16:
                base_congestion = np.random.uniform(0.2, 0.5)
                traffic_level = np.random.choice(['light', 'moderate'], p=[0.7, 0.3])
            else:  # Night/early morning
                base_congestion = np.random.uniform(0.1, 0.3)
                traffic_level = 'light'
            
            # Speed reduction factor
            speed_multiplier = 1 - base_congestion * 0.6
            
            traffic_data.append({
                'datetime': datetime.combine(date, datetime.min.time()) + timedelta(hours=hour),
                'hour': hour,
                'traffic_level': traffic_level,
                'congestion_factor': round(base_congestion, 3),
                'speed_multiplier': round(speed_multiplier, 3),
                'average_speed_kmh': round(50 * speed_multiplier, 1)
            })
        
        return pd.DataFrame(traffic_data)
    
    def generate_weather_data(self, n_days=7, start_date=None):
        """Generate realistic weather data"""
        if start_date is None:
            start_date = datetime.now().date()
        
        weather_data = []
        weather_conditions = ['clear', 'cloudy', 'rain', 'snow', 'storm']
        weather_weights = [0.4, 0.3, 0.2, 0.08, 0.02]  # Clear is most common
        
        for day in range(n_days):
            date = start_date + timedelta(days=day)
            condition = np.random.choice(weather_conditions, p=weather_weights)
            
            # Temperature varies by season (simplified)
            base_temp = 20  # Celsius
            temp = base_temp + np.random.normal(0, 8)
            
            # Precipitation based on condition
            if condition in ['rain', 'snow', 'storm']:
                precipitation = np.random.uniform(2, 15)
            else:
                precipitation = 0
            
            # Wind speed
            wind_speed = np.random.uniform(5, 25)
            if condition == 'storm':
                wind_speed = np.random.uniform(30, 50)
            
            weather_data.append({
                'date': date,
                'condition': condition,
                'temperature_c': round(temp, 1),
                'precipitation_mm': round(precipitation, 1),
                'wind_speed_kmh': round(wind_speed, 1),
                'humidity_percent': round(np.random.uniform(40, 90), 1)
            })
        
        return pd.DataFrame(weather_data)
    
    def generate_demand_forecast(self, n_days=30, start_date=None):
        """Generate demand forecast data"""
        if start_date is None:
            start_date = datetime.now().date()
        
        demand_data = []
        
        # Weekly pattern: higher demand on weekdays
        for day in range(n_days):
            date = start_date + timedelta(days=day)
            day_of_week = date.weekday()
            
            # Base demand
            if day_of_week < 5:  # Weekday
                base_demand = np.random.uniform(80, 120)
            else:  # Weekend
                base_demand = np.random.uniform(50, 80)
            
            # Add some trend and seasonality
            trend = day * 0.1
            seasonal = 10 * np.sin(2 * np.pi * day / 7)  # Weekly pattern
            
            forecasted_demand = max(10, base_demand + trend + seasonal + np.random.normal(0, 10))
            
            demand_data.append({
                'date': date,
                'day_of_week': date.strftime('%A'),
                'forecasted_orders': int(round(forecasted_demand)),
                'confidence_level': round(np.random.uniform(0.75, 0.95), 2)
            })
        
        return pd.DataFrame(demand_data)
    
    def generate_depot_locations(self, n_depots, center_lat=40.7128, center_lon=-74.0060):
        """Generate depot/warehouse locations"""
        depots = []
        
        for i in range(n_depots):
            # Spread depots around the area
            angle = 2 * np.pi * i / n_depots
            radius = np.random.uniform(5, 15)  # km
            
            lat = center_lat + (radius / 111) * np.cos(angle)
            lon = center_lon + (radius / (111 * np.cos(np.radians(center_lat)))) * np.sin(angle)
            
            depots.append({
                'depot_id': f'DEPOT_{i+1}',
                'name': f'Warehouse {i+1}',
                'latitude': round(lat, 6),
                'longitude': round(lon, 6),
                'capacity_vehicles': np.random.randint(10, 50)
            })
        
        return depots
    
    def generate_complete_dataset(self, n_deliveries=200, n_vehicles=15, n_depots=2,
                                 scenario_name="medium_city", city_name="New York"):
        """Generate a complete logistics dataset"""
        print(f"Generating dataset: {scenario_name}")
        
        # Generate depots
        depots = self.generate_depot_locations(n_depots)
        
        # Generate delivery locations
        delivery_locations = self.generate_delivery_locations(n_deliveries, city_name=city_name)
        
        # Generate customer orders
        orders = self.generate_customer_orders(delivery_locations)
        
        # Generate vehicle fleet
        vehicles = self.generate_vehicle_fleet(n_vehicles, depots)
        
        # Generate traffic data (24 hours)
        traffic = self.generate_traffic_data()
        
        # Generate weather data (7 days)
        weather = self.generate_weather_data()
        
        # Generate demand forecast (30 days)
        demand_forecast = self.generate_demand_forecast()
        
        dataset = {
            'scenario_name': scenario_name,
            'generation_date': datetime.now().isoformat(),
            'delivery_locations': delivery_locations,
            'orders': orders,
            'vehicles': vehicles,
            'depots': pd.DataFrame(depots),
            'traffic': traffic,
            'weather': weather,
            'demand_forecast': demand_forecast,
            'metadata': {
                'n_deliveries': n_deliveries,
                'n_vehicles': n_vehicles,
                'n_depots': n_depots,
                'city': city_name
            }
        }
        
        return dataset
    
    def save_dataset(self, dataset, output_dir='data/pre_generated'):
        """Save dataset to files"""
        os.makedirs(output_dir, exist_ok=True)
        scenario_name = dataset['scenario_name']
        
        # Save each component as CSV
        dataset['delivery_locations'].to_csv(
            f'{output_dir}/{scenario_name}_delivery_locations.csv', index=False
        )
        dataset['orders'].to_csv(
            f'{output_dir}/{scenario_name}_orders.csv', index=False
        )
        dataset['vehicles'].to_csv(
            f'{output_dir}/{scenario_name}_vehicles.csv', index=False
        )
        dataset['depots'].to_csv(
            f'{output_dir}/{scenario_name}_depots.csv', index=False
        )
        dataset['traffic'].to_csv(
            f'{output_dir}/{scenario_name}_traffic.csv', index=False
        )
        dataset['weather'].to_csv(
            f'{output_dir}/{scenario_name}_weather.csv', index=False
        )
        dataset['demand_forecast'].to_csv(
            f'{output_dir}/{scenario_name}_demand_forecast.csv', index=False
        )
        
        # Save metadata as JSON
        with open(f'{output_dir}/{scenario_name}_metadata.json', 'w') as f:
            json.dump(dataset['metadata'], f, indent=2, default=str)
        
        print(f"Dataset saved to {output_dir}/{scenario_name}_*")


def generate_premade_datasets():
    """Generate the 5 pre-made datasets"""
    generator = LogisticsDataGenerator(seed=42)
    
    scenarios = [
        {
            'name': 'small_urban',
            'n_deliveries': 50,
            'n_vehicles': 5,
            'n_depots': 1,
            'city': 'Boston'
        },
        {
            'name': 'medium_city',
            'n_deliveries': 200,
            'n_vehicles': 15,
            'n_depots': 2,
            'city': 'Chicago'
        },
        {
            'name': 'large_metropolitan',
            'n_deliveries': 500,
            'n_vehicles': 30,
            'n_depots': 3,
            'city': 'Los Angeles'
        },
        {
            'name': 'multi_city_network',
            'n_deliveries': 1000,
            'n_vehicles': 50,
            'n_depots': 5,
            'city': 'New York'
        },
        {
            'name': 'peak_season',
            'n_deliveries': 800,
            'n_vehicles': 25,
            'n_depots': 3,
            'city': 'San Francisco'
        }
    ]
    
    for scenario in scenarios:
        dataset = generator.generate_complete_dataset(
            n_deliveries=scenario['n_deliveries'],
            n_vehicles=scenario['n_vehicles'],
            n_depots=scenario['n_depots'],
            scenario_name=scenario['name'],
            city_name=scenario['city']
        )
        generator.save_dataset(dataset)
        print(f"✓ Generated {scenario['name']} dataset")


if __name__ == '__main__':
    generate_premade_datasets()

