"""
Configuration settings for AI Logistics & Route Optimization application
"""

# Map settings
DEFAULT_MAP_CENTER = [40.7128, -74.0060]  # New York City coordinates
DEFAULT_ZOOM = 12

# Cost parameters
FUEL_COST_PER_LITER = 1.2  # USD per liter
FUEL_CONSUMPTION_PER_KM = 0.08  # liters per km (average vehicle)
DRIVER_COST_PER_HOUR = 25.0  # USD per hour
VEHICLE_MAINTENANCE_PER_KM = 0.15  # USD per km

# Carbon footprint
CO2_EMISSIONS_PER_LITER_FUEL = 2.31  # kg CO2 per liter of fuel
AVERAGE_SPEED_KMH = 50  # km/h average speed

# Algorithm parameters
GA_POPULATION_SIZE = 100
GA_MUTATION_RATE = 0.1
GA_CROSSOVER_RATE = 0.8
GA_GENERATIONS = 200

RL_LEARNING_RATE = 0.1
RL_DISCOUNT_FACTOR = 0.95
RL_EPSILON_START = 1.0
RL_EPSILON_MIN = 0.01
RL_EPSILON_DECAY = 0.995

CLUSTERING_MAX_K = 15
CLUSTERING_MIN_K = 2

# Time windows
DEFAULT_TIME_WINDOW_START = 9  # 9 AM
DEFAULT_TIME_WINDOW_END = 17  # 5 PM
DELIVERY_TIME_MINUTES = 10  # Average time per delivery stop

# Vehicle types
VEHICLE_TYPES = {
    'small_van': {'capacity_kg': 500, 'capacity_vol': 5, 'fuel_efficiency': 0.07},
    'medium_truck': {'capacity_kg': 2000, 'capacity_vol': 15, 'fuel_efficiency': 0.10},
    'large_truck': {'capacity_kg': 5000, 'capacity_vol': 30, 'fuel_efficiency': 0.12}
}

# SLA parameters
STANDARD_SLA_HOURS = 24
PRIORITY_SLA_HOURS = 4
EXPRESS_SLA_HOURS = 1
SLA_VIOLATION_COST = 50.0  # USD per violation

# Traffic impact multipliers
TRAFFIC_MULTIPLIERS = {
    'light': 1.0,
    'moderate': 1.3,
    'heavy': 1.7,
    'severe': 2.2
}

# Weather impact multipliers
WEATHER_MULTIPLIERS = {
    'clear': 1.0,
    'cloudy': 1.1,
    'rain': 1.4,
    'snow': 2.0,
    'storm': 2.5
}

