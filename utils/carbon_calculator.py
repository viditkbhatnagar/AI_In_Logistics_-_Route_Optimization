"""
Carbon footprint calculator for logistics operations
"""

from config import CO2_EMISSIONS_PER_LITER_FUEL


def calculate_carbon_emissions(distance_km, fuel_efficiency_l_per_km):
    """
    Calculate CO2 emissions for a given distance
    
    Parameters:
    - distance_km: distance in kilometers
    - fuel_efficiency_l_per_km: fuel efficiency in liters per kilometer
    
    Returns:
    - CO2 emissions in kg
    """
    fuel_consumed = distance_km * fuel_efficiency_l_per_km
    co2_emissions = fuel_consumed * CO2_EMISSIONS_PER_LITER_FUEL
    return co2_emissions


def calculate_fleet_emissions(routes_info, vehicles_df):
    """
    Calculate total carbon emissions for a fleet
    
    Parameters:
    - routes_info: dict mapping vehicle_id to route metrics
    - vehicles_df: DataFrame with vehicle information
    
    Returns:
    - Total CO2 emissions in kg
    """
    total_emissions = 0
    
    for vehicle_id, metrics in routes_info.items():
        vehicle = vehicles_df[vehicles_df['vehicle_id'] == vehicle_id]
        if len(vehicle) > 0:
            vehicle = vehicle.iloc[0]
            emissions = calculate_carbon_emissions(
                metrics['total_distance'],
                vehicle['fuel_efficiency_l_per_km']
            )
            total_emissions += emissions
    
    return total_emissions


def calculate_emissions_reduction(baseline_emissions, optimized_emissions):
    """Calculate emissions reduction percentage"""
    if baseline_emissions == 0:
        return 0
    reduction = ((baseline_emissions - optimized_emissions) / baseline_emissions) * 100
    return max(0, reduction)

