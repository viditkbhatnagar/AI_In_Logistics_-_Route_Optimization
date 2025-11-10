"""
KPI calculations and metrics for logistics operations
"""

import pandas as pd
import numpy as np
from utils.carbon_calculator import calculate_fleet_emissions


def calculate_fleet_utilization(vehicles_df, routes_info):
    """
    Calculate fleet utilization metrics
    
    Returns:
    - Average weight utilization
    - Average volume utilization
    - Average time utilization
    - Number of vehicles used
    """
    total_weight_util = 0
    total_volume_util = 0
    total_time_util = 0
    vehicles_used = 0
    
    for vehicle_id, metrics in routes_info.items():
        if metrics.get('n_deliveries', 0) > 0:
            total_weight_util += metrics.get('weight_utilization', 0)
            total_volume_util += metrics.get('volume_utilization', 0)
            # Time utilization (assuming 8 hour work day)
            time_util = min(metrics.get('total_time', 0) / 8.0, 1.0)
            total_time_util += time_util
            vehicles_used += 1
    
    n_vehicles = len(routes_info) if routes_info else 1
    
    return {
        'average_weight_utilization': total_weight_util / n_vehicles if n_vehicles > 0 else 0,
        'average_volume_utilization': total_volume_util / n_vehicles if n_vehicles > 0 else 0,
        'average_time_utilization': total_time_util / n_vehicles if n_vehicles > 0 else 0,
        'vehicles_used': vehicles_used,
        'total_vehicles': len(vehicles_df)
    }


def calculate_sla_metrics(routes_info, orders_df):
    """
    Calculate SLA compliance metrics
    
    Returns:
    - Overall SLA compliance rate
    - Number of SLA violations
    - Violations by priority level
    """
    total_deliveries = 0
    total_violations = 0
    violations_by_priority = {'standard': 0, 'priority': 0, 'express': 0}
    deliveries_by_priority = {'standard': 0, 'priority': 0, 'express': 0}
    
    for vehicle_id, metrics in routes_info.items():
        total_deliveries += metrics.get('n_deliveries', 0)
        total_violations += metrics.get('sla_violations', 0)
    
    # Calculate violations by priority (simplified - would need route details)
    for _, order in orders_df.iterrows():
        deliveries_by_priority[order['priority']] = deliveries_by_priority.get(order['priority'], 0) + 1
    
    sla_compliance_rate = 1 - (total_violations / total_deliveries) if total_deliveries > 0 else 1.0
    
    return {
        'sla_compliance_rate': sla_compliance_rate,
        'total_violations': total_violations,
        'total_deliveries': total_deliveries,
        'violations_by_priority': violations_by_priority,
        'deliveries_by_priority': deliveries_by_priority
    }


def calculate_cost_metrics(routes_info):
    """
    Calculate cost-related metrics
    
    Returns:
    - Total cost
    - Cost per delivery
    - Cost per kilometer
    - Cost breakdown
    """
    total_cost = 0
    total_distance = 0
    total_deliveries = 0
    cost_breakdown = {
        'fuel': 0,
        'driver': 0,
        'maintenance': 0,
        'vehicle': 0
    }
    
    for vehicle_id, metrics in routes_info.items():
        total_cost += metrics.get('total_cost', 0)
        total_distance += metrics.get('total_distance', 0)
        total_deliveries += metrics.get('n_deliveries', 0)
        
        cost_breakdown['fuel'] += metrics.get('fuel_cost', 0)
        cost_breakdown['driver'] += metrics.get('driver_cost', 0)
        cost_breakdown['maintenance'] += metrics.get('maintenance_cost', 0)
        cost_breakdown['vehicle'] += metrics.get('vehicle_cost', 0)
    
    return {
        'total_cost': total_cost,
        'cost_per_delivery': total_cost / total_deliveries if total_deliveries > 0 else 0,
        'cost_per_km': total_cost / total_distance if total_distance > 0 else 0,
        'total_distance': total_distance,
        'total_deliveries': total_deliveries,
        'cost_breakdown': cost_breakdown
    }


def calculate_efficiency_metrics(routes_info):
    """
    Calculate efficiency metrics
    
    Returns:
    - Average deliveries per hour
    - Average distance per delivery
    - Average time per delivery
    """
    total_deliveries = 0
    total_time = 0
    total_distance = 0
    
    for vehicle_id, metrics in routes_info.items():
        total_deliveries += metrics.get('n_deliveries', 0)
        total_time += metrics.get('total_time', 0)
        total_distance += metrics.get('total_distance', 0)
    
    return {
        'deliveries_per_hour': total_deliveries / total_time if total_time > 0 else 0,
        'distance_per_delivery': total_distance / total_deliveries if total_deliveries > 0 else 0,
        'time_per_delivery': total_time / total_deliveries if total_deliveries > 0 else 0,
        'total_deliveries': total_deliveries,
        'total_time': total_time,
        'total_distance': total_distance
    }


def calculate_all_metrics(routes_info, vehicles_df, orders_df):
    """Calculate all KPIs"""
    fleet_util = calculate_fleet_utilization(vehicles_df, routes_info)
    sla_metrics = calculate_sla_metrics(routes_info, orders_df)
    cost_metrics = calculate_cost_metrics(routes_info)
    efficiency_metrics = calculate_efficiency_metrics(routes_info)
    emissions = calculate_fleet_emissions(routes_info, vehicles_df)
    
    return {
        'fleet_utilization': fleet_util,
        'sla_metrics': sla_metrics,
        'cost_metrics': cost_metrics,
        'efficiency_metrics': efficiency_metrics,
        'total_emissions_kg': emissions
    }

